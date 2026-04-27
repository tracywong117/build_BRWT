#include "brwt_builders.hpp"

#include <omp.h>
#include <queue>
#include <sstream>
#include <fstream>
#include <progress_bar.hpp>

#include "common/algorithms.hpp"
#include "common/logger.hpp"
#include "common/utils/file_utils.hpp"
#include "common/vectors/vector_algorithm.hpp"
#include "common/unix_tools.hpp"


namespace mtg {
namespace annot {
namespace matrix {

using mtg::common::logger;

namespace {
    std::string format_mem(size_t bytes) {
        return fmt::format("{:.2f} GB", static_cast<double>(bytes) / 1e9);
    }
    void log_mem() {
        logger->info("[MEM] Process RSS: {}", format_mem(get_curr_RSS()));
    }

    struct NodeCursor {
        uint64_t current_pos;
        uint64_t rank;
        uint64_t num_ones;
        size_t child_idx;
        const bit_vector* vec;

        bool next() {
            if (rank >= num_ones) return false;
            rank++;
            current_pos = vec->select1(rank);
            return true;
        }

        bool operator>(const NodeCursor& other) const {
            return current_pos > other.current_pos;
        }
    };
}

BRWT BRWTBottomUpBuilder::load_leaf_from_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        logger->error("Failed to open leaf file: {}", path);
        exit(1);
    }
    auto sdvec_ptr = std::make_shared<sdsl::sd_vector<>>();
    sdsl::load(*sdvec_ptr, in);
    
    BRWT node;
    node.assignments_ = RangePartition(Partition(1, { 0 }));
    
    sdsl::sd_vector<>::rank_1_type rank1(sdvec_ptr.get());
    uint64_t num_ones = rank1(sdvec_ptr->size());
    uint64_t size = sdvec_ptr->size();

    node.nonzero_rows_ = std::make_unique<bit_vector_smart>(
        [sdvec_ptr, num_ones](const auto &callback) {
            sdsl::sd_vector<>::select_1_type select_1(sdvec_ptr.get());
            for (uint64_t r = 1; r <= num_ones; ++r) {
                callback(select_1(r));
            }
        },
        size,
        num_ones
    );
    return node;
}


// ─── query_nodes ──────────────────────────────────────────────────────────────
//
// Query rows from a BRWT stored as a nodes folder.  Works without assembling
// the tree into a single .brwt file, loading individual node files on demand.
//
// Disk layout expected in node_dir:
//   <root_id>    — root's *union* bit-vector (actual row positions in matrix)
//   node_<id>    — every other node's *sub-index* (positions within parent rows)
//
// Algorithm: propagate a list of (rank_in_this_node, query_index) pairs downward.
// At each internal node, each child's node_<id> sub-index is queried to filter
// and re-rank before descending.  Reaching a leaf means that column is set.

std::vector<std::vector<uint64_t>>
BRWTBottomUpBuilder::query_nodes(const std::vector<uint64_t>& row_ids,
                                 const std::vector<std::vector<uint64_t>>& linkage,
                                 const std::filesystem::path& node_dir,
                                 bool log_stats) {
    using Item  = std::pair<uint64_t, size_t>;  // (rank_in_this_node, query_index)
    using Items = std::vector<Item>;

    std::vector<std::vector<uint64_t>> results(row_ids.size());
    if (linkage.empty() || row_ids.empty()) return results;

    uint64_t root_id = static_cast<uint64_t>(linkage.size()) - 1;

    // Counters for the stats summary (only tracked when log_stats=true).
    size_t nodes_loaded = 0; // total node files loaded from disk
    size_t total_hits   = 0; // total (query, column) pairs found

    // Load root — stored as `<root_id>`, contains the union bit-vector.
    BRWT root;
    {
        std::filesystem::path rp = node_dir / std::to_string(root_id);
        std::ifstream in(rp, std::ios::binary);
        if (!in.is_open() || !root.load(in))
            throw std::runtime_error("query-nodes: cannot load root node: " + rp.string());
        ++nodes_loaded;
    }

    // Map each query row ID through root's union bit-vector -> 0-indexed dense rank.
    // conditional_rank1(pos) returns 0 when bit is unset, else the 1-based rank.
    Items root_items;
    root_items.reserve(row_ids.size());
    for (size_t qi = 0; qi < row_ids.size(); ++qi) {
        uint64_t r = row_ids[qi];
        if (r >= root.num_rows()) {
            logger->warn("query-nodes: row {} out of range (tree has {} rows), skipping.",
                         r, root.num_rows());
            continue;
        }
        if (uint64_t rank = root.nonzero_rows_->conditional_rank1(r))
            root_items.push_back({rank - 1, qi}); // convert to 0-indexed
        // rows absent from root have no set bit anywhere -> leave their result empty
    }

    // Recursive descent implemented as a std::function so it can call itself.
    // Being defined inside a BRWTBottomUpBuilder member function gives us
    // full friend access to BRWT::nonzero_rows_ throughout this scope.
    std::function<void(uint64_t, const Items&)> descend;
    descend = [&](uint64_t node_id, const Items& items) {
        if (items.empty()) return;

        for (size_t k = 0; k < linkage[node_id].size(); ++k) {
            uint64_t child_id   = linkage[node_id][k];
            bool child_is_leaf  = linkage[child_id].empty();

            // Every non-root node (internal or leaf) has a node_<id> sub-index file
            // written by dump_node_final() during the build step.
            BRWT child;
            {
                std::filesystem::path cp = node_dir / ("node_" + std::to_string(child_id));
                std::ifstream in(cp, std::ios::binary);
                if (!in.is_open() || !child.load(in))
                    throw std::runtime_error("query-nodes: cannot load node: " + cp.string());
                ++nodes_loaded;
            }

            Items child_items;
            for (auto& [rank, qi] : items) {
                if (uint64_t r = child.nonzero_rows_->conditional_rank1(rank)) {
                    if (child_is_leaf) {
                        results[qi].push_back(child_id); // column hit
                        ++total_hits;
                    } else {
                        child_items.push_back({r - 1, qi}); // re-rank for next level
                    }
                }
            }

            if (!child_is_leaf)
                descend(child_id, child_items);
        }
    };

    descend(root_id, root_items);

    if (log_stats) {
        uint64_t root_rows   = root.num_rows();
        uint64_t root_ones   = root.nonzero_rows_ ? root.nonzero_rows_->num_set_bits() : 0;
        size_t queries_with_hit = 0;
        for (const auto& r : results) if (!r.empty()) ++queries_with_hit;
        logger->info("[query-nodes stats]"
                     " rows_queried={}"
                     " rows_with_hit={}"
                     " total_column_hits={}"
                     " nodes_loaded={}"
                     " root_rows={}"
                     " root_set_bits={}",
                     row_ids.size(), queries_with_hit, total_hits,
                     nodes_loaded, root_rows, root_ones);
    }

    return results;
}

// ─────────────────────────────────────────────────────────────────────────────


BRWTBottomUpBuilder::Partitioner
BRWTBottomUpBuilder::get_basic_partitioner(size_t arity) {
    assert(arity > 0u);

    return [arity](const VectorPtrs &vectors) {
        if (!vectors.size())
            return Partition(0);

        Partition partition((vectors.size() - 1) / arity + 1);
        for (size_t i = 0 ; i < vectors.size(); ++i) {
            partition[i / arity].push_back(i);
        }
        return partition;
    };
}

BRWT BRWTBottomUpBuilder::concatenate(std::vector<BRWT>&& submatrices,
                                      sdsl::bit_vector *buffer,
                                      ThreadPool &thread_pool,
                                      const std::filesystem::path &node_tmp_path) {
    /*
    concatenate() called
       │
       ├── density > 0.5 AND buffer available?
       │         YES → Strategy C: RAM-based Bitmap-OR
       │
       └── density ≤ 0.5 (or no buffer)
                 │
                 ├── density > 0.015 AND tmp_path set?
                 │         YES → Strategy B: Disk-based Index
                 │
                 └── else
                           → Strategy A: RAM-based Index

    build() loop for node i:
    │
    ├── concatenate()  ← there are 3 strategies
    │       │
    │       ├── Strategy A: pure RAM vectors → returns BRWT in memory
    │       ├── Strategy B: writes scratch files to <tmpDir>/{i}_buf.*
    │       │              reads them back → deletes scratch files → returns BRWT in memory
    │       └── Strategy C: bitwise OR into RAM buffer → returns BRWT in memory
    │
    └── dump_node(node, i)
            │
            └── writes <tmpDir>/i   (the final persistent node file)
    
    BRWT node (parent)
    ├── node.nonzero_rows_          ← parent's bit-vector (union of all children)
    ├── node.child_nodes_[0]        ← child 0's BRWT (with recomputed sub-index)
    ├── node.child_nodes_[1]        ← child 1's BRWT
    └── ...

    dump_node(child[0], id0)
    dump_node(child[1], id1)
    dump_node(parent, i)

    */
    assert(submatrices.size());

    if (submatrices.size() == 1)
        return std::move(submatrices[0]);

    BRWT parent;
    size_t K = submatrices.size();
    uint64_t num_rows = submatrices[0].num_rows();

    uint64_t total_child_ones = 0;
    for (const auto& sm : submatrices) {
        if (sm.nonzero_rows_)
            total_child_ones += sm.nonzero_rows_->num_set_bits();
    }
    double est_density = (double)total_child_ones / num_rows;

    if (est_density > 0.5 && buffer != nullptr && buffer->size() == num_rows) {
        // Strategy C: Bitmap-OR
        sdsl::util::set_to_value(*buffer, 0);
        for (const auto& sm : submatrices) {
            sm.nonzero_rows_->add_to(buffer);
        }
        
        parent.nonzero_rows_ = std::make_unique<bit_vector_smart>(*buffer);
        bit_vector_stat ref_bv(*buffer);

        parent.child_nodes_.resize(K);
        for (size_t k = 0; k < K; ++k) {
            sdsl::bit_vector subindex = generate_subindex(*submatrices[k].nonzero_rows_, ref_bv, thread_pool);
            submatrices[k].nonzero_rows_ = std::make_unique<bit_vector_smallrank>(std::move(subindex));
            parent.child_nodes_[k].reset(new BRWT(std::move(submatrices[k])));
        }
    } else {
        // Strategy A or B: Index-Merge
        std::priority_queue<NodeCursor, std::vector<NodeCursor>, std::greater<NodeCursor>> pq;
        for (size_t k = 0; k < K; ++k) {
            NodeCursor cursor;
            cursor.child_idx = k;
            cursor.vec = submatrices[k].nonzero_rows_.get();
            if (!cursor.vec) continue;
            cursor.num_ones = cursor.vec->num_set_bits();
            cursor.rank = 0;
            if (cursor.next()) pq.push(cursor);
        }

        bool use_disk_buffer = (est_density > 0.015 && !node_tmp_path.empty());
        uint64_t p_rank = 0;

        if (use_disk_buffer) {
            // Strategy B: Disk-Buffered Stream
            std::string p_path = node_tmp_path.string() + ".p.bin";
            std::ofstream p_out(p_path, std::ios::binary);
            std::vector<std::string> c_paths(K);
            std::vector<std::unique_ptr<std::ofstream>> c_outs(K);
            for (size_t k = 0; k < K; ++k) {
                c_paths[k] = node_tmp_path.string() + ".c" + std::to_string(k) + ".bin";
                c_outs[k] = std::make_unique<std::ofstream>(c_paths[k], std::ios::binary);
            }

            while (!pq.empty()) {
                uint64_t min_pos = pq.top().current_pos;
                p_rank++;
                p_out.write((char*)&min_pos, 8);

                while (!pq.empty() && pq.top().current_pos == min_pos) {
                    NodeCursor cursor = pq.top();
                    pq.pop();
                    uint64_t r = p_rank - 1;
                    c_outs[cursor.child_idx]->write((char*)&r, 8);
                    if (cursor.next()) pq.push(cursor);
                }
            }
            p_out.close();
            for (auto& s : c_outs) s->close();

            parent.nonzero_rows_ = std::make_unique<bit_vector_smart>([p_path](const auto& cb) {
                std::ifstream in(p_path, std::ios::binary);
                uint64_t p;
                while (in.read((char*)&p, 8)) cb(p);
            }, num_rows, p_rank);

            parent.child_nodes_.resize(K);
            for (size_t k = 0; k < K; ++k) {
                sdsl::bit_vector sub_bv(p_rank, 0);
                std::ifstream in(c_paths[k], std::ios::binary);
                uint64_t r;
                while (in.read((char*)&r, 8)) sub_bv[r] = 1;
                in.close();
                std::filesystem::remove(c_paths[k]);
                submatrices[k].nonzero_rows_ = std::make_unique<bit_vector_smallrank>(std::move(sub_bv));
                parent.child_nodes_[k].reset(new BRWT(std::move(submatrices[k])));
            }
            std::filesystem::remove(p_path);
        } else {
            // Strategy A: RAM Array
            auto p_ones = std::make_shared<std::vector<uint64_t>>(); 
            std::vector<std::vector<uint64_t>> c_ranks(K);
            while (!pq.empty()) {
                uint64_t min_pos = pq.top().current_pos;
                p_rank++;
                p_ones->push_back(min_pos);
                while (!pq.empty() && pq.top().current_pos == min_pos) {
                    NodeCursor cursor = pq.top();
                    pq.pop();
                    c_ranks[cursor.child_idx].push_back(p_rank - 1); 
                    if (cursor.next()) pq.push(cursor);
                }
            }

            parent.nonzero_rows_ = std::make_unique<bit_vector_smart>([p_ones](const auto& cb) {
                for (uint64_t p : *p_ones) cb(p);
            }, num_rows, p_rank);

            parent.child_nodes_.resize(K);
            for (size_t k = 0; k < K; ++k) {
                sdsl::bit_vector sub_bv(p_rank, 0);
                for (uint64_t r : c_ranks[k]) sub_bv[r] = 1;
                submatrices[k].nonzero_rows_ = std::make_unique<bit_vector_smallrank>(std::move(sub_bv));
                parent.child_nodes_[k].reset(new BRWT(std::move(submatrices[k])));
            }
        }
    }

    uint64_t num_cols = 0;
    Partition part;
    for (const auto& sm : parent.child_nodes_) {
        part.push_back(utils::arange(num_cols, sm->num_columns()));
        num_cols += sm->num_columns();
    }
    parent.assignments_ = RangePartition(std::move(part));

    return parent;
}

BRWT BRWTBottomUpBuilder::build(std::vector<std::unique_ptr<bit_vector>>&& columns,
                                Partitioner partitioner,
                                size_t num_nodes_parallel,
                                size_t num_threads) {
    if (!columns.size()) return BRWT();
    std::vector<BRWT> nodes(columns.size());
    for (size_t i = 0; i < columns.size(); ++i) {
        nodes[i].assignments_ = RangePartition(Partition(1, { 0 }));
        nodes[i].nonzero_rows_ = std::move(columns[i]);
    }
    return BRWT(); 
}

BRWT BRWTBottomUpBuilder::build(
        const std::function<void(const CallColumn &)> &get_columns,
        const std::vector<std::vector<uint64_t>> &linkage,
        const std::filesystem::path &tmp_path,
        const std::vector<std::vector<uint64_t>> &stored_columns,
        const std::vector<std::string> &column_files,
        size_t num_full_nodes,
        size_t num_partial_nodes,
        size_t num_threads,
        std::string *actual_tmp_dir_out,
        bool resume) {

    size_t total_nodes = linkage.size(); // total_nodes = 2 * columns - 1 for a full binary tree
    std::vector<size_t> node_levels(total_nodes, 0); // node_levels[i] = level of node i in the tree (leaves are level 0)
    size_t max_level = 0;
    for (size_t i = 0; i < total_nodes; ++i) {
        for (uint64_t child : linkage[i]) {
            node_levels[i] = std::max(node_levels[i], node_levels[child] + 1);
        }
        max_level = std::max(max_level, node_levels[i]);
    }

    std::filesystem::path actual_tmp_dir;
    if (resume && !tmp_path.empty() && std::filesystem::is_directory(tmp_path)) {
        actual_tmp_dir = tmp_path;
    } else {
        if (!tmp_path.empty() && !std::filesystem::exists(tmp_path)) {
            std::filesystem::create_directories(tmp_path);
        }
        std::string pattern = (tmp_path / "XXXXXX").string();
        char* res = mkdtemp(pattern.data());
        if (!res) {
            logger->error("Failed to create a temporary directory in {}", tmp_path.string());
            exit(1);
        }
        actual_tmp_dir = res;
        logger->info("Created persistent temporary directory: {}", actual_tmp_dir.string());
    }
    if (actual_tmp_dir_out) *actual_tmp_dir_out = actual_tmp_dir.string();

    // dump_node: saves node's union bit-vector to tmpDir/<id>
    // Used for building (loaded by next level as input to concatenate)
    auto dump_node = [&](BRWT& node, uint64_t id) {
        std::string final_name = (actual_tmp_dir/std::to_string(id)).string();
        std::string partial_name = final_name + ".partial";
        
        std::ofstream out(partial_name, std::ios::binary);
        if (!out.is_open()) {
            logger->error("Failed to open temp node file for writing: {}", partial_name);
            exit(1);
        }
        node.child_nodes_.clear();
        node.serialize(out);
        out.close();
        
        std::filesystem::rename(partial_name, final_name);
        node = BRWT();
    };

    // dump_node_final: saves node's sub-index to tmpDir/node_<id>
    // Used for assembly only; written after parent is confirmed saved.
    // Does NOT overwrite the union bit-vector tmpDir/<id> so resume stays safe.
    auto dump_node_final = [&](BRWT& node, uint64_t id) {
        std::string final_name = (actual_tmp_dir / ("node_" + std::to_string(id))).string();
        std::string partial_name = final_name + ".partial";

        std::ofstream out(partial_name, std::ios::binary);
        if (!out.is_open()) {
            logger->error("Failed to open final node file for writing: {}", partial_name);
            exit(1);
        }
        node.child_nodes_.clear();
        node.serialize(out);
        out.close();

        std::filesystem::rename(partial_name, final_name);
        node = BRWT();
    };

    auto get_node_lambda = [&](uint64_t id) {
        if (id < column_files.size()) return load_leaf_from_file(column_files[id]);
        BRWT node;
        auto filename = actual_tmp_dir/std::to_string(id);
        std::unique_ptr<std::ifstream> in = utils::open_ifstream(filename);
        if (!in || !in->is_open()) {
            logger->error("Failed to open temp node file: {}", filename.string());
            exit(1);
        }
        if (!node.load(*in)) {
            logger->error("Failed to load BRWT from temp node file: {}", filename.string());
            exit(1);
        }
        return node;
    };

    /*
    Phase 1: get_columns() — runs to completion before ANY merging
    │
    ├── metadata_only=true:
    │       for each file: callback(i, label, nullptr)
    │           → column_files[i] = path   ← only path stored
    │           → done[i] = true
    │
    └── metadata_only=false:
            for each file: load sd_vector → build bit_vector → callback(i, label, bv)
                → column_files[i] = path   ← path stored
                → num_rows = bv->size()    ← grab row count
                → done[i] = true
                → bv destroyed!            ← immediately freed

    Phase 2: merge loop (level by level)
    │
    │   For each internal node at level lv:
    │       for each child j: get_node_lambda(j)
    │           ├── leaf: load_leaf_from_file(column_files[j])   ← loads NOW from disk
    │           └── internal: load from tmpDir/<j>               ← loads NOW from tmpDir
    │       concatenate(children) → dump_node to tmpDir
    │       children freed from RAM
    */
    uint64_t num_rows = 0;
    std::vector<bool> done(total_nodes, false);
    std::mutex done_mu;
    std::condition_variable done_cond;

    get_columns([&](uint64_t i, std::unique_ptr<bit_vector>&& column) {
        if (column) num_rows = column->size(); // 
        {
            std::unique_lock<std::mutex> lock(done_mu);
            done[i] = true;
        }
        done_cond.notify_all();
    });
    logger->info("[STEP] Building Multi-BRWT with {} nodes ({} columns)", total_nodes, column_files.size());

    if (!num_rows && !column_files.empty()) num_rows = load_leaf_from_file(column_files[0]).num_rows();

    if (num_partial_nodes == 0) {
        size_t total_ram = get_total_RAM();
        if (total_ram == 0) total_ram = (size_t)500 * 1024 * 1024 * 1024;
        num_partial_nodes = total_ram / (6.26 * 1e9 * 2); 
        if (num_partial_nodes == 0) num_partial_nodes = 1;
    }
    if (num_full_nodes == 0) num_full_nodes = num_threads;

    ProgressBar progress_bar(total_nodes, "Building BRWT", std::cerr, !common::get_verbose());
    ThreadPool thread_pool(num_threads, 100'000 * num_threads);

    // A node is "done" if its union bit-vector <id> still exists (built but parent not yet
    // processed), OR its finalized sub-index node_<id> exists (parent already saved and
    // <id> was deleted as a disk-space optimisation).
    auto is_node_done = [&](size_t i) -> bool {
        if (!resume) return false;
        return std::filesystem::exists(actual_tmp_dir / std::to_string(i)) ||
               std::filesystem::exists(actual_tmp_dir / ("node_" + std::to_string(i)));
    };

    for (size_t lv = 1; lv <= max_level; ++lv) {
        std::vector<size_t> level_nodes;
        for (size_t i = 0; i < total_nodes; ++i) if (node_levels[i] == lv) level_nodes.push_back(i);

        // Fast path: if every node at this level is already done (either <id> or node_<id>
        // exists), skip density sampling, buffer allocation, and the parallel loop entirely.
        // This avoids recomputing levels whose <id> files were cleaned up by a previous run.
        if (resume) {
            bool all_done = std::all_of(level_nodes.begin(), level_nodes.end(),
                                        [&](size_t i){ return is_node_done(i); });
            if (all_done) {
                logger->info("[STEP] Level {}/{} - All nodes already complete, skipping.", lv, max_level);
                for (size_t i : level_nodes) {
                    { std::unique_lock<std::mutex> lock(done_mu); done[i] = true; }
                    done_cond.notify_all();
                    // Delete any children's <id> files that may have been recreated
                    // in this resume run before we determined the level was done.
                    for (uint64_t j : linkage[i]) {
                        if (j >= column_files.size()) {
                            std::error_code ec;
                            std::filesystem::remove(actual_tmp_dir / std::to_string(j), ec);
                        }
                    }
                }
                continue; // next level
            }
        }

        // Density sampling — guard against children whose <id> was already deleted
        // (they are done; their node_<id> exists but <id> does not).
        double level_density = 0;
        size_t valid_samples = 0;
        size_t sample_size = std::min(level_nodes.size(), (size_t)10);
        for (size_t s = 0; s < sample_size; ++s) {
            uint64_t child_id = linkage[level_nodes[s]][0];
            // Skip if child's <id> was already removed (it's done, not useful for density)
            if (child_id >= column_files.size() &&
                !std::filesystem::exists(actual_tmp_dir / std::to_string(child_id))) {
                continue;
            }
            BRWT sm = get_node_lambda(child_id);
            if (sm.nonzero_rows_) {
                level_density += (double)sm.nonzero_rows_->num_set_bits() / num_rows;
                ++valid_samples;
            }
        }
        if (valid_samples > 0) level_density /= valid_samples;

        size_t p_nodes = (level_density > 0.1) ? num_partial_nodes : num_full_nodes;

        // Bitmap-OR: if density > 0.5
        // Disk-based Index: density > 0.015
        // RAM-based Index: else
        bool use_bitmap = (level_density > 0.5);
        bool use_disk_buffer = (level_density > 0.015);
        std::string mode;
        if (use_bitmap) mode = "Bitmap";
        else if (use_disk_buffer) mode = "Disk-Index";
        else mode = "RAM-Index";

        logger->info("[STEP] Level {}/{} (Density: {:.2f}%) - Parallel Nodes: {} (Mode: {})",
                 lv, max_level, level_density * 100, p_nodes, mode);
        log_mem();

        // Pre-scan: count how many nodes at this level need computation vs already done.
        size_t nodes_to_compute = 0, children_to_load = 0;
        for (size_t i : level_nodes) {
            if (!is_node_done(i)) {
                ++nodes_to_compute;
                children_to_load += linkage[i].size();
            }
        }
        logger->info("  → {}/{} parent nodes to compute ({} children to load)",
                     nodes_to_compute, level_nodes.size(), children_to_load);

        // Progress milestone tracking (every 10%, if ≥10 nodes need computation).
        // 'computed_nodes' and 'last_milestone_pct' are only accessed inside
        // '#pragma omp critical' below, so no atomics are needed.
        const bool report_progress = (nodes_to_compute >= 10);
        size_t computed_nodes = 0;
        int    last_milestone_pct = 0;
        auto   t_level_start = std::chrono::high_resolution_clock::now();

        std::vector<sdsl::bit_vector> buffers(p_nodes); // Used for Bitmap-OR strategy, pre-allocated to avoid repeated resizing
        if (use_bitmap) {
            for(auto& b : buffers) b = sdsl::bit_vector(num_rows, 0);
        }

        // Each thread processes a node:
        // waits for children to be done, loads them, concatenates, dumps result, marks done
        #pragma omp parallel for num_threads(p_nodes) schedule(dynamic)
        for (size_t idx = 0; idx < level_nodes.size(); ++idx) {
            size_t i = level_nodes[idx];

            // Resume: skip if <id> (union bv) still exists, OR node_<id> (sub-index)
            // exists meaning the node was fully finalized and <id> was already cleaned up.
            if (resume) {
                auto union_path = actual_tmp_dir / std::to_string(i);
                auto final_path = actual_tmp_dir / ("node_" + std::to_string(i));
                bool skip = false;

                if (std::filesystem::exists(union_path)) {
                    BRWT node;
                    std::ifstream in(union_path, std::ios::binary);
                    if (node.load(in)) {
                        skip = true;
                    } else {
                        logger->warn("Node {} corrupted, recomputing...", i);
                        std::filesystem::remove(union_path);
                    }
                } else if (std::filesystem::exists(final_path)) {
                    // <id> was deleted after parent confirmed — node is done
                    skip = true;
                }

                if (skip) {
                    { std::unique_lock<std::mutex> lock(done_mu); done[i] = true; }
                    done_cond.notify_all();
                    for (uint64_t j : linkage[i]) {
                        if (j >= column_files.size()) {
                            std::error_code ec;
                            std::filesystem::remove(actual_tmp_dir / std::to_string(j), ec);
                        }
                    }
                    continue;
                }
            }

            // Wait for children to be done and load them
            std::vector<BRWT> children;
            for (uint64_t j : linkage[i]) {
                std::unique_lock<std::mutex> lock(done_mu);
                done_cond.wait(lock, [&] { return done[j]; });
                children.push_back(get_node_lambda(j));
            }

            // Concatenate the child nodes to build the parent node
            BRWT node = concatenate(std::move(children), 
                                    use_bitmap ? &buffers[omp_get_thread_num()] : nullptr, 
                                    thread_pool, actual_tmp_dir/fmt::format("{}_buf", i));
            
            // Save each child's sub-index to tmpDir/node_<child_id>.
            // This does NOT overwrite tmpDir/<child_id> (the union bit-vector),
            // keeping it intact for resume safety.
            for (size_t r = 0; r < node.child_nodes_.size(); ++r) {
                if (node.child_nodes_[r]) dump_node_final(*node.child_nodes_[r], linkage[i][r]);
            }
            // Save parent's union bit-vector to tmpDir/<id> (atomic rename).
            // Only after this succeeds is it safe to delete children's union bit-vectors.
            dump_node(node, i);

            // Parent is now confirmed saved. Delete children's union bit-vector files
            // (internal nodes only — leaves have no <id> file in tmpDir).
            for (uint64_t j : linkage[i]) {
                if (j >= column_files.size()) { // internal node: has a tmpDir/<j> file
                    std::error_code ec;
                    std::filesystem::remove(actual_tmp_dir / std::to_string(j), ec);
                }
            }

            { std::unique_lock<std::mutex> lock(done_mu); done[i] = true; }
            done_cond.notify_all();
            #pragma omp critical
            {
                ++progress_bar;
                if (report_progress) {
                    ++computed_nodes;
                    int pct = (int)((computed_nodes * 100) / nodes_to_compute);
                    int milestone_pct = (pct / 10) * 10; // round down to nearest 10
                    if (milestone_pct > last_milestone_pct && milestone_pct < 100) {
                        last_milestone_pct = milestone_pct;
                        auto now = std::chrono::high_resolution_clock::now();
                        double elapsed = std::chrono::duration<double>(now - t_level_start).count();
                        double frac    = (double)computed_nodes / nodes_to_compute;
                        double eta     = elapsed / frac * (1.0 - frac);
                        logger->info("  [Progress] {}% ({}/{}) - Elapsed: {:.1f}s - ETA: {:.1f}s",
                                     milestone_pct, computed_nodes, nodes_to_compute,
                                     elapsed, eta);
                    }
                }
            }
        }
    }

    return get_node_lambda(total_nodes - 1);
}

void BRWTBottomUpBuilder::assemble_streaming(std::ostream &out,
                                             const std::vector<std::vector<uint64_t>> &linkage,
                                             const std::vector<std::vector<uint64_t>> &stored_columns,
                                             const std::filesystem::path &tmp_dir,
                                             const std::vector<std::string> &column_files,
                                             bool cleanup_tmp) {
    logger->info("[STEP] Final Recursive Streaming Assembly...");
    std::function<void(uint64_t id)> serialize_recursive = [&](uint64_t id) {
        BRWT node;
        bool is_root = (id == linkage.size() - 1);
        // Root: load union bit-vector from tmpDir/<id>
        // Non-root: load sub-index from tmpDir/node_<id>
        auto filename = is_root ? tmp_dir / std::to_string(id)
                                : tmp_dir / ("node_" + std::to_string(id));
        std::unique_ptr<std::ifstream> in = utils::open_ifstream(filename);
        if (!in || !in->is_open()) {
            logger->error("Assembly: failed to open node file: {}", filename.string());
            exit(1);
        }
        if (!node.load(*in)) {
            logger->error("Assembly: failed to load node from: {}", filename.string());
            exit(1);
        }
        if (is_root) {
            // Root's union bit-vector → convert to smallrank for the final .brwt format
            node.nonzero_rows_ = std::make_unique<bit_vector_smallrank>(node.nonzero_rows_->convert_to<bit_vector_smallrank>());
        }
        node.assignments_.serialize(out);
        node.nonzero_rows_->serialize(out);
        serialize_number(out, linkage[id].size());
        node = BRWT();
        for (uint64_t child_id : linkage[id]) serialize_recursive(child_id);
    };
    serialize_recursive(linkage.size() - 1);
    if (cleanup_tmp) std::filesystem::remove_all(tmp_dir);
}

// ─── merge (monolithic .brwt files) ──────────────────────────────────────────
//
// Loads both BRWT trees fully into RAM, concatenates them under a new 2-child
// root, and serialises the result.  Requires identical num_rows in both trees.

void BRWTBottomUpBuilder::merge(const std::filesystem::path &brwt_a,
                                const std::filesystem::path &brwt_b,
                                const std::filesystem::path &output_prefix,
                                size_t num_threads) {
    logger->info("[merge] Loading Tree A from {}", brwt_a.string());
    BRWT tree_a;
    {
        std::ifstream in(brwt_a, std::ios::binary);
        if (!in.is_open() || !tree_a.load(in))
            throw std::runtime_error("merge: cannot load BRWT A: " + brwt_a.string());
    }

    logger->info("[merge] Loading Tree B from {}", brwt_b.string());
    BRWT tree_b;
    {
        std::ifstream in(brwt_b, std::ios::binary);
        if (!in.is_open() || !tree_b.load(in))
            throw std::runtime_error("merge: cannot load BRWT B: " + brwt_b.string());
    }

    if (tree_a.num_rows() != tree_b.num_rows())
        throw std::runtime_error(
            "merge: row length mismatch: A=" + std::to_string(tree_a.num_rows()) +
            " B=" + std::to_string(tree_b.num_rows()));

    logger->info("[merge] Merging {} cols + {} cols ({} rows)",
                 tree_a.num_columns(), tree_b.num_columns(), tree_a.num_rows());

    // Estimate density to pick concatenate() strategy.
    uint64_t num_rows = tree_a.num_rows();
    uint64_t ones_a = tree_a.nonzero_rows_ ? tree_a.nonzero_rows_->num_set_bits() : 0;
    uint64_t ones_b = tree_b.nonzero_rows_ ? tree_b.nonzero_rows_->num_set_bits() : 0;
    double density = (double)(ones_a + ones_b) / num_rows;

    ThreadPool thread_pool(num_threads, 100'000 * num_threads);
    sdsl::bit_vector buffer;
    sdsl::bit_vector *buf_ptr = nullptr;
    if (density > 0.5) {
        buffer = sdsl::bit_vector(num_rows, 0);
        buf_ptr = &buffer;
    }

    std::vector<BRWT> children;
    children.push_back(std::move(tree_a));
    children.push_back(std::move(tree_b));

    BRWT merged_root = concatenate(std::move(children), buf_ptr, thread_pool, "");

    // The final .brwt format uses bit_vector_smallrank for the root union bv.
    merged_root.nonzero_rows_ = std::make_unique<bit_vector_smallrank>(
        merged_root.nonzero_rows_->convert_to<bit_vector_smallrank>());

    std::filesystem::path out_brwt = output_prefix.string() + ".brwt";
    logger->info("[merge] Writing merged BRWT to {}", out_brwt.string());
    std::ofstream out(out_brwt, std::ios::binary);
    if (!out.is_open())
        throw std::runtime_error("merge: cannot open output: " + out_brwt.string());
    merged_root.serialize(out);
    logger->info("[merge] Done. Merged index has {} columns.", merged_root.num_columns());
}

// ─── merge_nodes (node-folder format) ────────────────────────────────────────
//
// Disk layout expected per index:
//   <root_id>        — root's union bit-vector (BRWT file)
//   node_<id>        — every other node's sub-index (BRWT file)
//
// Algorithm:
//  1. Copy/hard-link all node files from dir_a into output (IDs unchanged).
//  2. Copy all node files from dir_b into output with IDs shifted by offset
//     (offset = linkage_a.size()).
//  3. Load old root_A and old root_B_renumbered from output dir.
//  4. Call concatenate() → new root + recomputed sub-indices for both old roots.
//  5. Write node_<root_A_id> and node_<root_B_renumbered> (their new sub-indices).
//  6. Write <new_root_id> (new root's union bv).
//  7. Build and write merged linkage (variable-arity format).

void BRWTBottomUpBuilder::merge_nodes(const std::filesystem::path &linkage_a_path,
                                      const std::filesystem::path &node_dir_a,
                                      const std::filesystem::path &linkage_b_path,
                                      const std::filesystem::path &node_dir_b,
                                      const std::filesystem::path &output_node_dir,
                                      const std::filesystem::path &output_linkage,
                                      size_t num_threads) {
    // ── 1. Parse linkage matrices ────────────────────────────────────────────
    logger->info("[merge-nodes] Parsing linkage A: {}", linkage_a_path.string());
    auto parse_lm = [](const std::filesystem::path &p) {
        std::ifstream in(p);
        if (!in.is_open()) throw std::runtime_error("merge-nodes: cannot open: " + p.string());
        std::vector<std::vector<uint64_t>> lm;
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            std::istringstream ss(line);
            std::vector<uint64_t> tokens;
            uint64_t v;
            while (ss >> v) tokens.push_back(v);
            if (tokens.size() < 3) continue;
            if (tokens.size() == 4 && tokens[3] > tokens[0] && tokens[3] > tokens[1]) {
                // 4-column build format: c1 c2 num_ones node_id
                uint64_t nid = tokens[3];
                if (nid >= lm.size()) lm.resize(nid + 1);
                lm[nid] = {tokens[0], tokens[1]};
            } else {
                // variable-column format: node_id child1 child2 ...
                uint64_t nid = tokens[0];
                if (nid >= lm.size()) lm.resize(nid + 1);
                lm[nid].assign(tokens.begin() + 1, tokens.end());
            }
        }
        return lm;
    };

    auto linkage_a = parse_lm(linkage_a_path);
    auto linkage_b = parse_lm(linkage_b_path);
    if (linkage_a.empty()) throw std::runtime_error("merge-nodes: linkage A is empty");
    if (linkage_b.empty()) throw std::runtime_error("merge-nodes: linkage B is empty");

    uint64_t root_a_id          = static_cast<uint64_t>(linkage_a.size()) - 1;
    uint64_t offset             = static_cast<uint64_t>(linkage_a.size());
    uint64_t root_b_id_orig     = static_cast<uint64_t>(linkage_b.size()) - 1;
    uint64_t root_b_id_renamed  = root_b_id_orig + offset;
    uint64_t new_root_id        = offset + static_cast<uint64_t>(linkage_b.size());

    logger->info("[merge-nodes] root_A={} root_B_renamed={} new_root={}",
                 root_a_id, root_b_id_renamed, new_root_id);

    // ── 2. Create output directory ───────────────────────────────────────────
    std::filesystem::create_directories(output_node_dir);

    // Helper: try hard-link first, fall back to copy.
    auto link_or_copy = [](const std::filesystem::path &src,
                           const std::filesystem::path &dst) {
        std::error_code ec;
        std::filesystem::create_hard_link(src, dst, ec);
        if (ec) std::filesystem::copy_file(src, dst,
                     std::filesystem::copy_options::overwrite_existing);
    };

    // ── 3. Copy node files from dir_a (IDs unchanged) ───────────────────────
    logger->info("[merge-nodes] Copying/linking nodes from A: {}", node_dir_a.string());
    for (const auto &entry : std::filesystem::directory_iterator(node_dir_a)) {
        if (!entry.is_regular_file()) continue;
        link_or_copy(entry.path(), output_node_dir / entry.path().filename());
    }

    // ── 4. Copy node files from dir_b with renamed IDs ──────────────────────
    logger->info("[merge-nodes] Copying/linking nodes from B: {}", node_dir_b.string());
    for (const auto &entry : std::filesystem::directory_iterator(node_dir_b)) {
        if (!entry.is_regular_file()) continue;
        std::string fname = entry.path().filename().string();
        std::string new_fname;
        if (fname.rfind("node_", 0) == 0) {
            // sub-index file: node_<id>  →  node_<id+offset>
            uint64_t id = std::stoull(fname.substr(5));
            new_fname = "node_" + std::to_string(id + offset);
        } else {
            // root union-bv file: <id>  →  <id+offset>
            uint64_t id = std::stoull(fname);
            new_fname = std::to_string(id + offset);
        }
        link_or_copy(entry.path(), output_node_dir / new_fname);
    }

    // ── 5. Load old roots and re-concatenate ────────────────────────────────
    auto load_brwt = [&](const std::filesystem::path &p) {
        BRWT node;
        std::ifstream in(p, std::ios::binary);
        if (!in.is_open() || !node.load(in))
            throw std::runtime_error("merge-nodes: cannot load node: " + p.string());
        return node;
    };

    BRWT root_a = load_brwt(output_node_dir / std::to_string(root_a_id));
    BRWT root_b = load_brwt(output_node_dir / std::to_string(root_b_id_renamed));

    if (root_a.num_rows() != root_b.num_rows())
        throw std::runtime_error(
            "merge-nodes: row length mismatch: A=" + std::to_string(root_a.num_rows()) +
            " B=" + std::to_string(root_b.num_rows()));

    uint64_t num_rows = root_a.num_rows();
    uint64_t ones_a   = root_a.nonzero_rows_ ? root_a.nonzero_rows_->num_set_bits() : 0;
    uint64_t ones_b   = root_b.nonzero_rows_ ? root_b.nonzero_rows_->num_set_bits() : 0;
    double density = (double)(ones_a + ones_b) / num_rows;

    ThreadPool thread_pool(num_threads, 100'000 * num_threads);
    sdsl::bit_vector buffer;
    sdsl::bit_vector *buf_ptr = nullptr;
    if (density > 0.5) {
        buffer = sdsl::bit_vector(num_rows, 0);
        buf_ptr = &buffer;
    }

    std::vector<BRWT> children;
    children.push_back(std::move(root_a));
    children.push_back(std::move(root_b));

    logger->info("[merge-nodes] Building new root via concatenate() (density={:.2f}%)",
                 density * 100);
    BRWT new_root = concatenate(std::move(children), buf_ptr, thread_pool,
                                output_node_dir / fmt::format("{}_merge_buf", new_root_id));

    // ── 6. Write recomputed sub-indices for the two old roots ────────────────
    // (They are now children of the new root and must have sub-indices relative to it.)
    auto dump_sub_index = [&](BRWT &node, uint64_t id) {
        std::string final_name = (output_node_dir / ("node_" + std::to_string(id))).string();
        std::string partial    = final_name + ".partial";
        std::ofstream out(partial, std::ios::binary);
        if (!out.is_open())
            throw std::runtime_error("merge-nodes: cannot write sub-index for node " +
                                     std::to_string(id));
        node.child_nodes_.clear();
        node.serialize(out);
        out.close();
        std::filesystem::rename(partial, final_name);
        node = BRWT();
    };

    // Delete the old root files in output dir (they are no longer roots);
    // their new sub-index will replace them as node_<id> files.
    std::error_code ec;
    std::filesystem::remove(output_node_dir / std::to_string(root_a_id), ec);
    std::filesystem::remove(output_node_dir / std::to_string(root_b_id_renamed), ec);

    dump_sub_index(*new_root.child_nodes_[0], root_a_id);
    dump_sub_index(*new_root.child_nodes_[1], root_b_id_renamed);

    // ── 7. Write new root's union bit-vector ────────────────────────────────
    {
        std::string final_name = (output_node_dir / std::to_string(new_root_id)).string();
        std::string partial    = final_name + ".partial";
        std::ofstream out(partial, std::ios::binary);
        if (!out.is_open())
            throw std::runtime_error("merge-nodes: cannot write new root");
        new_root.child_nodes_.clear();
        new_root.serialize(out);
        out.close();
        std::filesystem::rename(partial, final_name);
    }
    logger->info("[merge-nodes] New root written: {}", new_root_id);

    // ── 8. Write merged linkage (variable-arity format) ──────────────────────
    // Format: node_id child1 child2 [child3 ...]   (one internal node per line)
    std::ofstream lm_out(output_linkage);
    if (!lm_out.is_open())
        throw std::runtime_error("merge-nodes: cannot write linkage: " +
                                 output_linkage.string());

    // Nodes from A (unchanged IDs)
    for (size_t i = 0; i < linkage_a.size(); ++i) {
        if (linkage_a[i].empty()) continue; // leaf
        lm_out << i;
        for (uint64_t c : linkage_a[i]) lm_out << " " << c;
        lm_out << "\n";
    }
    // Nodes from B (IDs shifted by offset)
    for (size_t i = 0; i < linkage_b.size(); ++i) {
        if (linkage_b[i].empty()) continue; // leaf
        lm_out << (i + offset);
        for (uint64_t c : linkage_b[i]) lm_out << " " << (c + offset);
        lm_out << "\n";
    }
    // New root
    lm_out << new_root_id << " " << root_a_id << " " << root_b_id_renamed << "\n";
    lm_out.close();

    logger->info("[merge-nodes] Merged linkage written to {}", output_linkage.string());
    logger->info("[merge-nodes] Done. Merged index: {} columns, new root id {}.",
                 new_root.num_columns(), new_root_id);
}

void BRWTOptimizer::relax(BRWT *brwt_matrix, uint64_t max_arity, size_t num_threads) {
    assert(brwt_matrix);
    std::deque<BRWT*> parents;
    brwt_matrix->BFT([&](const BRWT &node) {
        if (node.child_nodes_.size())
            parents.push_front(const_cast<BRWT*>(&node));
    });

    ProgressBar progress_bar(parents.size(), "Relax Multi-BRWT",
                             std::cerr, !common::get_verbose());

    for (BRWT *parent : parents) {
        for (int g = parent->child_nodes_.size() - 1; g >= 0; --g) {
            const auto *node = dynamic_cast<const BRWT*>(parent->child_nodes_[g].get());
            if (node && should_prune(*node)
                     && parent->child_nodes_.size() - 1
                            + node->child_nodes_.size() <= max_arity) {
                reassign(g, parent, num_threads);
            }
        }
        ++progress_bar;
    }
}

void BRWTOptimizer::relax_nodes(std::vector<std::vector<uint64_t>> &linkage,
                                const std::filesystem::path &node_dir,
                                uint64_t max_arity,
                                size_t num_threads) {
    size_t num_nodes = linkage.size();
    if (!num_nodes) return;

    // 1. Identify internal nodes and build BFS order
    std::deque<uint64_t> queue;
    std::vector<uint64_t> bfs_order;
    queue.push_back(num_nodes - 1); // Start from Root

    std::vector<bool> visited(num_nodes, false);
    while (!queue.empty()) {
        uint64_t curr = queue.front();
        queue.pop_front();
        if (curr < num_nodes && !linkage[curr].empty() && !visited[curr]) {
            visited[curr] = true;
            bfs_order.push_back(curr);
            for (uint64_t child : linkage[curr]) queue.push_back(child);
        }
    }
    // Reverse to process bottom-up
    std::reverse(bfs_order.begin(), bfs_order.end());

    auto load_node = [&](uint64_t id) {
        BRWT node;
        // After a completed build the root is stored as `<id>` (union bv) and
        // every other node is stored as `node_<id>` (sub-index).  Try both so
        // relax-nodes works whether the folder is mid-build or fully assembled.
        auto path_bare  = node_dir / std::to_string(id);
        auto path_prefixed = node_dir / ("node_" + std::to_string(id));

        std::filesystem::path chosen;
        if (std::filesystem::exists(path_bare))           chosen = path_bare;
        else if (std::filesystem::exists(path_prefixed))  chosen = path_prefixed;
        else {
            logger->error("Relax: Cannot find node {} — tried:\n  {}\n  {}",
                          id, path_bare.string(), path_prefixed.string());
            exit(1);
        }

        std::unique_ptr<std::ifstream> in = utils::open_ifstream(chosen);
        if (!in || !in->is_open() || !node.load(*in)) {
            logger->error("Relax: Failed to load node {} from {}", id, chosen.string());
            exit(1);
        }
        return node;
    };

    auto save_node = [&](BRWT& node, uint64_t id) {
        // Mirror load_node: write back to whichever file already exists.
        auto path_bare     = node_dir / std::to_string(id);
        auto path_prefixed = node_dir / ("node_" + std::to_string(id));
        std::filesystem::path target = std::filesystem::exists(path_bare)
                                     ? path_bare : path_prefixed;
        std::string partial_name = target.string() + ".partial";
        std::ofstream out(partial_name, std::ios::binary);
        node.child_nodes_.clear();
        node.serialize(out);
        out.close();
        std::filesystem::rename(partial_name, target);
    };

    ProgressBar progress_bar(bfs_order.size(), "Relaxing Nodes", std::cerr, !common::get_verbose());

    for (uint64_t p_id : bfs_order) {
        BRWT parent = load_node(p_id);
        bool changed = false;

        // Populate parent with actual child objects for the optimizer logic
        parent.child_nodes_.resize(linkage[p_id].size());
        for (size_t g = 0; g < linkage[p_id].size(); ++g) {
            parent.child_nodes_[g].reset(new BRWT(load_node(linkage[p_id][g])));
        }

        // Try to prune each child
        for (int g = parent.child_nodes_.size() - 1; g >= 0; --g) {
            BRWT* child = dynamic_cast<BRWT*>(parent.child_nodes_[g].get());
            uint64_t c_id = linkage[p_id][g];

            if (child && !linkage[c_id].empty() && should_prune(*child)
                && parent.child_nodes_.size() - 1 + linkage[c_id].size() <= max_arity) {
                
                // Get the grandchild IDs before they are merged
                std::vector<uint64_t> grandchildren_ids = linkage[c_id];
                
                reassign(g, &parent, num_threads);
                
                // Update Global Linkage: remove child, insert grandchildren
                linkage[p_id].erase(linkage[p_id].begin() + g);
                linkage[p_id].insert(linkage[p_id].begin() + g, grandchildren_ids.begin(), grandchildren_ids.end());
                
                // Delete the pruned node from disk (try both naming conventions)
                std::error_code ec;
                if (!std::filesystem::remove(node_dir / std::to_string(c_id), ec))
                    std::filesystem::remove(node_dir / ("node_" + std::to_string(c_id)), ec);
                changed = true;
            }
        }

        if (changed) {
            // Save updated grandchildren (they now have new sub-indices against the grandparent)
            for (size_t g = 0; g < linkage[p_id].size(); ++g) {
                save_node(*parent.child_nodes_[g], linkage[p_id][g]);
            }
            save_node(parent, p_id);
        }
        ++progress_bar;
    }
}

bool BRWTOptimizer::should_prune(const BRWT &node) {
    if (!node.child_nodes_.size()) return false;
    for (const auto &child_node : node.child_nodes_) {
        if (!dynamic_cast<const BRWT *>(child_node.get())) return false;
    }
    return pruning_delta(node) <= 0;
}

void BRWTOptimizer::reassign(size_t node_rank, BRWT *parent, size_t num_threads) {
    BRWT &node = dynamic_cast<BRWT&>(*parent->child_nodes_.at(node_rank));
    std::vector<uint64_t> column_arrangement;
    std::vector<size_t> group_sizes;
    for (size_t g = 0; g < parent->assignments_.num_groups(); ++g) {
        if (g == node_rank) continue;
        size_t group_size = parent->child_nodes_[g]->num_columns();
        group_sizes.push_back(group_size);
        for (size_t r = 0; r < group_size; ++r) {
            column_arrangement.push_back(parent->assignments_.get(g, r));
        }
    }
    for (size_t g = 0; g < node.assignments_.num_groups(); ++g) {
        size_t group_size = node.child_nodes_[g]->num_columns();
        group_sizes.push_back(group_size);
        for (size_t r = 0; r < group_size; ++r) {
            column_arrangement.push_back(
                parent->assignments_.get(node_rank, node.assignments_.get(g, r))
            );
        }
    }
    auto children_reassigned = std::move(node.child_nodes_);
    const auto index_column = node.nonzero_rows_->convert_to<sdsl::bit_vector>();
    parent->child_nodes_.erase(parent->child_nodes_.begin() + node_rank);
    parent->child_nodes_.resize(parent->child_nodes_.size() + children_reassigned.size());
    parent->assignments_ = RangePartition(column_arrangement, group_sizes);

    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (size_t t = 0; t < children_reassigned.size(); ++t) {
        std::unique_ptr<BRWT> grand_child { dynamic_cast<BRWT*>(children_reassigned[t].release()) };
        sdsl::bit_vector subindex(index_column.size(), false);
        uint64_t child_i = 0;
        uint64_t w = 0;
        call_ones(index_column, [&](auto i) {
            if (child_i % 64 == 0) {
                w = grand_child->nonzero_rows_->get_int(child_i, std::min(static_cast<uint64_t>(64), grand_child->nonzero_rows_->size() - child_i));
            }
            if (w & 1) subindex[i] = true;
            w >>= 1;
            child_i++;
        });
        grand_child->nonzero_rows_ = std::make_unique<bit_vector_smallrank>(std::move(subindex));
        parent->child_nodes_[parent->child_nodes_.size() - children_reassigned.size() + t] = std::move(grand_child);
    }
}

double BRWTOptimizer::pruning_delta(const BRWT &node) {
    double delta = 0;
    for (const auto &submatrix : node.child_nodes_) {
        auto *brwt_child = dynamic_cast<const BRWT*>(submatrix.get());
        delta += bit_vector_smallrank::predict_size(node.num_rows(), brwt_child->nonzero_rows_->num_set_bits());
        delta -= bit_vector_smallrank::predict_size(brwt_child->nonzero_rows_->size(), brwt_child->nonzero_rows_->num_set_bits());
    }
    delta -= bit_vector_smallrank::predict_size(node.nonzero_rows_->size(), node.nonzero_rows_->num_set_bits());
    return delta;
}

} // namespace matrix
} // namespace annot
} // namespace mtg
