#include "brwt_builders.hpp"

#include <omp.h>
#include <queue>
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
    sdsl::sd_vector<> sdvec;
    sdsl::load(sdvec, in);
    
    BRWT node;
    node.assignments_ = RangePartition(Partition(1, { 0 }));
    
    sdsl::sd_vector<>::rank_1_type rank1(&sdvec);
    uint64_t num_ones = rank1(sdvec.size());
    uint64_t size = sdvec.size();

    node.nonzero_rows_ = std::make_unique<bit_vector_smart>(
        [&](const auto &callback) {
            sdsl::sd_vector<>::select_1_type select_1(&sdvec);
            for (uint64_t r = 1; r <= num_ones; ++r) {
                callback(select_1(r));
            }
        },
        size,
        num_ones
    );
    return node;
}


BRWTBottomUpBuilder::Partitioner
BRWTBottomUpBuilder::get_basic_partitioner(size_t arity) {
    assert(arity > 0u);

    return [arity](const VectorPtrs &vectors) {
        if (!vectors.size())
            return Partition(0);

        Partition partition((vectors.size() - 1) / arity + 1);
        for (size_t i = 0; i < vectors.size(); ++i) {
            partition[i / arity].push_back(i);
        }
        return partition;
    };
}

BRWT BRWTBottomUpBuilder::concatenate(std::vector<BRWT>&& submatrices,
                                      sdsl::bit_vector *buffer,
                                      ThreadPool &thread_pool,
                                      const std::filesystem::path &node_tmp_path) {
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
        // Strategy C: Bitmap-OR (Fastest for dense nodes)
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
            cursor.num_ones = cursor.vec->num_set_bits();
            cursor.rank = 0;
            if (cursor.next()) pq.push(cursor);
        }

        bool use_disk_buffer = (est_density > 0.015 && !node_tmp_path.empty());
        uint64_t p_rank = 0;

        if (use_disk_buffer) {
            // Strategy B: Disk-Buffered Stream (RAM-Safe)
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
            // Strategy A: RAM Array (Fastest for sparse nodes)
            std::vector<uint64_t> p_ones;
            std::vector<std::vector<uint64_t>> c_ranks(K);
            while (!pq.empty()) {
                uint64_t min_pos = pq.top().current_pos;
                p_rank++;
                p_ones.push_back(min_pos);
                while (!pq.empty() && pq.top().current_pos == min_pos) {
                    NodeCursor cursor = pq.top();
                    pq.pop();
                    c_ranks[cursor.child_idx].push_back(p_rank - 1);
                    if (cursor.next()) pq.push(cursor);
                }
            }

            parent.nonzero_rows_ = std::make_unique<bit_vector_smart>([p_ones](const auto& cb) {
                for (uint64_t p : p_ones) cb(p);
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

    size_t total_nodes = linkage.size();
    std::vector<size_t> node_levels(total_nodes, 0);
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
        
        // Manual creation to bypass automatic registration in utils::TMP_DIRS
        std::string pattern = (tmp_path / "temp_brwt_XXXXXX").string();
        char* res = mkdtemp(pattern.data());
        if (!res) {
            logger->error("Failed to create a temporary directory in {}", tmp_path.string());
            exit(1);
        }
        actual_tmp_dir = res;
        logger->info("Created persistent temporary directory: {}", actual_tmp_dir.string());
    }
    if (actual_tmp_dir_out) *actual_tmp_dir_out = actual_tmp_dir.string();

    auto dump_node = [&](BRWT& node, uint64_t id) {
        std::string filename = (actual_tmp_dir/std::to_string(id)).string();
        std::ofstream out(filename, std::ios::binary);
        if (!out.is_open()) {
            logger->error("Failed to open temp node file for writing: {}", filename);
            exit(1);
        }
        node.child_nodes_.clear();
        node.serialize(out);
        out.close();
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

    uint64_t num_rows = 0;
    std::vector<bool> done(total_nodes, false);
    std::mutex done_mu;
    std::condition_variable done_cond;

    get_columns([&](uint64_t i, std::unique_ptr<bit_vector>&& column) {
        if (column) num_rows = column->size();
        {
            std::unique_lock<std::mutex> lock(done_mu);
            done[i] = true;
        }
        done_cond.notify_all();
    });

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

    for (size_t lv = 1; lv <= max_level; ++lv) {
        std::vector<size_t> level_nodes;
        for (size_t i = 0; i < total_nodes; ++i) if (node_levels[i] == lv) level_nodes.push_back(i);

        // Density sampling to choose level strategy
        double level_density = 0;
        size_t sample_size = std::min(level_nodes.size(), (size_t)10);
        for (size_t s = 0; s < sample_size; ++s) {
            BRWT sm = get_node_lambda(linkage[level_nodes[s]][0]);
            if (sm.nonzero_rows_)
                level_density += (double)sm.nonzero_rows_->num_set_bits() / num_rows;
        }
        level_density /= sample_size;

        // Decision: Use Bitmap Workbench if density is high
        bool use_bitmap = (level_density > 0.5);
        size_t p_nodes = (level_density > 0.1) ? num_partial_nodes : num_full_nodes;
        
        logger->info("[STEP] Level {}/{} (Density: {:.2f}%) - Parallel Nodes: {} (Mode: {})", 
                     lv, max_level, level_density * 100, p_nodes, use_bitmap ? "Bitmap" : "Index");
        log_mem();

        std::vector<sdsl::bit_vector> buffers(p_nodes);
        if (use_bitmap) {
            for(auto& b : buffers) b = sdsl::bit_vector(num_rows, 0);
        }

        #pragma omp parallel for num_threads(p_nodes) schedule(dynamic)
        for (size_t idx = 0; idx < level_nodes.size(); ++idx) {
            size_t i = level_nodes[idx];
            if (resume && std::filesystem::exists(actual_tmp_dir/std::to_string(i))) {
                { std::unique_lock<std::mutex> lock(done_mu); done[i] = true; }
                done_cond.notify_all();
                continue;
            }

            std::vector<BRWT> children;
            for (uint64_t j : linkage[i]) {
                std::unique_lock<std::mutex> lock(done_mu);
                done_cond.wait(lock, [&] { return done[j]; });
                children.push_back(get_node_lambda(j));
            }

            BRWT node = concatenate(std::move(children), 
                                    use_bitmap ? &buffers[omp_get_thread_num()] : nullptr, 
                                    thread_pool, actual_tmp_dir/fmt::format("{}_buf", i));

            for (size_t r = 0; r < node.child_nodes_.size(); ++r) dump_node(*node.child_nodes_[r], linkage[i][r]);
            dump_node(node, i);

            { std::unique_lock<std::mutex> lock(done_mu); done[i] = true; }
            done_cond.notify_all();
            #pragma omp critical
            { ++progress_bar; }
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
        auto filename = tmp_dir/std::to_string(id);
        std::unique_ptr<std::ifstream> in = utils::open_ifstream(filename);
        if (!in || !in->is_open()) exit(1);
        if (!node.load(*in)) exit(1);
        if (id == linkage.size() - 1) {
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

BRWT BRWTBottomUpBuilder::merge(std::vector<BRWT>&& nodes, Partitioner partitioner, size_t num_nodes_parallel, size_t num_threads) {
    return std::move(nodes[0]);
}

void BRWTOptimizer::relax(BRWT *brwt_matrix, uint64_t max_arity, size_t num_threads) {}
bool BRWTOptimizer::should_prune(const BRWT &node) { return false; }
void BRWTOptimizer::reassign(size_t node_rank, BRWT *parent, size_t num_threads) {}
double BRWTOptimizer::pruning_delta(const BRWT &node) { return 0; }

} // namespace matrix
} // namespace annot
} // namespace mtg
