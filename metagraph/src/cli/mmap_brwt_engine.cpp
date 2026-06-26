#include "mmap_brwt_engine.hpp"

#include <iostream>
#include <limits>
#include <algorithm>
#include <cassert>

#include "common/utils/template_utils.hpp"

namespace mtg {
namespace cli {

// ──────────────────────────────────────────────────────────────────────────────
// Topology Reconstructor
// ──────────────────────────────────────────────────────────────────────────────

struct FlatNode {
    uint64_t offset;
    uint64_t size;
    uint64_t num_children;
};

void build_tree(MmapNode& node, const std::vector<FlatNode>& flat_nodes, size_t& current_idx) {
    if (current_idx >= flat_nodes.size()) return;
    
    node.offset = flat_nodes[current_idx].offset;
    node.size = flat_nodes[current_idx].size;
    node.num_children = flat_nodes[current_idx].num_children;
    current_idx++;
    
    node.children.resize(node.num_children);
    for (uint64_t i = 0; i < node.num_children; ++i) {
        build_tree(node.children[i], flat_nodes, current_idx);
    }
}

MmapNode build_global_topology(const std::string& index_file, const std::string& brwt_file) {
    std::cout << "Loading mmap index from " << index_file << "..." << std::endl;
    std::ifstream idx_in(index_file, std::ios::binary);
    if (!idx_in) throw std::runtime_error("Cannot open index file: " + index_file);
    
    uint64_t num_nodes = load_number(idx_in);
    std::vector<FlatNode> flat_nodes(num_nodes);
    
    for (uint64_t i = 0; i < num_nodes; ++i) {
        flat_nodes[i].offset = load_number(idx_in);
        flat_nodes[i].size = load_number(idx_in);
        flat_nodes[i].num_children = 0;
    }
    
    std::cout << "Reading num_children metadata from " << brwt_file << "..." << std::endl;
    std::ifstream brwt_in(brwt_file, std::ios::binary);
    if (!brwt_in) throw std::runtime_error("Cannot open brwt file: " + brwt_file);
    
    for (uint64_t i = 0; i < num_nodes; ++i) {
        // num_children is the very last 8 bytes of the node
        brwt_in.seekg(flat_nodes[i].offset + flat_nodes[i].size - 8, std::ios::beg);
        flat_nodes[i].num_children = load_number(brwt_in);
    }
    
    MmapNode root;
    size_t current_idx = 0;
    build_tree(root, flat_nodes, current_idx);
    
    if (current_idx != num_nodes) {
        throw std::runtime_error("Tree topology construction failed. Mismatch in node count.");
    }
    
    std::cout << "Successfully built in-RAM topology tree." << std::endl;
    return root;
}

// ──────────────────────────────────────────────────────────────────────────────
// MmapBRWTEngine
// ──────────────────────────────────────────────────────────────────────────────

MmapBRWTEngine::MmapBRWTEngine(const std::string& brwt_file, const MmapNode& root) 
    : brwt_file_(brwt_file), root_(root), in_(brwt_file, std::ios_base::in) {
    if (!in_) throw std::runtime_error("Engine failed to open mmap stream for " + brwt_file);
    mmap_context_ = in_.get_mmap_context();
}

MmapBRWTEngine::~MmapBRWTEngine() {}

std::vector<Vector<std::pair<uint64_t, uint64_t>>> MmapBRWTEngine::get_rows(const std::vector<uint64_t>& row_ids) {
    std::vector<Vector<std::pair<uint64_t, uint64_t>>> rows(row_ids.size());
    Vector<std::pair<uint64_t, uint64_t>> slice;
    slice.reserve(row_ids.size() * 4);
    
    slice_rows(root_, row_ids, &slice);
    
    auto row_begin = slice.begin();
    for (size_t i = 0; i < rows.size(); ++i) {
        auto row_end = row_begin;
        while (row_end->first != std::numeric_limits<uint64_t>::max()) {
            ++row_end;
        }
        rows[i].assign(row_begin, row_end);
        row_begin = row_end + 1;
    }
    return rows;
}

template <typename T>
void MmapBRWTEngine::slice_rows(const MmapNode& node, const std::vector<uint64_t>& row_ids, Vector<T>* slice) {
    T delim;
    if constexpr(utils::is_pair_v<T>) {
        delim = std::make_pair(std::numeric_limits<uint64_t>::max(), 0);
    } else {
        delim = std::numeric_limits<uint64_t>::max();
    }

    in_.seekg(node.offset, std::ios::beg);
    
    RangePartition assignments;
    if (!assignments.load(in_)) {
        throw std::runtime_error("Failed to load assignments at offset " + std::to_string(node.offset));
    }
    
    bit_vector_smallrank nonzero_rows;
    if (!nonzero_rows.load(in_)) {
        throw std::runtime_error("Failed to load nonzero_rows at offset " + std::to_string(node.offset));
    }

    if (node.num_children == 0) {
        for (uint64_t i : row_ids) {
            if constexpr(utils::is_pair_v<T>) {
                if (uint64_t rank = nonzero_rows.conditional_rank1(i)) {
                    slice->emplace_back(0, rank);
                }
            } else {
                if (nonzero_rows[i]) {
                    slice->push_back(0);
                }
            }
            slice->push_back(delim);
        }
        return;
    }

    std::vector<uint64_t> child_row_ids;
    child_row_ids.reserve(row_ids.size());
    std::vector<bool> skip_row(row_ids.size(), true);

    for (size_t i = 0; i < row_ids.size(); ++i) {
        uint64_t global_offset = row_ids[i];

        if (i + 4 < row_ids.size()
                && row_ids[i + 4] < global_offset + 64
                && row_ids[i + 4] >= global_offset
                && global_offset + 64 <= nonzero_rows.size()) {
            
            uint64_t word = nonzero_rows.get_int(global_offset, 64);
            uint64_t rank = -1ULL;

            do {
                uint8_t offset = row_ids[i] - global_offset;
                if (word & (1ULL << offset)) {
                    if (rank == -1ULL)
                        rank = global_offset > 0 ? nonzero_rows.rank1(global_offset - 1) : 0;

                    child_row_ids.push_back(rank + sdsl::bits::cnt(word & sdsl::bits::lo_set[offset + 1]) - 1);
                    skip_row[i] = false;
                }
            } while (++i < row_ids.size()
                        && row_ids[i] < global_offset + 64
                        && row_ids[i] >= global_offset);
            --i;

        } else {
            if (uint64_t rank = nonzero_rows.conditional_rank1(global_offset)) {
                child_row_ids.push_back(rank - 1);
                skip_row[i] = false;
            }
        }
    }

    if (!child_row_ids.size()) {
        for (size_t i = 0; i < row_ids.size(); ++i) {
            slice->push_back(delim);
        }
        return;
    }

    size_t slice_start = slice->size();
    std::vector<size_t> pos(node.children.size());

    for (size_t j = 0; j < node.children.size(); ++j) {
        pos[j] = slice->size();
        slice_rows<T>(node.children[j], child_row_ids, slice);

        for (size_t i = pos[j]; i < slice->size(); ++i) {
            auto &v = (*slice)[i];
            if (v != delim) {
                auto &col = utils::get_first(v);
                col = assignments.get(j, col);
            }
        }
    }

    size_t slice_offset = slice->size();

    for (size_t i = 0; i < row_ids.size(); ++i) {
        if (!skip_row[i]) {
            for (size_t &p : pos) {
                while ((*slice)[p++] != delim) {
                    slice->push_back((*slice)[p - 1]);
                }
            }
        }
        slice->push_back(delim);
    }

    slice->erase(slice->begin() + slice_start, slice->begin() + slice_offset);
}

// ──────────────────────────────────────────────────────────────────────────────
// EnginePool
// ──────────────────────────────────────────────────────────────────────────────

EnginePool::EnginePool(const std::string& brwt_file, const MmapNode& root) 
    : brwt_file_(brwt_file), root_(root) {}

std::shared_ptr<MmapBRWTEngine> EnginePool::acquire() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pool_.empty()) {
        auto engine = pool_.back();
        pool_.pop_back();
        return engine;
    }
    return std::make_shared<MmapBRWTEngine>(brwt_file_, root_);
}

void EnginePool::release(std::shared_ptr<MmapBRWTEngine> engine) {
    std::lock_guard<std::mutex> lock(mutex_);
    pool_.push_back(engine);
}

} // namespace cli
} // namespace mtg
