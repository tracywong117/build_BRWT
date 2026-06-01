#ifndef __BRWT_BUILDERS_HPP__
#define __BRWT_BUILDERS_HPP__

#include <vector>
#include <unordered_map>
#include <memory>
#include <filesystem>

#include "common/vectors/bit_vector.hpp"
#include "common/threads/threading.hpp"
#include "brwt.hpp"


namespace mtg {
namespace annot {
namespace matrix {

class BRWTBottomUpBuilder {
  public:
    typedef std::vector<const bit_vector *> VectorPtrs;
    typedef std::vector<std::vector<BRWT::Column>> Partition;
    typedef std::function<Partition(const VectorPtrs &)> Partitioner;

    static Partitioner get_basic_partitioner(size_t arity = 2);

    // Build the Multi-BRWT compressed representation of a binary matrix
    static BRWT build(std::vector<std::unique_ptr<bit_vector>>&& columns,
                      Partitioner partitioner = get_basic_partitioner(),
                      size_t num_nodes_parallel = 1,
                      size_t num_threads = 1);

    using CallColumn
        = std::function<void(uint64_t, std::unique_ptr<bit_vector>&&)>;

    // Disk-assisted build
    static BRWT build(const std::function<void(const CallColumn &)> &get_columns,
                      const std::vector<std::vector<uint64_t>> &linkage,
                      const std::filesystem::path &tmp_dir,
                      const std::vector<std::vector<uint64_t>> &stored_columns,
                      const std::vector<std::string> &column_files = {},
                      size_t num_full_nodes = 0,
                      size_t num_partial_nodes = 0,
                      size_t num_threads = 1,
                      std::string *actual_tmp_dir_out = nullptr,
                      bool resume = true);

    // Merge two monolithic .brwt files (same row length) into one output .brwt.
    // Column names are NOT handled here; see handle_merge() in main.cpp.
    static void merge(const std::filesystem::path &brwt_a,
                      const std::filesystem::path &brwt_b,
                      const std::filesystem::path &output_prefix,
                      size_t num_threads = 1);

    // Merge two node-folder BRWT representations (same row length).
    // Both formats: linkage file + directory of node_<id> files + one <root_id> file.
    // Output: new node folder + merged linkage file.
    static void merge_nodes(const std::filesystem::path &linkage_a,
                            const std::filesystem::path &node_dir_a,
                            const std::filesystem::path &linkage_b,
                            const std::filesystem::path &node_dir_b,
                            const std::filesystem::path &output_node_dir,
                            const std::filesystem::path &output_linkage,
                            size_t num_threads = 1);

    // Concatenate multiple Multi-BRWT submatrices
    static BRWT concatenate(std::vector<BRWT>&& submatrices,
                            sdsl::bit_vector *buffer,
                            ThreadPool &thread_pool,
                            const std::filesystem::path &node_tmp_path = "");

    // Streaming assembly to avoid "RAM Wall"
    static void assemble_streaming(std::ostream &out,
                                   const std::vector<std::vector<uint64_t>> &linkage,
                                   const std::vector<std::vector<uint64_t>> &stored_columns,
                                   const std::filesystem::path &tmp_dir,
                                   const std::vector<std::string> &column_files,
                                   bool cleanup_tmp = false);

    static BRWT load_leaf_from_file(const std::string& path);

    // Query rows from a BRWT stored as a nodes folder (disk-based format).
    // Returns one vector of set column IDs per query row (same order as `row_ids`).
    // `linkage`   : linkage matrix as produced by parse_linkage_matrix().
    // `node_dir`  : folder containing node_<id> files for all non-root nodes
    //               and a single <root_id> file for the root (union bit-vector).
    // `log_stats` : when true, logs a summary (nodes loaded, hits, etc.) at the end.
    static std::vector<std::vector<uint64_t>>
    query_nodes(const std::vector<uint64_t>& row_ids,
                const std::vector<std::vector<uint64_t>>& linkage,
                const std::filesystem::path& node_dir,
                bool log_stats = false);

    // Extract column bit-vectors from a monolithic .brwt file and write each
    // as an sdsl::sd_vector<> to out_dir/<name>.sd.
    // col_requests: pairs of (column_id, output_filename_stem)
    static void query_columns(const BRWT &brwt,
                              const std::vector<std::pair<uint64_t, std::string>> &col_requests,
                              const std::filesystem::path &out_dir);

    // Extract column bit-vectors from a node-folder BRWT (no-assemble format)
    // by walking root→leaf along the linkage tree and composing sub-indices.
    // col_requests: pairs of (column_id, output_filename_stem)
    static void query_columns_nodes(
                              const std::vector<std::pair<uint64_t, std::string>> &col_requests,
                              const std::vector<std::vector<uint64_t>> &linkage,
                              const std::filesystem::path &node_dir,
                              const std::filesystem::path &out_dir);

    // Extend the BRWT matrix by appending (target_len - num_rows) zero rows at
    // the end.  Only the root's nonzero_rows_ changes; child sub-indices are
    // rank-based and need no modification.
    //
    // root_append: monolithic .brwt — loads full tree, writes new .brwt file.
    static void root_append(const std::filesystem::path &input_brwt,
                            uint64_t target_len,
                            const std::filesystem::path &output_brwt);

    // root_append_nodes: node-folder format — touches only the root file.
    //   Backup: <root_id>  →  ori_node_<root_id>
    //   New root written to <root_id> (atomic via .partial).
    static void root_append_nodes(const std::vector<std::vector<uint64_t>> &linkage,
                                  const std::filesystem::path &node_dir,
                                  uint64_t target_len);
};


class BRWTOptimizer {
  public:
    virtual ~BRWTOptimizer() {}

    // remove some internal nodes to make the tree
    // smaller and increase the arity
    static void relax(BRWT *brwt_matrix,
                      uint64_t max_arity = -1,
                      size_t num_threads = 1);

    // GOAL: High-scale Node Folder Relaxation
    static void relax_nodes(std::vector<std::vector<uint64_t>> &linkage,
                            const std::filesystem::path &node_dir,
                            uint64_t max_arity,
                            size_t num_threads);

  private:
    // check if removing this node is going to reduce the size
    static bool should_prune(const BRWT &node);
    // remove the node and reassign all its children to its parent
    static void reassign(size_t node_rank, BRWT *parent, size_t num_threads);
    static double pruning_delta(const BRWT &node);
};

} // namespace matrix
} // namespace annot
} // namespace mtg

#endif // __BRWT_BUILDERS_HPP__
