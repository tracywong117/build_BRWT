#ifndef __MMAP_BRWT_ENGINE_HPP__
#define __MMAP_BRWT_ENGINE_HPP__

#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <unordered_map>
#include <mutex>

#include <sdsl/memory_management.hpp>

#include "common/serialization.hpp"
#include "common/range_partition.hpp"
#include "common/vectors/bit_vector_adaptive.hpp"
#include "common/vector.hpp"

namespace mtg {
namespace cli {

// In-RAM topology node
struct MmapNode {
    uint64_t offset;
    uint64_t size;
    uint64_t num_children;
    std::vector<MmapNode> children;
};

// Thread-local query engine
class MmapBRWTEngine {
public:
    MmapBRWTEngine(const std::string& brwt_file, const MmapNode& root);
    ~MmapBRWTEngine();

    std::vector<Vector<std::pair<uint64_t, uint64_t>>> get_rows(const std::vector<uint64_t>& row_ids);

private:
    std::string brwt_file_;
    const MmapNode& root_;
    sdsl::mmap_ifstream in_;
    std::shared_ptr<sdsl::mmap_context> mmap_context_;
    
    // Core slice rows function
    template <typename T>
    void slice_rows(const MmapNode& node, const std::vector<uint64_t>& row_ids, Vector<T>* slice);
};

// Global topology builder
MmapNode build_global_topology(const std::string& index_file, const std::string& brwt_file);

// Thread pool for concurrent queries
class EnginePool {
public:
    EnginePool(const std::string& brwt_file, const MmapNode& root);
    
    std::shared_ptr<MmapBRWTEngine> acquire();
    void release(std::shared_ptr<MmapBRWTEngine> engine);

private:
    std::string brwt_file_;
    const MmapNode& root_;
    std::vector<std::shared_ptr<MmapBRWTEngine>> pool_;
    std::mutex mutex_;
};

} // namespace cli
} // namespace mtg

#endif // __MMAP_BRWT_ENGINE_HPP__
