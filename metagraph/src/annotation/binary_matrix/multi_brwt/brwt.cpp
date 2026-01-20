#include "brwt.hpp"
#include "brwt_builders.hpp"

#include <queue>
#include <numeric>
#include <thread>

#include "common/algorithms.hpp"
#include "common/serialization.hpp"
#include "common/utils/template_utils.hpp"

namespace mtg {
namespace annot {
namespace matrix {

// Function to get the value at a specific row and column
bool BRWT::get(Row row, Column column) const {
    assert(row < num_rows()); // Ensure the row is within bounds
    assert(column < num_columns()); // Ensure the column is within bounds

    // If this is a leaf node
    if (!child_nodes_.size())
        return (*nonzero_rows_)[row]; // Return the value directly

    uint64_t rank = nonzero_rows_->conditional_rank1(row);
    // If the index bit is unset, return false
    if (!rank)
        return false;

    auto child_node = assignments_.group(column);
    return child_nodes_[child_node]->get(rank - 1, assignments_.rank(column));
}

// Function to get the positions of set bits in specific rows
std::vector<BRWT::SetBitPositions>
BRWT::get_rows(const std::vector<Row> &row_ids) const {
    std::vector<SetBitPositions> rows(row_ids.size());

    Vector<Column> slice;
    // Expect at least 3 relations per row
    slice.reserve(row_ids.size() * 4);

    slice_rows(row_ids, &slice);

    assert(slice.size() >= row_ids.size());

    auto row_begin = slice.begin();

    for (size_t i = 0; i < rows.size(); ++i) {
        // Every row in `slice` ends with `-1`
        auto row_end = std::find(row_begin, slice.end(),
                                 std::numeric_limits<Column>::max());
        rows[i].assign(row_begin, row_end);
        row_begin = row_end + 1;
    }

    return rows;
}

// Function to get the column ranks in specific rows
std::vector<Vector<std::pair<BRWT::Column, uint64_t>>>
BRWT::get_column_ranks(const std::vector<Row> &row_ids) const {
    std::vector<Vector<std::pair<Column, uint64_t>>> rows(row_ids.size());

    Vector<std::pair<Column, uint64_t>> slice;
    // Expect at least 3 relations per row
    slice.reserve(row_ids.size() * 4);

    slice_rows(row_ids, &slice);

    assert(slice.size() >= row_ids.size());

    auto row_begin = slice.begin();

    for (size_t i = 0; i < rows.size(); ++i) {
        // Every row in `slice` ends with `-1`
        auto row_end = row_begin;
        while (row_end->first != std::numeric_limits<Column>::max()) {
            ++row_end;
            assert(row_end != slice.end());
        }
        rows[i].assign(row_begin, row_end);
        row_begin = row_end + 1;
    }

    return rows;
}

// Template function to slice rows and append to `slice`
template <typename T>
void BRWT::slice_rows(const std::vector<Row> &row_ids, Vector<T> *slice) const {
    T delim;
    if constexpr(utils::is_pair_v<T>) {
        delim = std::make_pair(std::numeric_limits<Column>::max(), 0);
    } else {
        delim = std::numeric_limits<Column>::max();
    }

    // Check if this is a leaf node
    if (!child_nodes_.size()) {
        assert(assignments_.size() == 1);

        for (Row i : row_ids) {
            assert(i < num_rows());

            if constexpr(utils::is_pair_v<T>) {
                if (uint64_t rank = nonzero_rows_->conditional_rank1(i)) {
                    // Only a single column is stored in leaves
                    slice->emplace_back(0, rank);
                }
            } else {
                if ((*nonzero_rows_)[i]) {
                    // Only a single column is stored in leaves
                    slice->push_back(0);
                }
            }
            slice->push_back(delim);
        }

        return;
    }

    // Construct indexing for children and the inverse mapping
    std::vector<Row> child_row_ids;
    child_row_ids.reserve(row_ids.size());

    std::vector<bool> skip_row(row_ids.size(), true);

    for (size_t i = 0; i < row_ids.size(); ++i) {
        assert(row_ids[i] < num_rows());

        uint64_t global_offset = row_ids[i];

        // If next word contains 5 or more positions, query the whole word
        if (i + 4 < row_ids.size()
                && row_ids[i + 4] < global_offset + 64
                && row_ids[i + 4] >= global_offset
                && global_offset + 64 <= nonzero_rows_->size()) {
            // Get the word
            uint64_t word = nonzero_rows_->get_int(global_offset, 64);
            uint64_t rank = -1ULL;

            do {
                // Check index
                uint8_t offset = row_ids[i] - global_offset;
                if (word & (1ULL << offset)) {
                    if (rank == -1ULL)
                        rank = global_offset > 0
                                ? nonzero_rows_->rank1(global_offset - 1)
                                : 0;

                    // Map index from parent's to children's coordinate system
                    child_row_ids.push_back(rank + sdsl::bits::cnt(word & sdsl::bits::lo_set[offset + 1]) - 1);
                    skip_row[i] = false;
                }
            } while (++i < row_ids.size()
                        && row_ids[i] < global_offset + 64
                        && row_ids[i] >= global_offset);
            --i;

        } else {
            // Check index
            if (uint64_t rank = nonzero_rows_->conditional_rank1(global_offset)) {
                // Map index from parent's to children's coordinate system
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

    // Query all children subtrees and get relations from them
    size_t slice_start = slice->size();

    std::vector<size_t> pos(child_nodes_.size());

    for (size_t j = 0; j < child_nodes_.size(); ++j) {
        pos[j] = slice->size();
        child_nodes_[j]->slice_rows<T>(child_row_ids, slice);

        assert(slice->size() >= pos[j] + child_row_ids.size());

        // Transform column indexes
        for (size_t i = pos[j]; i < slice->size(); ++i) {
            auto &v = (*slice)[i];
            if (v != delim) {
                auto &col = utils::get_first(v);
                col = assignments_.get(j, col);
            }
        }
    }

    size_t slice_offset = slice->size();

    for (size_t i = 0; i < row_ids.size(); ++i) {
        if (!skip_row[i]) {
            // Merge rows from child submatrices
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

// Function to get the rows corresponding to a specific column
std::vector<BRWT::Row> BRWT::get_column(Column column) const {
    assert(column < num_columns()); // Ensure the column is within bounds

    auto num_nonzero_rows = nonzero_rows_->num_set_bits();

    // Check if the column is empty
    if (!num_nonzero_rows)
        return {};

    // Check whether it is a leaf node
    if (!child_nodes_.size()) {
        // Return the index column
        std::vector<BRWT::Row> result;
        result.reserve(num_nonzero_rows);
        nonzero_rows_->call_ones([&](auto i) { result.push_back(i); });
        return result;
    }

    auto child_node = assignments_.group(column);
    auto rows = child_nodes_[child_node]->get_column(assignments_.rank(column));

    // Check if we need to update the row indexes
    if (num_nonzero_rows == nonzero_rows_->size())
        return rows;

    // Shift indexes
    for (size_t i = 0; i < rows.size(); ++i) {
        rows[i] = nonzero_rows_->select1(rows[i] + 1);
    }
    return rows;
}

// Function to load the BRWT from an input stream
bool BRWT::load(std::istream &in) {
    if (!in.good())
        return false;

    try {
        if (!assignments_.load(in))
            return false;

        if (!nonzero_rows_->load(in))
            return false;

        size_t num_child_nodes = load_number(in);
        child_nodes_.clear();
        child_nodes_.reserve(num_child_nodes);
        for (size_t i = 0; i < num_child_nodes; ++i) {
            child_nodes_.emplace_back(new BRWT());
            if (!child_nodes_.back()->load(in))
                return false;
        }
        return !child_nodes_.size()
                    || child_nodes_.size() == assignments_.num_groups();
    } catch (...) {
        return false;
    }
}

// Function to serialize the BRWT to an output stream
void BRWT::serialize(std::ostream &out) const {
    if (!out.good())
        throw std::ofstream::failure("Error when dumping BRWT");

    assignments_.serialize(out);

    assert(!child_nodes_.size()
                || child_nodes_.size() == assignments_.num_groups());

    nonzero_rows_->serialize(out);

    serialize_number(out, child_nodes_.size());
    for (const auto &child : child_nodes_) {
        child->serialize(out);
    }
}

// Function to get the number of relations in the BRWT
uint64_t BRWT::num_relations() const {
    if (!child_nodes_.size())
        return nonzero_rows_->num_set_bits();

    uint64_t num_set_bits = 0;
    for (const auto &submatrix_ptr : child_nodes_) {
        num_set_bits += submatrix_ptr->num_relations();
    }

    return num_set_bits;
}

// Function to get the average arity of the BRWT
double BRWT::avg_arity() const {
    if (!child_nodes_.size())
        return 0;

    uint64_t num_nodes = 0;
    uint64_t total_num_child_nodes = 0;

    BFT([&](const BRWT &node) {
        if (node.child_nodes_.size()) {
            num_nodes++;
            total_num_child_nodes += node.child_nodes_.size();
        }
    });

    return num_nodes
            ? static_cast<double>(total_num_child_nodes) / num_nodes
            : 0;
}

// Function to get the number of nodes in the BRWT
uint64_t BRWT::num_nodes() const {
    uint64_t num_nodes = 0;

    BFT([&num_nodes](const BRWT &) { num_nodes++; });

    return num_nodes;
}

// Function to get the shrinking rate of the BRWT
double BRWT::shrinking_rate() const {
    double rate_sum = 0;
    uint64_t num_nodes = 0;

    BFT([&](const BRWT &node) {
        if (node.child_nodes_.size()) {
            num_nodes++;
            rate_sum += static_cast<double>(node.nonzero_rows_->num_set_bits())
                            / node.nonzero_rows_->size();
        }
    });

    return rate_sum / num_nodes;
}

// Function to print the tree structure of the BRWT
// void BRWT::print_tree_structure(std::ostream &os) const {
//     BFT([&os](const BRWT &node) {
//         // Print node and its stats
//         os << &node << "," << node.nonzero_rows_->size()
//                     << "," << node.nonzero_rows_->num_set_bits(); // num_set_bits: number of ones in the matrix
//         // Print all its children
//         for (const auto &child : node.child_nodes_) {
//             os << "," << child.get();
//         }
//         os << std::endl;
//     });
// }
// void BRWT::print_tree_structure(std::ostream &os) const {
//     BFT([&os](const BRWT &node) {
//         // Print current node information
//         // os << "Node: " << &node << "\n";
//         // os << "  Nonzero Rows: " << node.nonzero_rows_->size() << "\n";
//         // os << "  Number of Ones: " << node.nonzero_rows_->num_set_bits() << "\n";

//         // Print the content of nonzero_rows_ (bit vector)
//         os << "  Content of nonzero_rows_: ";
//         for (size_t i = 0; i < node.nonzero_rows_->size(); ++i) {
//             os << (*node.nonzero_rows_)[i];
//         }
//         // os << "\n";

//         // Print child pointers
//         // os << "  Children: ";
//         // for (const auto &child : node.child_nodes_) {
//             // os << child.get() << " ";
//         // }
//         os << "\n";
//     });
// }
void BRWT::print_tree_structure(std::ostream &os) const {
    int current_level = 0; // Keep track of the current level

    // Breadth-First Traversal with a lambda for printing
    BFT([&os, &current_level](const BRWT &node) {
        // Determine the level of the current node based on its depth in the tree
        static std::unordered_map<const BRWT*, int> levels;
        if (levels.find(&node) == levels.end()) {
            levels[&node] = current_level;
        }
        int node_level = levels[&node];
        
        // Print level and indent appropriately
        os << std::string(node_level * 2, ' '); // Indentation (2 spaces per level)
        os << "Level " << node_level << ": ";

        // Print the content of nonzero_rows_ (bit vector)
        for (size_t i = 0; i < node.nonzero_rows_->size(); ++i) {
            os << (*node.nonzero_rows_)[i] << " ";
        }
        os << "\n";

        // Assign levels for children
        for (const auto &child : node.child_nodes_) {
            if (child) { // Ensure the child is not null
                levels[child.get()] = node_level + 1;
            }
        }
    });
}

// Breadth-First Traversal (BFT) of the BRWT
void BRWT::BFT(std::function<void(const BRWT &node)> callback) const {
    std::queue<const BRWT*> nodes_queue;
    nodes_queue.push(this);

    while (!nodes_queue.empty()) {
        const auto &node = *nodes_queue.front();

        callback(node);

        for (const auto &child_node : node.child_nodes_) {
            const auto *brwt_node_ptr = dynamic_cast<const BRWT*>(child_node.get());
            if (brwt_node_ptr)
                nodes_queue.push(brwt_node_ptr);
        }
        nodes_queue.pop();
    }
}

void BRWT::update_merge(BRWT& other) {
    if (other.num_rows() < num_rows()) {
        throw std::runtime_error("New BRWT must have at least as many rows as old BRWT");
    }

    // If new BRWT has more rows, we simply pad the current one with zeros
    // This is valid because the internal structure (children) only depends on the set bits (ranks),
    // and appending zeros involves no new set bits, so ranks of existing 1s are preserved.
    if (other.num_rows() > num_rows()) {
        auto old_bv = nonzero_rows_->to_vector(); // Convert to sdsl::bit_vector
        sdsl::bit_vector new_bv(other.num_rows(), 0);
        // Copy old bits
        for (size_t i = 0; i < old_bv.size(); ++i) {
            if (old_bv[i]) new_bv[i] = 1;
        }
        // Replace the nonzero_rows_ vector
        nonzero_rows_ = std::make_unique<bit_vector_smallrank>(std::move(new_bv));
    }

    std::vector<BRWT> submatrices;
    submatrices.reserve(2);

    // Move current BRWT content into a temp object
    BRWT self_moved;
    self_moved.assignments_ = std::move(assignments_);
    self_moved.nonzero_rows_ = std::move(nonzero_rows_);
    self_moved.child_nodes_ = std::move(child_nodes_);

    submatrices.push_back(std::move(self_moved));
    submatrices.push_back(std::move(other));

    ThreadPool pool(std::thread::hardware_concurrency());
    // Allocate buffer for concatenation (needs size of num_rows)
    sdsl::bit_vector buffer(submatrices[0].num_rows(), 0);

    // Concatenate and assign back to this
    *this = BRWTBottomUpBuilder::concatenate(std::move(submatrices), &buffer, pool);
}

} // namespace matrix
} // namespace annot
} // namespace mtg