#include "clustering.hpp"

#include <algorithm>
#include <random>
#include <unordered_set>

#include <ips4o.hpp>
#include <progress_bar.hpp>

#include "common/logger.hpp"
#include "common/algorithms.hpp"
#include "common/vectors/vector_algorithm.hpp"


namespace mtg {
namespace annot {
namespace matrix {

using mtg::common::logger;

typedef std::vector<std::vector<uint64_t>> Partition;
typedef std::vector<const bit_vector *> VectorPtrs;


std::vector<sdsl::bit_vector>
get_submatrix(const VectorPtrs &columns,
              const std::vector<uint64_t> &row_indexes,
              size_t num_threads) {
    assert(std::is_sorted(row_indexes.begin(), row_indexes.end()));

    if (!columns.size())
        return {};

    assert(row_indexes.size() <= columns[0]->size());

    std::vector<sdsl::bit_vector> submatrix(columns.size());

    ProgressBar progress_bar(columns.size(), "Subsampling",
                             std::cerr, !common::get_verbose());

    #pragma omp parallel for num_threads(num_threads)
    for (size_t i = 0; i < columns.size(); ++i) {
        const bit_vector &col = *columns[i];
        sdsl::bit_vector &subvector = submatrix[i];

        assert(row_indexes.size() <= col.size());

        subvector = sdsl::bit_vector(row_indexes.size(), false);

        for (size_t j = 0; j < row_indexes.size(); ++j) {
            if (col[row_indexes[j]])
                subvector[j] = true;
        }

        ++progress_bar;
    }

    return submatrix;
}

// returns shrunk columns
std::vector<uint64_t>
sample_row_indexes(uint64_t num_rows, uint64_t num_samples, int seed) {
    std::mt19937 gen(seed);

    num_samples = std::min(num_samples, num_rows);

    auto indexes = utils::sample_indexes(num_rows, num_samples, gen);
    // sort indexes
    std::sort(indexes.begin(), indexes.end());
    // check if indexes are sampled without replacement
    assert(std::unique(indexes.begin(), indexes.end()) == indexes.end());

    return indexes;
}

// returns shrinked columns
std::vector<sdsl::bit_vector>
random_submatrix(const VectorPtrs &columns,
                 uint64_t num_rows_sampled, size_t num_threads, int seed) {
    if (!columns.size())
        return {};

    auto indexes = sample_row_indexes(columns[0]->size(), num_rows_sampled, seed);

    return get_submatrix(columns, indexes, num_threads);
}


// Partitionings for Multi-BRWT

// input: columns
// output: partition, for instance -- a set of column pairs
std::vector<uint64_t> inverted_arrangement(const VectorPtrs &vectors) {
    auto init_arrangement
            = utils::arange<uint64_t>(0, vectors.size());

    return { init_arrangement.rbegin(), init_arrangement.rend() };
}

// Compute the number of shared set bits divided by the size of the bitmaps
double intersection_ratio(const sdsl::bit_vector &first, const sdsl::bit_vector &second) {
    assert(first.size() == second.size());
    return static_cast<double>(::inner_prod(first, second)) / first.size();
}

// Estimate the number of shared set bits divided by the bitmap size.
// |first| and |second| are assumed to have the same origin (starting position),
// but one of them may be shorter than the other.
// The result is equivalent to resizing the longest vector to even the sizes
// and computing the intersection ratio as usual.
double intersection_ratio(const SparseColumn &first, const SparseColumn &second) {
    const auto &[size_1, col_1] = first;
    const auto &[size_2, col_2] = second;
    auto it_1 = col_1.begin();
    auto it_2 = col_2.begin();

    if (!size_1 || !size_2)
        throw std::runtime_error("Vector size must be non-zero");

    uint64_t prod = 0;

    while (it_1 != col_1.end() && it_2 != col_2.end()) {
        assert(*it_1 < size_1 && *it_2 < size_2);
        if (*it_1 < *it_2) {
            ++it_1;
        } else if (*it_1 > *it_2) {
            ++it_2;
        } else {
            prod++;
            ++it_1;
            ++it_2;
        }
    }

    return static_cast<double>(prod) / std::min(size_1, size_2);
}

template <class T>
std::vector<std::tuple<uint32_t, uint32_t, float>>
correlation_similarity(const std::vector<T> &cols, size_t num_threads) {
    if (cols.size() > std::numeric_limits<uint32_t>::max()) {
        std::cerr << "ERROR: too many columns" << std::endl;
        exit(1);
    }

    if (!cols.size())
        return {};

    std::vector<std::tuple<uint32_t, uint32_t, float>>
            similarities(cols.size() * (cols.size() - 1) / 2);

    ProgressBar progress_bar(similarities.size(), "Correlations",
                             std::cerr, !common::get_verbose());

    #pragma omp parallel for num_threads(num_threads) collapse(2) schedule(static, 5)
    for (uint64_t j = 1; j < cols.size(); ++j) {
        for (uint64_t i = 0; i < cols.size(); ++i) {
            if (i < j) {
                float sim = intersection_ratio(cols[i], cols[j]);
                similarities[(j - 1) * j / 2 + i] = std::tie(i, j, sim);
                ++progress_bar;
            }
        }
    }

    return similarities;
}

std::vector<std::vector<double>>
jaccard_similarity(const std::vector<sdsl::bit_vector> &cols, size_t num_threads) {
    std::vector<std::vector<double>> similarities(cols.size());

    for (size_t j = 1; j < cols.size(); ++j) {
        similarities[j].assign(j, 0);
    }

    std::vector<uint64_t> num_set_bits(cols.size(), 0);

    #pragma omp parallel for num_threads(num_threads)
    for (size_t j = 0; j < cols.size(); ++j) {
        num_set_bits[j] = sdsl::util::cnt_one_bits(cols[j]);
    }

    ProgressBar progress_bar(cols.size() * (cols.size() - 1) / 2, "Jaccard",
                             std::cerr, !common::get_verbose());

    #pragma omp parallel for num_threads(num_threads) collapse(2) schedule(static, 5)
    for (size_t j = 0; j < cols.size(); ++j) {
        for (size_t k = 0; k < cols.size(); ++k) {
            if (k >= j)
                continue;

            uint64_t intersect = inner_prod(cols[j], cols[k]);
            similarities[j][k]
                = intersect / (num_set_bits[j] + num_set_bits[k] - intersect);
            ++progress_bar;
        }
    }

    return similarities;
}

template <typename T>
inline T dist(T first, T second) {
    return first > second
                ? first - second
                : second - first;
}

template <typename P>
inline bool first_closest(const P &first, const P &second) {
    auto first_dist = dist(std::get<0>(first), std::get<1>(first));
    auto second_dist = dist(std::get<0>(second), std::get<1>(second));
    return first_dist < second_dist
        || (first_dist == second_dist
                && std::min(std::get<0>(first), std::get<1>(first))
                    < std::min(std::get<0>(second), std::get<1>(second)));
}

// Input: columns, where each column `T` is either `sdsl::bit_vector` or
// `SparseColumn` storing the column size and the positions of its set bits.
// Output: a set of greedily matched column pairs.
template <class T>
Partition greedy_matching(const std::vector<T> &columns, size_t num_threads) {
    if (!columns.size())
        return {};

    if (columns.size() > std::numeric_limits<uint32_t>::max()) {
        std::cerr << "ERROR: too many columns" << std::endl;
        exit(1);
    }

    auto similarities = correlation_similarity(columns, num_threads);

    ProgressBar progress_bar(similarities.size(), "Matching",
                             std::cerr, !common::get_verbose());

    // pick either a pair of the most similar columns,
    // or pair closest in the initial arrangement
    ips4o::parallel::sort(similarities.begin(), similarities.end(),
        [](const auto &first, const auto &second) {
              return std::get<2>(first) > std::get<2>(second)
                || (std::get<2>(first) == std::get<2>(second)
                        && first_closest(first, second));
        },
        num_threads
    );

    Partition partition;
    partition.reserve((columns.size() + 1) / 2);

    std::vector<uint_fast8_t> matched(columns.size(), false);

    for (const auto &[i, j, sim] : similarities) {
        if (!matched[i] && !matched[j]) {
            matched[i] = matched[j] = true;
            partition.push_back({ i, j });
        }
        ++progress_bar;
    }

    for (size_t i = 0; i < columns.size(); ++i) {
        if (!matched[i])
            partition.push_back({ i });
    }

    return partition;
}

void union_merge(const sdsl::bit_vector &first, sdsl::bit_vector *second) {
    assert(second);
    assert(first.size() == second->size());
    *second |= first;
}

void union_merge(const SparseColumn &first, SparseColumn *second) {
    assert(second);

    const auto &col_first = first.set_bits;
    const auto &col_second = second->set_bits;

    SparseColumn merged;
    merged.size = std::min(first.size, second->size);
    merged.set_bits.reserve(col_first.size() + col_second.size());

    auto first_end = std::lower_bound(col_first.begin(), col_first.end(), merged.size);
    auto second_end = std::lower_bound(col_second.begin(), col_second.end(), merged.size);
    std::set_union(col_first.begin(), first_end,
                   col_second.begin(), second_end,
                   std::back_inserter(merged.set_bits));

    std::swap(*second, merged);
}

uint64_t count_set_bits(const sdsl::bit_vector &v) {
    return sdsl::util::cnt_one_bits(v);
}

uint64_t count_set_bits(const SparseColumn &v) {
    return v.set_bits.size();
}

template <class T>
LinkageMatrix agglomerative_greedy_linkage(std::vector<T>&& columns, size_t num_threads) {
    if (columns.empty())
        return LinkageMatrix(0, 4);

    LinkageMatrix linkage_matrix(columns.size() - 1, 4);
    size_t i = 0;

    uint64_t num_clusters = columns.size();
    std::vector<uint64_t> column_ids
            = utils::arange<uint64_t>(0, columns.size());

    for (size_t level = 1; columns.size() > 1; ++level) {
        logger->info("Clustering: level {}", level);

        Partition groups = greedy_matching(columns, num_threads);

        assert(groups.size() > 0);
        assert(groups.size() < columns.size());

        std::vector<T> cluster_centers(groups.size());
        std::vector<uint64_t> cluster_ids(groups.size());

        ProgressBar progress_bar(groups.size(), "Merging clusters",
                                 std::cerr, !common::get_verbose());

        #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
        for (size_t g = 0; g < groups.size(); ++g) {
            // merge into new clusters
            cluster_centers[g] = std::move(columns[groups[g].at(0)]);
            for (size_t i = 1; i < groups[g].size(); ++i) {
                union_merge(columns[groups[g][i]], &cluster_centers[g]);
                columns[groups[g][i]] = T();
            }

            uint64_t num_set_bits = count_set_bits(cluster_centers[g]);

            #pragma omp critical
            {
                if (groups[g].size() > 1) {
                    assert(groups[g].size() == 2);
                    cluster_ids[g] = num_clusters;
                    linkage_matrix(i, 0) = column_ids[groups[g][0]];
                    linkage_matrix(i, 1) = column_ids[groups[g][1]];
                    linkage_matrix(i, 2) = num_set_bits;
                    linkage_matrix(i, 3) = cluster_ids[g];
                    num_clusters++;
                    i++;
                } else {
                    assert(groups[g].size() == 1);
                    cluster_ids[g] = column_ids[groups[g][0]];
                }
            }

            ++progress_bar;
        }

        columns.swap(cluster_centers);
        column_ids.swap(cluster_ids);
    }

    assert(i == static_cast<size_t>(linkage_matrix.rows()));

    return linkage_matrix;
}

// --- 1) Limited‐neighbor greedy matching (O(k·N) sims per round) ---
template<class T>
mtg::annot::matrix::Partition
greedy_matching_limited(const std::vector<T> &columns,
                        size_t num_threads,
                        size_t k = 10,
                        uint64_t seed = 42)
{
    uint32_t N = columns.size();
    if (N <= 1) return {};

    struct Edge { uint32_t i, j; float sim; };
    std::vector<Edge> best(N);

    // clamp k
    size_t real_k = std::min(k, (size_t)N - 1);

    // prepare per‐thread RNG seeds
    std::vector<uint64_t> rnd_seeds(num_threads);
    {
        std::mt19937_64 tmp(seed);
        for (size_t t = 0; t < num_threads; ++t)
            rnd_seeds[t] = tmp();
    }

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        std::mt19937_64 gen(rnd_seeds[tid]);
        std::uniform_int_distribution<uint32_t> dist(0, N-2);

        #pragma omp for schedule(static)
        for (uint32_t i = 0; i < N; ++i) {
            float  best_sim = -1.0f;
            uint32_t best_j = i;

            if (real_k == N - 1) {
                // exact scan over all j ≠ i
                for (uint32_t j = 0; j < N; ++j) {
                    if (j == i) continue;
                    float sim = intersection_ratio(columns[i], columns[j]);
                    if (sim > best_sim
                        || (sim == best_sim
                            && first_closest(std::make_pair(i,j),
                                             std::make_pair(i,best_j))))
                    {
                        best_sim = sim;
                        best_j   = j;
                    }
                }
            } else {
                // pick real_k distinct random j ≠ i
                std::unordered_set<uint32_t> picks;
                picks.reserve(real_k);
                while (picks.size() < real_k) {
                    uint32_t x = dist(gen);
                    uint32_t y = (x < i ? x : x + 1);
                    picks.insert(y);
                }
                for (auto j : picks) {
                    float sim = intersection_ratio(columns[i], columns[j]);
                    if (sim > best_sim
                        || (sim == best_sim
                            && first_closest(std::make_pair(i,j),
                                             std::make_pair(i,best_j))))
                    {
                        best_sim = sim;
                        best_j   = j;
                    }
                }
            }

            best[i] = { i, best_j, best_sim };
        }
    } // omp

    // sort only N edges → O(N log N)
    std::sort(best.begin(), best.end(),
        [&](auto &A, auto &B) {
            if (A.sim != B.sim) return A.sim > B.sim;
            return first_closest(std::make_pair(A.i,A.j),
                                 std::make_pair(B.i,B.j));
        });

    // greedy match
    std::vector<uint8_t> used(N, 0);
    mtg::annot::matrix::Partition P;
    P.reserve((N+1)/2);
    for (auto &e : best) {
        if (!used[e.i] && !used[e.j]) {
            used[e.i] = used[e.j] = 1;
            P.push_back({ e.i, e.j });
        }
    }
    // singletons
    for (uint32_t i = 0; i < N; ++i)
        if (!used[i])
            P.push_back({ i });

    return P;
}

// --- 2) Modified agglomerative_greedy_linkage to call the above ---
template <class T>
LinkageMatrix agglomerative_greedy_linkage_k(std::vector<T>&& columns,
                                           size_t num_threads, size_t k, uint64_t seed)
{
    // k is the number of neighbors to consider for each column
    // at each round of clustering
    // seed is for RNG
    assert(k > 0);
    
    size_t N = columns.size();
    if (N < 2) return LinkageMatrix(0,4);

    LinkageMatrix linkage_matrix(N-1, 4);
    size_t linkage_row = 0;
    uint64_t next_cluster_id = N;

    // track current IDs of each “column”
    std::vector<uint64_t> column_ids(N);
    std::iota(column_ids.begin(), column_ids.end(), 0ULL);

    size_t level = 1;

    while (columns.size() > 1) {
        logger->info("Clustering level {}: {} clusters", level, columns.size());

        // 1) form pairs by testing only k candidates each
        Partition groups = greedy_matching_limited(columns, num_threads, k, seed + level);

        // 2) merge paired columns
        size_t G = groups.size();
        std::vector<T> new_centers(G);
        std::vector<uint64_t> new_ids(G);

        ProgressBar pb(G, "Merging clusters", std::cout, !common::get_verbose());

        #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
        for (size_t g = 0; g < G; ++g) {
            // start with the first member of the group
            new_centers[g] = std::move(columns[groups[g][0]]);

            // merge any others in this group
            for (size_t t = 1; t < groups[g].size(); ++t) {
                union_merge(columns[groups[g][t]], &new_centers[g]);
                columns[groups[g][t]] = T(); // free memory
            }

            uint64_t bits = count_set_bits(new_centers[g]);

            #pragma omp critical
            {
                if (groups[g].size() == 2) {
                    // a true merge → record in linkage matrix
                    linkage_matrix(linkage_row,0) = column_ids[groups[g][0]];
                    linkage_matrix(linkage_row,1) = column_ids[groups[g][1]];
                    linkage_matrix(linkage_row,2) = bits;
                    linkage_matrix(linkage_row,3) = next_cluster_id;
                    new_ids[g] = next_cluster_id++;
                    linkage_row++;
                } else {
                    // singleton carried forward
                    new_ids[g] = column_ids[groups[g][0]];
                }
            }
            ++pb;
        }

        // swap in the new level
        columns.swap(new_centers);
        column_ids.swap(new_ids);
        level++;
    }

    assert(linkage_row == linkage_matrix.rows());
    return linkage_matrix;
}

template
LinkageMatrix agglomerative_greedy_linkage(std::vector<sdsl::bit_vector>&&, size_t);

template
LinkageMatrix agglomerative_greedy_linkage(std::vector<SparseColumn>&&, size_t);

template
LinkageMatrix agglomerative_greedy_linkage_k(std::vector<sdsl::bit_vector>&&, size_t, size_t, uint64_t);

template
LinkageMatrix agglomerative_greedy_linkage_k(std::vector<SparseColumn>&&, size_t, size_t, uint64_t);


LinkageMatrix agglomerative_linkage_trivial(size_t num_columns) {
    if (!num_columns)
        return LinkageMatrix(0, 4);

    LinkageMatrix linkage_matrix(num_columns - 1, 4);
    size_t i = 0;

    uint64_t num_clusters = num_columns;
    std::vector<uint64_t> column_ids
            = utils::arange<uint64_t>(0, num_columns);

    for (size_t level = 1; column_ids.size() > 1; ++level) {
        logger->trace("Clustering: level {}", level);

        Partition groups((column_ids.size() - 1) / 2 + 1);
        for (size_t j = 0; j < column_ids.size(); ++j) {
            groups[j / 2].push_back(j);
        }

        assert(groups.size() > 0);
        assert(groups.size() < column_ids.size());

        std::vector<uint64_t> cluster_ids(groups.size());

        for (size_t g = 0; g < groups.size(); ++g) {
            // merge into new clusters
            if (groups[g].size() > 1) {
                assert(groups[g].size() == 2);
                cluster_ids[g] = num_clusters;
                linkage_matrix(i, 0) = column_ids[groups[g][0]];
                linkage_matrix(i, 1) = column_ids[groups[g][1]];
                linkage_matrix(i, 2) = 0;
                linkage_matrix(i, 3) = cluster_ids[g];
                num_clusters++;
                i++;
            } else {
                assert(groups[g].size() == 1);
                cluster_ids[g] = column_ids[groups[g][0]];
            }
        }

        column_ids.swap(cluster_ids);
    }

    assert(i == static_cast<size_t>(linkage_matrix.rows()));

    return linkage_matrix;
}

} // namespace matrix
} // namespace annot
} // namespace mtg
