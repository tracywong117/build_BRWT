// Note: 
// 1. Promoted BRWTBottomUpBuilder::concatenate and BRWTBottomUpBuilder::concatenate_sparse to public
#include <spdlog/sinks/stdout_color_sinks.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <filesystem>
#include <cassert>
#include <thread>
#include <fstream>
#include <mutex>
#include <Eigen/Dense>

#include <server_http.hpp>
#include <json/json.h>

#include "common/logger.hpp"
#include "common/unix_tools.hpp"
#include "cli/server_utils.hpp"

#include "annotation/binary_matrix/multi_brwt/clustering.hpp"
#include "annotation/binary_matrix/multi_brwt/brwt_builders.hpp"
#include "common/vectors/vector_algorithm.hpp"
#include "common/vectors/bit_vector_sdsl.hpp"
#include "common/vectors/bit_vector_sd.hpp"

namespace fs = std::filesystem;

namespace {
using namespace mtg;
using namespace mtg::annot::matrix;
using HttpServer = SimpleWeb::Server<SimpleWeb::HTTP>;

using mtg::common::logger;
using mtg::cli::parse_json_string; // (and, if you use them)
using mtg::cli::process_request; //     for process_request

static const Eigen::IOFormat
        CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", "\n");


// Helper: Sample a set of bits from a file using given row indexes
// for bit vector file
// sdsl::bit_vector sample_subcolumn_from_file(const std::string& file_path,
//                                             const std::vector<uint64_t>& row_indexes) {
//     sdsl::bit_vector subvec(row_indexes.size(), 0);
//     std::ifstream file(file_path, std::ios::binary);
//     if (!file)
//         throw std::runtime_error("Could not open file: " + file_path);
//     for (size_t j = 0; j < row_indexes.size(); ++j) {
//         size_t bit_index = row_indexes[j];
//         size_t byte_index = bit_index / 8;
//         size_t bit_offset = bit_index % 8;
//         file.seekg(byte_index);
//         char byte; // LSB first
//         // unsigned char byte; // MSB first
//         file.read(reinterpret_cast<char*>(&byte), 1);
//         subvec[j] = ((byte >> (bit_offset)) & 1); // LSB first
//         // subvec[j] = ((byte >> (7 - bit_offset)) & 1); // MSB first
//     }
//     return subvec;
// }

// Helper: Sample a set of bits from a file storing an sd_vector
// for sd_vector file
// method 2: directly sample from sd_vector without converting to bit_vector
sdsl::bit_vector sample_subcolumn_from_file(const std::string& file_path,
                                            const std::vector<uint64_t>& row_indexes) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file)
        throw std::runtime_error("Could not open file: " + file_path);

    sdsl::sd_vector<> sdvec;
    sdsl::load(sdvec, file);

    // Directly sample bits from sd_vector without converting to bit_vector
    sdsl::sd_vector<>::rank_1_type rank1(&sdvec);
    sdsl::sd_vector<>::select_1_type select_1(&sdvec); // UNUSED and potentially problematic

    sdsl::bit_vector subvec(row_indexes.size(), 0);
    for (size_t j = 0; j < row_indexes.size(); ++j) {
        uint64_t bit_index = row_indexes[j];
        if (bit_index >= sdvec.size())
            throw std::out_of_range("Requested index out of bounds in sd_vector");
        subvec[j] = rank1(bit_index + 1) - rank1(bit_index) > 0 ? 1 : 0;
    }
    return subvec;
}

// // Helper: Sample a set of bits from a file storing an sd_vector
// // for sd_vector file
// sdsl::bit_vector sample_subcolumn_from_file(const std::string& file_path,
//                                             const std::vector<uint64_t>& row_indexes) {
//     // Load sd_vector from file
//     std::ifstream file(file_path, std::ios::binary);
//     if (!file)
//         throw std::runtime_error("Could not open file: " + file_path);

//     sdsl::sd_vector<> sdvec;
//     sdsl::load(sdvec, file);

//     // Convert sd_vector<> to bit_vector
//     sdsl::bit_vector bv(sdvec.size(), 0);
//     sdsl::sd_vector<>::select_1_type select_1(&sdvec);
//     sdsl::sd_vector<>::rank_1_type rank1(&sdvec);
//     uint64_t ones = rank1(sdvec.size());
//     for (uint64_t rank = 1; rank <= ones; ++rank) {
//         bv[select_1(rank)] = 1;
//     }

//     // Now sample the requested bits
//     sdsl::bit_vector subvec(row_indexes.size(), 0);
//     for (size_t j = 0; j < row_indexes.size(); ++j) {
//         uint64_t bit_index = row_indexes[j];
//         if (bit_index < bv.size())
//             subvec[j] = bv[bit_index];
//         else
//             throw std::out_of_range("Requested index out of bounds in bit vector");
//     }
//     return subvec;
// }


void BM_BRWTLinkageMatrix_streaming(const std::string& folder_path,
                                    const std::string& prefix,
                                    const std::string& linkage_file,
                                    const std::string& file_list,
                                    size_t num_threads,
                                    size_t linkage_k,
                                    size_t linkage_seed,
                                    bool linkage_trivial) {
    namespace fs = std::filesystem;
    std::vector<std::string> file_paths;

    // If file_list is provided, use only those files
    if (!file_list.empty()) {
        std::ifstream flist(file_list);
        if (!flist)
            throw std::runtime_error("Could not open file_list: " + file_list);
        std::string line;
        while (std::getline(flist, line)) {
            if (!line.empty()) {
                fs::path p = fs::path(folder_path) / line;
                if (fs::exists(p) && fs::is_regular_file(p))
                    file_paths.push_back(p.string());
                else
                    throw std::runtime_error("File in file_list not found: " + p.string());
            }
        }
    } else {
        // Find all files with prefix
        for (const auto& entry : fs::directory_iterator(folder_path)) {
            if (entry.is_regular_file()) {
                const std::string file_name = entry.path().filename().string();
                if (file_name.rfind(prefix, 0) == 0) {
                    file_paths.push_back(entry.path().string());
                }
            }
        }
    }

    // Debug: print first 10 file paths
    logger->info("Found {} files with prefix '{}'", file_paths.size(), prefix);
    // for (size_t i = 0; i < std::min(file_paths.size(), size_t(10)); ++i) {
    //     logger->info("  {}", file_paths[i]);
    // }

    if (file_paths.empty())
        throw std::runtime_error("No files found.");

    // Determine column length from first file
    // for bit vector file
    // std::ifstream first_file(file_paths[0], std::ios::binary | std::ios::ate);
    // if (!first_file)
    //     throw std::runtime_error("Failed to open: " + file_paths[0]);
    // uint64_t column_length = (uint64_t)first_file.tellg() * 8;
    // first_file.close();
    // =========================================
    // for sd_vector file
    std::ifstream first_sd_file(file_paths[0], std::ios::binary);
    if (!first_sd_file)
        throw std::runtime_error("Failed to open: " + file_paths[0]);
    sdsl::sd_vector<> sdvec;
    sdsl::load(sdvec, first_sd_file);
    uint64_t column_length = sdvec.size();

    logger->info("Column length determined from first file: {}", column_length);

    // Sample row indexes once
    std::vector<uint64_t> row_indexes
            = annot::matrix::sample_row_indexes(column_length, 1'000'000);

    // Prepare storage for subcolumns
    logger->info("Sampling subcolumns in parallel (method 2) using {} threads...", num_threads);
    auto t_start = std::chrono::high_resolution_clock::now();

    std::vector<std::unique_ptr<sdsl::bit_vector>> subcolumn_ptrs(file_paths.size());
    std::mutex mu;
    ThreadPool pool(num_threads);

    // For each file, sample subcolumn in parallel
    for (size_t i = 0; i < file_paths.size(); ++i) {
        pool.enqueue([&, i]() {
            try {
                auto subvec = std::make_unique<sdsl::bit_vector>(
                        sample_subcolumn_from_file(file_paths[i], row_indexes));
                {
                    std::lock_guard<std::mutex> lock(mu);
                    subcolumn_ptrs[i] = std::move(subvec);
                    if ((i + 1) % 1000 == 0 || i + 1 == file_paths.size()) {
                        logger->info("Sampled {} subcolumns...", i + 1);
                    }
                }
            } catch (const std::bad_alloc& e) {
                logger->error("Memory allocation failed for file {}: {}", file_paths[i], e.what());
                throw;
            } catch (const std::exception& e) {
                logger->error("Error processing file {}: {}", file_paths[i], e.what());
                throw; // Rethrow to let the pool/app handle it (or crash)
            }
        });
    }
    pool.join();

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(t_end - t_start).count();
    logger->info("Sampling subcolumns finished in {:.2f} seconds.", elapsed_sec);

    // Rearrange and continue as before
    logger->info("Rearranging subcolumns...");
    auto t_rearrange_start = std::chrono::high_resolution_clock::now();
    std::vector<sdsl::bit_vector> subcolumns(subcolumn_ptrs.size());
    for (size_t i = 0; i < subcolumn_ptrs.size(); ++i)
        subcolumns[i] = std::move(*subcolumn_ptrs[i]);
    auto t_rearrange_end = std::chrono::high_resolution_clock::now();
    double rearrange_sec = std::chrono::duration<double>(t_rearrange_end - t_rearrange_start).count();
    logger->info("Rearranging subcolumns finished in {:.2f} seconds.", rearrange_sec);

    // Compute linkage matrix (assuming your original code here)
    logger->info("Computing linkage matrix for {} columns.", subcolumns.size());
    LinkageMatrix linkage_matrix;
    if (linkage_trivial) {
        linkage_matrix = annot::matrix::agglomerative_linkage_trivial(subcolumns.size());
    } else if (linkage_k > 0) {
        linkage_matrix = annot::matrix::agglomerative_greedy_linkage_k(std::move(subcolumns),
                                                                       get_num_threads(), linkage_k, linkage_seed);
    } else {
        linkage_matrix = annot::matrix::agglomerative_greedy_linkage(std::move(subcolumns),
                                                                     get_num_threads());
    }

    // Output as before
    std::ofstream out(linkage_file);
    out << linkage_matrix.format(CSVFormat) << std::endl;
}


std::vector<std::string> split_string(const std::string& string,
                                      const std::string& delimiter,
                                      bool skip_empty_parts) {
    if (!string.size())
        return {};

    if (!delimiter.size())
        return {
            string,
        };

    std::vector<std::string> result;

    size_t current_pos = 0;
    size_t delimiter_pos;

    while ((delimiter_pos = string.find(delimiter, current_pos)) != std::string::npos) {
        if (delimiter_pos > current_pos || !skip_empty_parts)
            result.push_back(string.substr(current_pos, delimiter_pos - current_pos));
        current_pos = delimiter_pos + delimiter.size();
    }
    if (current_pos < string.size()) {
        result.push_back(string.substr(current_pos));
    }

    assert(result.size());
    return result;
}


std::vector<std::vector<uint64_t>> parse_linkage_matrix(const std::string& filename) {
    std::ifstream in(filename);

    std::vector<std::vector<uint64_t>> linkage;
    std::string line;
    while (std::getline(in, line)) {
        std::vector<std::string> parts = split_string(line, " ", true);
        if (parts.empty())
            continue;

        try {
            if (parts.size() != 4)
                throw std::runtime_error("Invalid format");

            uint64_t first = std::stoi(parts.at(0));
            uint64_t second = std::stoi(parts.at(1));
            uint64_t merged = std::stoi(parts.at(3));

            if (first == second || first >= merged || second >= merged) {
                exit(1);
            }

            while (linkage.size() <= merged) {
                linkage.push_back({});
            }

            linkage[merged].push_back(first);
            linkage[merged].push_back(second);

        } catch (const std::exception& e) {
            exit(1);
        }
    }

    return linkage;
}

// Load all bit vector from files in a folder with a callback
// for bit vector file
// bool load_columns_from_folder_callback(
//         const std::string& folder_path,
//         const std::function<void(uint64_t, const std::string&, std::unique_ptr<bit_vector>&&)>& callback,
//         size_t num_threads,
//         const std::string& prefix) {
//     namespace fs = std::filesystem;

//     // Find all files in the folder with the prefix "anno" and extension ".bin"
//     std::vector<std::string> file_paths;
//     for (const auto& entry : fs::directory_iterator(folder_path)) {
//         if (entry.is_regular_file()) {
//             const std::string file_name = entry.path().filename().string();
//             if (file_name.rfind(prefix, 0) == 0
//                 && file_name.substr(file_name.size() - 4) == ".bin") {
//                 file_paths.push_back(entry.path().string());
//             }
//         }
//     }

//     // Check if no matching files were found
//     if (file_paths.empty()) {
//         throw std::runtime_error("No files with prefix " + prefix
//                                  + " and extension '.bin' found in the folder.");
//         return false;
//     }

//     // Determine the column length from the size of the first file
//     const std::string first_file = file_paths[0];
//     std::ifstream first_file_stream(first_file,
//                                     std::ios::binary | std::ios::ate); // Open in binary mode
//     if (!first_file_stream.is_open()) {
//         throw std::runtime_error("Failed to open file: " + first_file);
//         return false;
//     }
//     std::streamsize file_size = first_file_stream.tellg(); // Get file size in bytes
//     first_file_stream.close();

//     uint64_t column_length = static_cast<uint64_t>(file_size) * 8; // Each byte is 8 bits

//     // Process files in parallel using OpenMP (if enabled)
//     bool success = true; // To track overall success
//     std::mutex callback_mutex; // Mutex for thread-safe callback calls

// #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
//     for (size_t i = 0; i < file_paths.size(); ++i) {
//         const auto& file_path = file_paths[i];

//         try {
//             // Open the file in binary mode
//             std::ifstream file_stream(file_path, std::ios::binary);
//             if (!file_stream.is_open()) {
// #pragma omp critical
//                 {
//                     throw std::runtime_error("Failed to open file: " + file_path);
//                     success = false;
//                 }
//                 continue;
//             }

//             // Read the binary data
//             std::vector<uint8_t> buffer(file_size);
//             file_stream.read(reinterpret_cast<char*>(buffer.data()), file_size);
//             file_stream.close();

//             // Create an `sdsl::bit_vector` for the column
//             sdsl::bit_vector column(column_length, 0);
//             for (size_t byte_idx = 0; byte_idx < buffer.size(); ++byte_idx) {
//                 for (int bit = 0; bit < 8; ++bit) {
//                     if (buffer[byte_idx] & (1 << bit)) {
//                         column[(byte_idx * 8) + bit] = 1;
//                     }
//                 }
//             }

//             // Wrap the `sdsl::bit_vector` in a `std::unique_ptr` of `bit_vector_stat`
//             auto column_ptr = std::make_unique<bit_vector_stat>(std::move(column));

//             // Generate a label for the column based on its file name
//             std::string label = fs::path(file_path).filename().string();

//             // Pass the column to the callback
//             {
//                 std::lock_guard<std::mutex> lock(callback_mutex);
//                 callback(i, label, std::move(column_ptr));
//             }
//         } catch (const std::exception& e) {
// #pragma omp critical
//             {
//                 throw std::runtime_error("Error processing file: " + file_path);
//                 success = false;
//             }
//         }
//     }

//     return success;
// }


// Load all sdsl::sd_vector<> from files in a folder with a callback
// for sd_vector file
bool load_columns_from_folder_callback(
        const std::string& folder_path,
        const std::string& file_list,
        const std::function<void(uint64_t, const std::string&, std::unique_ptr<bit_vector_stat>&&)>& callback,
        size_t num_threads,
        const std::string& prefix) {
    namespace fs = std::filesystem;

    // Find all files in the folder with the given prefix or from file_list
    std::vector<std::string> file_paths;
    if (!file_list.empty()) {
        std::ifstream flist(file_list);
        if (!flist)
            throw std::runtime_error("Could not open file_list: " + file_list);
        std::string line;
        while (std::getline(flist, line)) {
            if (!line.empty()) {
                fs::path p = fs::path(folder_path) / line;
                if (fs::exists(p) && fs::is_regular_file(p))
                    file_paths.push_back(p.string());
                else
                    throw std::runtime_error("File in file_list not found: " + p.string());
            }
        }
    } else {
        for (const auto& entry : fs::directory_iterator(folder_path)) {
            if (entry.is_regular_file()) {
                const std::string file_name = entry.path().filename().string();
                if (file_name.rfind(prefix, 0) == 0) { // Check if file_name starts with prefix
                    file_paths.push_back(entry.path().string());
                }
            }
        }
    }

    if (file_paths.empty()) {
        throw std::runtime_error("No files with prefix " + prefix
                                + " found in the folder.");
        return false;
    }

    // Determine the column length from the first file
    uint64_t column_length = 0;
    {
        const std::string first_file = file_paths[0];
        std::ifstream first_file_stream(first_file, std::ios::binary);
        if (!first_file_stream.is_open()) {
            throw std::runtime_error("Failed to open file: " + first_file);
            return false;
        }
        sdsl::sd_vector<> sdv;
        sdsl::load(sdv, first_file_stream);
        column_length = sdv.size();
        logger->info("Column length determined from first file: {}", column_length);
        first_file_stream.close();
    }

    bool success = true;
    std::mutex callback_mutex;

#pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (size_t i = 0; i < file_paths.size(); ++i) {
        const auto& file_path = file_paths[i];

        try {
            std::ifstream file_stream(file_path, std::ios::binary);
            if (!file_stream.is_open()) {
#pragma omp critical
                {
                    success = false;
                    throw std::runtime_error("Failed to open file: " + file_path);
                }
                continue;
            }

            // Load sd_vector from file
            sdsl::sd_vector<> sdvec;
            sdsl::load(sdvec, file_stream);

            // Optional: check size matches
            if (sdvec.size() != column_length) {
#pragma omp critical
                {
                    success = false;
                    throw std::runtime_error("Column length mismatch in file: " + file_path);
                }
                continue;
            }

            // Convert sd_vector<> to bit_vector
            sdsl::bit_vector bv(sdvec.size(), 0);
            sdsl::sd_vector<>::select_1_type select_1(
                    &sdvec); // select_1 is a functor that returns the position of the i-th 1 in the sd_vector
            sdsl::sd_vector<>::rank_1_type rank1(
                    &sdvec); // rank1 is a functor that returns the number of 1s in the sd_vector up to a given position
            uint64_t ones = rank1(sdvec.size()); // number of 1s
            for (uint64_t rank = 1; rank <= ones; ++rank) {
                bv[select_1(rank)] = 1;
            }

            // Wrap the bit_vector in your wrapper
            auto column_ptr = std::make_unique<bit_vector_stat>(std::move(bv));

            std::string label = fs::path(file_path).filename().string();

            {
                std::lock_guard<std::mutex> lock(callback_mutex);
                callback(i, label, std::move(column_ptr));
            }
        } catch (const std::exception& e) {
#pragma omp critical
            {
                success = false;
                throw std::runtime_error(std::string("Error processing file: ")
                                         + file_path + "\n" + e.what());
            }
        }
    }

    return success;
}

// Function to serialize std::vector<std::pair<uint64_t, std::string>> column_names
void serialize_column_names(const std::vector<std::pair<uint64_t, std::string>>& column_names,
                            const std::string& filename) {
    std::ofstream out_file(filename, std::ios::binary); // Open file in binary mode
    if (!out_file) {
        throw std::runtime_error("Failed to open file for writing");
    }

    // Write the number of pairs
    size_t size = column_names.size();
    out_file.write(reinterpret_cast<const char*>(&size), sizeof(size)); // Write size of vector

    // Write each pair's first and second elements
    for (const auto& pair : column_names) {
        out_file.write(reinterpret_cast<const char*>(&pair.first),
                       sizeof(pair.first)); // Write first element
        size_t length = pair.second.size(); // Get the length of the string
        out_file.write(reinterpret_cast<const char*>(&length), sizeof(length)); // Write length
        out_file.write(pair.second.data(), length); // Write string content
    }

    out_file.close();
}

// Build BRWT
void build_BRWT(const std::string columns_folder_path,
                const std::string prefix,
                const std::vector<std::vector<uint64_t>>& linkage,
                const std::string& tmp_path,
                const std::string& file_list,
                size_t num_nodes_parallel,
                size_t num_threads,
                const std::string& output_path) {
    std::unique_ptr<BRWT> binary_matrix;

    std::mutex mu; // Mutex for thread-safe access
    std::vector<std::pair<uint64_t, std::string>> column_names; // Store column index and label
    auto get_columns = [&](const BRWTBottomUpBuilder::CallColumn& call_column) {
        // Use load_columns_from_folder to process columns
        bool success = load_columns_from_folder_callback(
                columns_folder_path, file_list,
                [&](uint64_t j, const std::string& label,
                    std::unique_ptr<bit_vector>&& column) {
                    // j: column index, label: column label, column: column data
                    // Pass the column to the BRWT builder
                    call_column(j, std::move(column));

                    // Store column names (thread-safe)
                    std::lock_guard<std::mutex> lock(mu);
                    column_names.emplace_back(j, label);
                },
                num_threads, prefix);

        if (!success) {
            throw std::runtime_error("Failed to load columns from folder: "
                                     + columns_folder_path);
            exit(1); // Exit if there's an error
        }
    };
    auto t_start = std::chrono::high_resolution_clock::now();

    binary_matrix = std::make_unique<BRWT>(BRWTBottomUpBuilder::build(
            get_columns, linkage, tmp_path, num_nodes_parallel, num_threads));

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(t_end - t_start).count();
    logger->info("BRWT build time: {:.2f} seconds.", elapsed_sec);
    logger->info("BRWT stats: #rows = {}, #columns = {}, #relations = {}, avg_arity = {:.2f}, #nodes = {}, shrinking_rate = {:.2f}",
                 binary_matrix->num_rows(), binary_matrix->num_columns(),
                 binary_matrix->num_relations(), binary_matrix->avg_arity(),
                 binary_matrix->num_nodes(), binary_matrix->shrinking_rate());

    // Serialize the BRWT and measure time
    auto t_ser_start = std::chrono::high_resolution_clock::now();
    std::ofstream out(output_path + ".brwt", std::ios::binary);
    binary_matrix->serialize(out);
    auto t_ser_end = std::chrono::high_resolution_clock::now();
    double ser_elapsed_sec = std::chrono::duration<double>(t_ser_end - t_ser_start).count();
    logger->info("BRWT serialization time: {:.2f} seconds.", ser_elapsed_sec);

    // Serialize the column names
    serialize_column_names(column_names, output_path + ".columns");
}


// Function to deserialize std::vector<std::pair<uint64_t, std::string>> from a file
std::vector<std::pair<uint64_t, std::string>>
deserialize_column_names(const std::string& filename) {
    std::ifstream in_file(filename, std::ios::binary); // Open file in binary mode
    if (!in_file) {
        throw std::runtime_error("Failed to open file for reading");
    }

    // Read the size of the vector
    size_t size = 0;
    in_file.read(reinterpret_cast<char*>(&size), sizeof(size)); // Read size of vector

    std::vector<std::pair<uint64_t, std::string>> column_names;
    column_names.reserve(size); // Reserve space for efficiency

    // Read each pair's data
    for (size_t i = 0; i < size; ++i) {
        uint64_t first = 0;
        in_file.read(reinterpret_cast<char*>(&first), sizeof(first)); // Read first element

        size_t length = 0;
        in_file.read(reinterpret_cast<char*>(&length), sizeof(length)); // Read string length

        std::string second(length, '\0'); // Create string with `length` characters
        in_file.read(&second[0], length); // Read string content into `second`

        column_names.emplace_back(first, second); // Add pair to the vector
    }

    in_file.close();
    return column_names;
}


// Relax BRWT
void relax_brwt(const std::string& filename,
                uint64_t max_arity,
                size_t num_threads,
                const std::string& output_file) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open BRWT file: " + filename);
    }

    auto brwt = std::make_unique<BRWT>();
    brwt->load(in);

    // Relax the BRWT
    BRWTOptimizer::relax(brwt.get(), max_arity, num_threads);

    // Serialize the relaxed BRWT
    std::ofstream out(output_file, std::ios::binary);
    brwt->serialize(out);
}


// Function to handle the `build` command
void handle_build(const std::string& annotation_dir,
                  const std::string& annotation_file_prefix,
                  const std::string& output_file,
                  const std::string& tmp_dir,
                  const std::string& file_list,
                  size_t num_threads,
                  size_t linkage_k,
                  size_t linkage_seed,
                  bool linkage_trivial) {
    logger->info("Building BRWT from the provided files...");
    // Precompute the linkage matrix
    std::string linkage_file = output_file + ".linkage";
    logger->info("Computing linkage matrix...");
    BM_BRWTLinkageMatrix_streaming(annotation_dir, annotation_file_prefix, linkage_file, file_list,
                                   num_threads, linkage_k, linkage_seed, linkage_trivial);

    // Load the linkage matrix
    logger->info("Parsing linkage matrix from: {}", linkage_file);
    std::vector<std::vector<uint64_t>> linkage = parse_linkage_matrix(linkage_file);

    // Build the BRWT and serialize it
    logger->info("Building BRWT...");
    build_BRWT(annotation_dir, annotation_file_prefix, linkage, tmp_dir, file_list, num_threads, num_threads, output_file);

    logger->info("BRWT successfully built and serialized to: {}", output_file);
}


// Function to handle the `query` command
void handle_query(const std::string& row_ids,
                  const std::string& brwt_file,
                  const std::string& columns_file) {
    // Parse row IDs in the format {id_0},{id_1},...
    std::vector<uint64_t> rows_to_query;
    std::string cleaned_row_ids = row_ids;
    cleaned_row_ids.erase(std::remove(cleaned_row_ids.begin(), cleaned_row_ids.end(), '{'),
                          cleaned_row_ids.end());
    cleaned_row_ids.erase(std::remove(cleaned_row_ids.begin(), cleaned_row_ids.end(), '}'),
                          cleaned_row_ids.end());

    std::istringstream row_ids_stream(cleaned_row_ids);
    std::string id;
    while (std::getline(row_ids_stream, id, ',')) {
        try {
            rows_to_query.push_back(std::stoull(id));
        } catch (const std::exception& e) {
            std::cerr << "Error: Invalid row ID format: " << id << "\n";
            return;
        }
    }

    if (rows_to_query.empty()) {
        std::cerr << "Error: No valid row IDs provided.\n";
        return;
    }

    // Load the BRWT
    std::ifstream brwt_in(brwt_file, std::ios::binary);
    if (!brwt_in.is_open()) {
        std::cerr << "Error: Failed to open BRWT file: " << brwt_file << "\n";
        return;
    }

    auto brwt = std::make_unique<BRWT>();
    brwt->load(brwt_in);

    // Query the rows
    std::vector<annot::matrix::BinaryMatrix::SetBitPositions> row_results
            = brwt->get_rows(rows_to_query);

    // Print the results (positions of set bits)
    // for (size_t i = 0; i < rows_to_query.size(); ++i) {
    //     std::cout << "Row " << rows_to_query[i] << ": ";
    //     const auto& set_bits = row_results[i];
    //     for (const auto& bit_pos : set_bits) {
    //         std::cout << bit_pos << " "; // Print the bit position
    //     }
    //     std::cout << "\n";
    // }

    // Deserialize the vector from the file
    std::vector<std::pair<uint64_t, std::string>> deserialized_column_names
            = deserialize_column_names(columns_file);

    // Print the corresponding column names
    for (size_t i = 0; i < rows_to_query.size(); ++i) {
        std::cout << "Row " << rows_to_query[i] << ": ";
        const auto& set_bits = row_results[i];
        for (const auto& bit_pos : set_bits) {
            // Find the corresponding column name
            auto it = std::find_if(deserialized_column_names.begin(),
                                   deserialized_column_names.end(),
                                   [&](const auto& pair) { return pair.first == bit_pos; });
            if (it != deserialized_column_names.end()) {
                std::cout << it->second << " "; // Print the column name
            }
        }
        std::cout << "\n";
    }
}

/*************************************************************
 *           utilities for the new HTTP /row_query           *
 *************************************************************/

// alias that bundles BRWT + column-name vector
using BrwtData
        = std::pair<std::shared_ptr<BRWT>, std::vector<std::pair<uint64_t, std::string>>>;

// convert "{1},{2}" into vector<uint64_t>{1,2}
static std::vector<uint64_t> parse_row_id_string(std::string s) {
    s.erase(std::remove(s.begin(), s.end(), '{'), s.end());
    s.erase(std::remove(s.begin(), s.end(), '}'), s.end());
    std::vector<uint64_t> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
        if (!tok.empty())
            out.push_back(std::stoull(tok));
    return out;
}

// build JSON reply for one request
static std::string process_row_query_request(const std::string& body, const BrwtData& data) {
    Json::Value json = parse_json_string(body);

    std::vector<uint64_t> rows;
    if (json["row_ids"].isArray()) {
        for (const auto& v : json["row_ids"])
            rows.push_back(v.asUInt64());
    } else if (json["row_ids"].isString()) {
        rows = parse_row_id_string(json["row_ids"].asString());
    } else {
        throw std::invalid_argument("row_ids must be array or string");
    }
    if (rows.empty())
        throw std::invalid_argument("row_ids empty");

    const auto& brwt = *data.first;
    const auto& col = data.second;
    using SetBits = annot::matrix::BinaryMatrix::SetBitPositions;
    std::vector<SetBits> res = brwt.get_rows(rows);

    Json::Value root(Json::objectValue);
    for (size_t i = 0; i < rows.size(); ++i) {
        Json::Value arr(Json::arrayValue);
        for (auto bit : res[i]) {
            auto it = std::find_if(col.begin(), col.end(),
                                   [&](auto& p) { return p.first == bit; });
            if (it != col.end())
                arr.append(it->second);
        }
        root[std::to_string(rows[i])] = arr;
    }
    Json::StreamWriterBuilder w;
    return Json::writeString(w, root);
}

/********************************************************************
 *             tiny HTTP server that exposes /row_query             *
 ********************************************************************/
// static void run_row_query_server(const std::string& brwt_file,
//                                  const std::string& columns_file,
//                                  uint16_t port) {
//     // load BRWT + column names once
//     auto data_future = std::async(std::launch::async, [&]() -> BrwtData {
//         std::ifstream bin(brwt_file, std::ios::binary);
//         if (!bin)
//             throw std::runtime_error("Cannot open BRWT file");
//         auto brwt = std::make_shared<BRWT>();
//         brwt->load(bin);
//         auto names = deserialize_column_names(columns_file);
//         logger->info("[serve] BRWT loaded");
//         return { brwt, std::move(names) };
//     });

//     // HTTP server
//     HttpServer server;
//     server.config.port = port;
//     server.config.thread_pool_size = std::max(1u, get_num_threads());

//     server.resource["^/row_query$"]["POST"] = [&](std::shared_ptr<HttpServer::Response> resp,
//                                                   std::shared_ptr<HttpServer::Request> req) {
//         if (data_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
//             resp->write(SimpleWeb::StatusCode::server_error_service_unavailable,
//                         "Server initialising, please retry later.");
//             return;
//         }
//         process_request(resp, req, [&](const std::string& body) {
//             return process_row_query_request(body, data_future.get());
//         });
//     };

//     server.default_resource["GET"] = [](auto resp, auto req) {
//         resp->write(SimpleWeb::StatusCode::client_error_not_found,
//                     "Unknown path " + req->path);
//     };
//     server.default_resource["POST"] = server.default_resource["GET"];
//     server.on_error = [](auto /*req*/, const SimpleWeb::error_code& ec) {
//         if (ec != asio::error::operation_aborted && ec != asio::stream_errc::eof)
//             logger->warn("HTTP error {} ({})", ec.message(), ec.value());
//     };

//     std::cout << "[serve] listening on port " << port << " …\n";
//     server.start(); // blocks
// }
static void run_row_query_server(const std::string& brwt_file,
                                 const std::string& columns_file,
                                 uint16_t port) {
    /* ---------- asynchronous (background loading) one-time load -------------------------------- */
    auto data_future = std::async(std::launch::async, [=] {
        std::ifstream in(brwt_file, std::ios::binary);
        if (!in)
            throw std::runtime_error("Cannot open BRWT file");
        auto brwt = std::make_shared<BRWT>();
        brwt->load(in);
        auto names = deserialize_column_names(columns_file);
        logger->info("[serve] BRWT loaded");
        return BrwtData { brwt, std::move(names) };
    });

    /* shared pointer that every request will use */
    std::shared_ptr<BrwtData> data_ptr; // nullptr until first access

    /* -------------------- HTTP server ------------------------------ */
    HttpServer server;
    server.config.port = port;
    server.config.thread_pool_size = std::max(1u, get_num_threads());

    server.resource["^/row_query$"]["POST"] = [&, data_ptr](auto resp, auto req) mutable { // note: mutable!
        /* initialise once */
        if (!data_ptr) { // test the pointer
            if (data_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                resp->write(SimpleWeb::StatusCode::server_error_service_unavailable,
                            "Server initialising, retry later.");
                return;
            }
            data_ptr = std::make_shared<BrwtData>(data_future.get());
        }
        /* use the already-loaded BRWT */
        process_request(resp, req, [&](const std::string& body) {
            return process_row_query_request(body, *data_ptr);
        });
    };

    logger->info("[serve] listening on port {}", port);
    server.start();
}

// 
void handle_concat(const std::string& brwt_file1,
                   const std::string& brwt_file2,
                   const std::string& output_file,
                   bool sparse_mode) {
    // Load both BRWTs
    auto brwt1 = std::make_unique<BRWT>();
    auto brwt2 = std::make_unique<BRWT>();

    {
        std::ifstream in1(brwt_file1, std::ios::binary);
        if (!in1) throw std::runtime_error("Failed to open " + brwt_file1);
        brwt1->load(in1);
    }
    {
        std::ifstream in2(brwt_file2, std::ios::binary);
        if (!in2) throw std::runtime_error("Failed to open " + brwt_file2);
        brwt2->load(in2);
    }

    // Prepare vector of submatrices
    std::vector<BRWT> submatrices;
    submatrices.push_back(std::move(*brwt1));
    submatrices.push_back(std::move(*brwt2));

    // Thread pool for parallel tasks
    size_t num_threads = std::thread::hardware_concurrency();
    ThreadPool pool(num_threads);

    // Buffer for dense concatenation
    sdsl::bit_vector buffer(submatrices[0].num_rows());

    BRWT result;
    if (sparse_mode) {
        result = BRWTBottomUpBuilder::concatenate_sparse(
            std::move(submatrices), &buffer, pool);
    } else {
        result = BRWTBottomUpBuilder::concatenate(
            std::move(submatrices), &buffer, pool);
    }

    // Serialize result
    std::ofstream out(output_file, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open " + output_file);
    result.serialize(out);

    std::cout << "Concatenated BRWT written to " << output_file << "\n";
}

// Helper to guess columns file path
std::string get_columns_path(const std::string& brwt_path) {
    if (brwt_path.size() > 5 && brwt_path.substr(brwt_path.size() - 5) == ".brwt") {
        return brwt_path.substr(0, brwt_path.size() - 5) + ".columns";
    }
    return brwt_path + ".columns";
}

// Update an existing BRWT with a new one (merge columns)
void handle_update(const std::string& old_brwt_file,
                   const std::string& new_brwt_file,
                   const std::string& output_file) {
    auto brwt = std::make_unique<BRWT>();
    {
        std::ifstream in(old_brwt_file, std::ios::binary);
        if (!in) throw std::runtime_error("Failed to open " + old_brwt_file);
        brwt->load(in);
    }

    auto new_brwt = std::make_unique<BRWT>();
    {
        std::ifstream in(new_brwt_file, std::ios::binary);
        if (!in) throw std::runtime_error("Failed to open " + new_brwt_file);
        new_brwt->load(in);
    }

    uint64_t old_num_columns = brwt->num_columns();

    logger->info("Merging new BRWT into the old one...");
    brwt->update_merge(*new_brwt);

    // Serialize result
    std::ofstream out(output_file, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open " + output_file);
    brwt->serialize(out);

    logger->info("Updated BRWT written to {}", output_file);

    // Merge columns
    // Attempt to locate old.columns and new.columns
    std::string old_cols_path = get_columns_path(old_brwt_file);
    std::string new_cols_path = get_columns_path(new_brwt_file);
    std::string out_cols_path = get_columns_path(output_file);

    if (fs::exists(old_cols_path) && fs::exists(new_cols_path)) {
        logger->info("Merging column names: {} + {} -> {}", old_cols_path, new_cols_path, out_cols_path);
        
        auto old_cols = deserialize_column_names(old_cols_path);
        auto new_cols = deserialize_column_names(new_cols_path);
        
        // Append new columns with updated indices
        // When updating, 'old' columns are still 0..old_num_cols-1
        // 'new' columns are appended, so their indices shift by old_num_cols.
        // HOWEVER, wait: The indices stored in the pairs are generally 0..K-1.
        // BRWTBottomUpBuilder::concatenate appends columns.
        // So column 'j' of new_brwt becomes column 'old_num_columns + j' of the merged BRWT.
        // But the stored pairs just map 'index -> name'.
        // We need to shift the index in the pair.
        
        for (auto& p : new_cols) {
            p.first += old_num_columns;
        }
        
        // Concatenate
        old_cols.insert(old_cols.end(), new_cols.begin(), new_cols.end());
        
        serialize_column_names(old_cols, out_cols_path);
    } else {
        logger->warn("Could not find .columns files, skipping column name merge.");
    }
}

} // namespace


// Helper function to show usage
void show_usage(const std::string& program_name) {
    std::cerr
            << "Version: 0.1.1\n"  
            << "Usage:\n"
            << "  "
            << program_name << " build <annotationDir> <annotationFilePrefix> <outputFile> [tmpDir] [--file_list <fileList>] [--threads <num_threads>] [--linkage_k <k>] [--linkage_seed <seed>] [--linkage_trivial]\n"
            << "      - Build and serialize BRWT to a disk file.\n"
            << "  " << program_name << " query <brwtFile> <columnsFile> <rowIds>\n"
            << "      - Retrieve rows in BRWT and return its representing column names.\n"
            << "  " << program_name
            << " relax <brwtFile> <maxArity> <outputFile> [--threads <num_threads>]\n"
            << "      - Relax a BRWT file to have a maximum arity per node and write "
               "output.\n"
            << "  " << program_name << " serve <brwtFile> <columnsFile> [port]\n"
            << "      - Serve query as a REST API.\n"
            << "  " << program_name
            << " concat <brwtFile1> <brwtFile2> <outputFile> [--sparse]\n"
            << "      - Concatenate two BRWT files into one, with optional sparse mode.\n"
            << "  " << program_name
            << " update <oldBRWT> <newBRWT> <outputFile>\n"
            << "      - Update an existing BRWT with a new one (merge columns). New BRWT must have >= rows.\n";
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        show_usage(argv[0]);
        return 1;
    }

    logger->set_level(common::get_verbose()
                            ? spdlog::level::trace
                            : spdlog::level::info);
    //logger->set_pattern("%^date %x....%$  %v");
    //spdlog::set_pattern("[%H:%M:%S %z] [%n] [%^---%L---%$] [thread %t] %v");
    //console_sink->set_color(spdlog::level::trace, "\033[37m");
    spdlog::flush_every(std::chrono::seconds(1));

    std::string command = argv[1];

    try {
        if (command == "build") {
            // Required: annotationDir, annotationFilePrefix, outputFile; optional: tmpDir and extra flags
            if (argc < 5) {
                show_usage(argv[0]);
                return 1;
            }
            std::string annotation_dir = argv[2];
            std::string annotation_file_prefix = argv[3];
            std::string output_file = argv[4];

            std::string tmp_dir = "";
            int index = 5;
            // Optional tmp_dir if next parameter is not a flag
            if (index < argc && std::string(argv[index]).find("--") != 0) {
                tmp_dir = argv[index++];
            }

            std::string file_list = ""; // Optional file list
            unsigned int num_threads = std::thread::hardware_concurrency();
            size_t linkage_k = 0;
            size_t linkage_seed = 42;
            bool linkage_trivial = false;
            // Parse optional flags
            while (index < argc) {
                std::string arg = argv[index];
                if (arg == "--file_list") {
                    if (++index < argc) {
                        file_list = argv[index++];
                    } else {
                        std::cerr << "Error: --fileList option requires a value.\n";
                        return 1;
                    }
                } else if (arg == "--threads") {
                    if (++index < argc) {
                        num_threads = std::stoul(argv[index++]);
                    } else {
                        std::cerr << "Error: --threads option requires a value.\n";
                        return 1;
                    }
                } else if (arg == "--linkage_k") {
                    if (++index < argc) {
                        linkage_k = std::stoul(argv[index++]);
                    } else {
                        std::cerr << "Error: --linkage_k option requires a value.\n";
                        return 1;
                    }
                } else if (arg == "--linkage_seed") {
                    if (++index < argc) {
                        linkage_seed = std::stoul(argv[index++]);
                    } else {
                        std::cerr << "Error: --linkage_seed option requires a value.\n";
                        return 1;
                    }
                } else if (arg == "--linkage_trivial") {
                    linkage_trivial = true;
                    ++index;
                } else {
                    std::cerr << "Unknown option: " << arg << "\n";
                    return 1;
                }
            }
            logger->info("Using {} threads for building BRWT.", num_threads);
            if (linkage_k)
                logger->info("Using linkage_k = {}", linkage_k);
            if (linkage_seed)
                logger->info("Using linkage_seed = {}", linkage_seed);

            handle_build(annotation_dir, annotation_file_prefix, output_file, tmp_dir, file_list,
                         num_threads, linkage_k, linkage_seed, linkage_trivial);

        } else if (command == "query") {
            if (argc != 5) {
                show_usage(argv[0]);
                return 1;
            }
            std::string brwt_file = argv[2];
            std::string columns_file = argv[3];
            std::string row_ids = argv[4];

            handle_query(row_ids, brwt_file, columns_file);

        } else if (command == "relax") {
            // Required: brwtFile, maxArity, outputFile; optional: --threads flag
            if (argc < 5) {
                show_usage(argv[0]);
                return 1;
            }
            std::string brwt_file = argv[2];
            uint64_t max_arity = std::stoull(argv[3]);
            std::string output_file = argv[4];

            unsigned int num_threads = std::thread::hardware_concurrency();
            int index = 5;
            // Optional --threads flag to override number of threads
            if (index < argc) {
                std::string arg = argv[index];
                if (arg == "--threads") {
                    if (++index < argc) {
                        num_threads = std::stoul(argv[index]);
                        index++;
                    } else {
                        std::cerr << "Error: --threads option requires a value.\n";
                        return 1;
                    }
                }
            }
            relax_brwt(brwt_file, max_arity, num_threads, output_file);
            std::cout << "BRWT successfully relaxed and written to " << output_file << "\n";
        } else if (command == "serve") {
            if (argc < 4 || argc > 5) {
                std::cerr << "Usage: " << argv[0]
                          << " serve <brwtFile> <columnsFile> [port]\n";
                return 1;
            }
            std::string brwt_file = argv[2];
            std::string columns_file = argv[3];
            uint16_t port = (argc == 5) ? static_cast<uint16_t>(std::stoi(argv[4])) : 9000;
            logger->info("Starting...");
            run_row_query_server(brwt_file, columns_file, port);
        } else if (command == "concat") {
            if (argc < 5 || argc > 6) {
                show_usage(argv[0]);
                return 1;
            }
            std::string brwt_file1 = argv[2];
            std::string brwt_file2 = argv[3];
            std::string output_file = argv[4];
            bool sparse_mode = (argc == 6 && std::string(argv[5]) == "--sparse");

            handle_concat(brwt_file1, brwt_file2, output_file, sparse_mode);
        } else if (command == "update") {
            if (argc != 5) {
                show_usage(argv[0]);
                return 1;
            }
            std::string old_brwt_file = argv[2];
            std::string new_brwt_file = argv[3];
            std::string output_file = argv[4];

            handle_update(old_brwt_file, new_brwt_file, output_file);
        } else {
            show_usage(argv[0]);
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}