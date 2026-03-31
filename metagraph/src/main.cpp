#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <mutex>
#include <thread>
#include <future>
#include <atomic>
#include <chrono>
#include <sstream>

#include <sdsl/bit_vectors.hpp>
#include <json/json.h>

#include "annotation/binary_matrix/multi_brwt/brwt_builders.hpp"
#include "annotation/binary_matrix/multi_brwt/brwt.hpp"
#include "annotation/binary_matrix/multi_brwt/clustering.hpp"
#include "common/logger.hpp"
#include "common/utils/file_utils.hpp"
#include "common/serialization.hpp"
#include "common/vectors/bit_vector.hpp"
#include "common/vectors/bit_vector_adaptive.hpp"
#include "common/vectors/bit_vector_sd.hpp"
#include "common/vectors/bit_vector_sdsl.hpp"
#include "common/threads/threading.hpp"

// Simple Web Server headers
#include <server_http.hpp>

using namespace mtg;
using namespace mtg::annot::matrix;
using mtg::common::logger;

typedef SimpleWeb::Server<SimpleWeb::HTTP> HttpServer;

namespace {

// Parse the linkage matrix from a file
std::vector<std::vector<uint64_t>> parse_linkage_matrix(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open linkage file: " + filename);
    }

    std::vector<std::vector<uint64_t>> linkage;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        uint64_t c1, c2, num_ones, new_id;
        if (!(ss >> c1 >> c2 >> num_ones >> new_id)) continue;

        if (new_id >= linkage.size()) {
            linkage.resize(new_id + 1);
        }
        linkage[new_id] = {c1, c2};
    }
    return linkage;
}

// Compute the linkage matrix and save to file
void BM_BRWTLinkageMatrix_streaming(const std::string& annotation_dir,
                                   const std::string& prefix,
                                   const std::string& output_file,
                                   const std::string& file_list,
                                   size_t num_threads,
                                   size_t linkage_k,
                                   size_t linkage_seed,
                                   bool linkage_trivial) {
    
    // 1. Get column filenames
    std::vector<std::string> file_paths;
    namespace fs = std::filesystem;
    if (!file_list.empty()) {
        std::ifstream list_file(file_list);
        std::string line;
        while (std::getline(list_file, line)) {
            if (!line.empty()) file_paths.push_back(annotation_dir + "/" + line);
        }
    } else {
        for (const auto& entry : fs::directory_iterator(annotation_dir)) {
            if (entry.is_regular_file() && entry.path().filename().string().find(prefix) == 0) {
                file_paths.push_back(entry.path().string());
            }
        }
    }
    std::sort(file_paths.begin(), file_paths.end());
    size_t num_columns = file_paths.size();

    LinkageMatrix lm;
    if (linkage_trivial) {
        lm = agglomerative_linkage_trivial(num_columns);
    } else {
        // Sample columns to build linkage (approximate)
        std::vector<SparseColumn> samples(num_columns);
        #pragma omp parallel for num_threads(num_threads)
        for (size_t i = 0; i < num_columns; ++i) {
            std::ifstream in(file_paths[i], std::ios::binary);
            sdsl::sd_vector<> sdvec;
            sdsl::load(sdvec, in);
            samples[i].size = sdvec.size();
            sdsl::sd_vector<>::rank_1_type rank1(&sdvec);
            uint64_t total_ones = rank1(sdvec.size());
            sdsl::sd_vector<>::select_1_type select_1(&sdvec);
            for (uint64_t r = 1; r <= std::min(total_ones, (uint64_t)10000); ++r) {
                samples[i].set_bits.push_back(select_1(r));
            }
        }
        lm = agglomerative_greedy_linkage_k(std::move(samples), num_threads, linkage_k ? linkage_k : 10, linkage_seed);
    }

    // Save linkage matrix to file
    std::ofstream out(output_file);
    for (int i = 0; i < lm.rows(); ++i) {
        out << lm(i, 0) << " " << lm(i, 1) << " " << lm(i, 2) << " " << lm(i, 3) << "\n";
    }
}

// Helper to parse JSON strings
Json::Value parse_json_string(const std::string& s) {
    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errs;
    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    if (!reader->parse(s.data(), s.data() + s.size(), &root, &errs)) {
        throw std::runtime_error("JSON parse error: " + errs);
    }
    return root;
}

// Load all sdsl::sd_vector<> from files in a folder with a callback
bool load_columns_from_folder_callback(const std::string& folder_path,
                                       const std::string& file_list,
                                       const std::function<void(uint64_t, const std::string&,
                                                                std::unique_ptr<bit_vector>&&)>& callback,
                                       size_t num_threads,
                                       const std::string& prefix,
                                       bool metadata_only = false) {
    namespace fs = std::filesystem;
    std::vector<std::string> file_paths;

    if (!file_list.empty()) {
        std::ifstream list_file(file_list);
        std::string line;
        while (std::getline(list_file, line)) {
            if (!line.empty()) file_paths.push_back(folder_path + "/" + line);
        }
    } else {
        for (const auto& entry : fs::directory_iterator(folder_path)) {
            if (entry.is_regular_file() && entry.path().filename().string().find(prefix) == 0) {
                file_paths.push_back(entry.path().string());
            }
        }
    }

    if (file_paths.empty()) return false;
    std::sort(file_paths.begin(), file_paths.end());

    uint64_t column_length = 0;
    {
        std::ifstream first_file(file_paths[0], std::ios::binary);
        if (first_file.is_open()) {
            sdsl::sd_vector<> sdvec;
            sdsl::load(sdvec, first_file);
            column_length = sdvec.size();
            logger->info("Column length determined from first file: {}", column_length);
        } else {
            return false;
        }
    }

    std::atomic<bool> success(true);
    std::mutex callback_mutex;

    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (size_t i = 0; i < file_paths.size(); ++i) {
        if (!success) continue;
        try {
            std::string label = fs::path(file_paths[i]).filename().string();
            if (metadata_only) {
                std::lock_guard<std::mutex> lock(callback_mutex);
                callback(i, label, nullptr);
                continue;
            }

            std::ifstream file_stream(file_paths[i], std::ios::binary);
            if (!file_stream.is_open()) {
                success = false;
                continue;
            }
            sdsl::sd_vector<> sdvec;
            sdsl::load(sdvec, file_stream);

            if (sdvec.size() != column_length) {
                #pragma omp critical
                {
                    success = false;
                    throw std::runtime_error("Column length mismatch in file: " + file_paths[i]);
                }
                continue;
            }

            sdsl::sd_vector<>::rank_1_type rank1(&sdvec);
            uint64_t num_ones = rank1(sdvec.size());
            uint64_t size = sdvec.size();

            auto column_ptr = std::make_unique<bit_vector_smart>(
                [&](const auto &callback) {
                    sdsl::sd_vector<>::select_1_type select_1(&sdvec);
                    for (uint64_t r = 1; r <= num_ones; ++r) {
                        callback(select_1(r));
                    }
                },
                size,
                num_ones
            );

            std::lock_guard<std::mutex> lock(callback_mutex);
            callback(i, label, std::move(column_ptr));
        } catch (...) { success = false; }
    }
    return success;
}

void serialize_column_names(const std::vector<std::pair<uint64_t, std::string>>& column_names,
                            const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    size_t size = column_names.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    for (const auto& pair : column_names) {
        out.write(reinterpret_cast<const char*>(&pair.first), sizeof(pair.first));
        size_t len = pair.second.size();
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(pair.second.data(), len);
    }
}

std::vector<std::pair<uint64_t, std::string>> deserialize_column_names(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) return {};
    size_t size = 0;
    if (!in.read(reinterpret_cast<char*>(&size), sizeof(size))) return {};
    std::vector<std::pair<uint64_t, std::string>> names;
    names.reserve(size);
    for (size_t i = 0; i < size; ++i) {
        uint64_t first;
        in.read(reinterpret_cast<char*>(&first), sizeof(first));
        size_t len;
        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        std::string second(len, '\0');
        in.read(&second[0], len);
        names.emplace_back(first, second);
    }
    return names;
}

void build_BRWT(const std::string columns_folder_path,
                const std::string prefix,
                const std::vector<std::vector<uint64_t>>& linkage,
                const std::string& tmp_path,
                const std::string& file_list,
                size_t num_full_nodes,
                size_t num_partial_nodes,
                size_t num_threads,
                const std::string& output_path,
                bool resume) {
    std::mutex mu;
    std::vector<std::pair<uint64_t, std::string>> column_names;
    std::vector<std::string> column_files;

    std::vector<std::vector<uint64_t>> stored_columns(linkage.size());
    for (size_t i = 0; i < linkage.size(); ++i) {
        if (linkage[i].empty()) continue;
        for (size_t j : linkage[i]) {
            if (linkage[j].empty()) stored_columns[i].push_back(j);
            else for (size_t c : stored_columns[j]) stored_columns[i].push_back(c);
        }
    }

    auto get_columns = [&](const BRWTBottomUpBuilder::CallColumn& call_column) {
        bool metadata_only = !tmp_path.empty();
        load_columns_from_folder_callback(columns_folder_path, file_list,
            [&](uint64_t j, const std::string& label, std::unique_ptr<bit_vector>&& column) {
                std::lock_guard<std::mutex> lock(mu);
                if (j >= column_files.size()) column_files.resize(j + 1);
                column_files[j] = columns_folder_path + "/" + label;
                column_names.emplace_back(j, label);
                call_column(j, std::move(column));
            }, num_threads, prefix, metadata_only);
    };

    std::string actual_tmp_nodes_dir;
    BRWT root = BRWTBottomUpBuilder::build(get_columns, linkage, tmp_path, stored_columns, column_files, num_full_nodes, num_partial_nodes, num_threads, &actual_tmp_nodes_dir, resume);

    std::ofstream out(output_path + ".brwt", std::ios::binary);
    if (!tmp_path.empty()) {
        BRWTBottomUpBuilder::assemble_streaming(out, linkage, stored_columns, actual_tmp_nodes_dir, column_files);
    } else {
        root.serialize(out);
    }
    serialize_column_names(column_names, output_path + ".columns");
}

void handle_build(const std::string& annotation_dir, const std::string& prefix, const std::string& output, const std::string& tmp, const std::string& list, size_t threads, size_t full_nodes, size_t partial_nodes, size_t k, size_t seed, bool trivial, bool resume) {
    std::string linkage_file = output + ".linkage";
    
    bool recompute = !std::filesystem::exists(linkage_file);
    if (!recompute) {
        auto linkage = parse_linkage_matrix(linkage_file);
        size_t linkage_leaves = 0;
        for (const auto& children : linkage) if (children.empty()) linkage_leaves++;
        
        size_t current_columns = 0;
        if (!list.empty()) {
            std::ifstream f(list); std::string line;
            while (std::getline(f, line)) if (!line.empty()) current_columns++;
        } else {
            for (const auto& entry : std::filesystem::directory_iterator(annotation_dir))
                if (entry.is_regular_file() && entry.path().filename().string().find(prefix) == 0) current_columns++;
        }

        if (linkage_leaves != current_columns) {
            logger->warn("Existing linkage file has {} columns, but current run has {}. Recomputing...", linkage_leaves, current_columns);
            recompute = true;
        }
    }

    if (recompute) {
        logger->info("Computing linkage matrix...");
        BM_BRWTLinkageMatrix_streaming(annotation_dir, prefix, linkage_file, list, threads, k, seed, trivial);
    }
    
    auto linkage = parse_linkage_matrix(linkage_file);
    build_BRWT(annotation_dir, prefix, linkage, tmp, list, full_nodes, partial_nodes, threads, output, resume);
}

void handle_query(const std::string& ids_str, const std::string& brwt_f, const std::string& cols_f) {
    std::vector<uint64_t> ids;
    std::string cleaned = ids_str;
    cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), '{'), cleaned.end());
    cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), '}'), cleaned.end());
    std::stringstream ss(cleaned);
    std::string segment;
    while(std::getline(ss, segment, ',')) if (!segment.empty()) ids.push_back(std::stoull(segment));
    
    std::ifstream in(brwt_f, std::ios::binary);
    BRWT brwt; brwt.load(in);
    auto results = brwt.get_rows(ids);
    auto names = deserialize_column_names(cols_f);

    for (size_t i = 0; i < ids.size(); ++i) {
        std::cout << "Row " << ids[i] << ": ";
        for (auto bit : results[i]) {
            auto it = std::find_if(names.begin(), names.end(), [&](auto& p){ return p.first == bit; });
            if (it != names.end()) std::cout << it->second << " ";
        }
        std::cout << "\n";
    }
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) return 1;
    std::string cmd = argv[1];
    try {
        if (cmd == "build") {
            std::string dir = argv[2], pre = argv[3], out = argv[4], tmp = (argc > 5 && std::string(argv[5]).find("--") != 0) ? argv[5] : "";
            size_t threads = std::thread::hardware_concurrency();
            size_t full_nodes = 0, partial_nodes = 0;
            bool trivial = false, resume = false;
            for(int i=5; i<argc; ++i) {
                if (std::string(argv[i]) == "--threads") threads = std::stoul(argv[++i]);
                else if (std::string(argv[i]) == "--full-nodes") full_nodes = std::stoul(argv[++i]);
                else if (std::string(argv[i]) == "--partial-nodes") partial_nodes = std::stoul(argv[++i]);
                else if (std::string(argv[i]) == "--linkage_trivial") trivial = true;
                else if (std::string(argv[i]) == "--resume") resume = true;
            }
            handle_build(dir, pre, out, tmp, "", threads, full_nodes, partial_nodes, 0, 42, trivial, resume);
        } else if (cmd == "query") {
            handle_query(argv[4], argv[2], argv[3]);
        }
    } catch (const std::exception& e) { std::cerr << e.what() << "\n"; return 1; }
    return 0;
}
