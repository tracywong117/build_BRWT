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
#include "cli/server_utils.hpp"
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
    if (!in.is_open()) {
        std::cerr << "Error: Failed to open columns file: " << filename << "\n";
        return {};
    }
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
    if (!in.is_open()) {
        std::cerr << "Error: Failed to open BRWT file: " << brwt_f << "\n";
        return;
    }
    BRWT brwt;
    if (!brwt.load(in)) {
        std::cerr << "Error: Failed to load BRWT from " << brwt_f << "\n";
        return;
    }
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

void print_general_usage(const char* progname) {
    std::cerr << "Usage: " << progname << " <command> [options]\n"
              << "\n"
              << "Commands:\n"
              << "  build   Build a BRWT index from annotation columns\n"
              << "  query   Query rows from a BRWT index\n"
              << "  stats   Print statistics of a BRWT index\n"
              << "  server  Start an HTTP server to query the BRWT index\n"
              << "\n"
              << "Run '" << progname << " <command> --help' for more information on a command.\n";
}

void print_build_usage(const char* progname) {
    std::cerr << "Usage: " << progname << " build <annotation_dir> <prefix> <output> [tmp_dir] [options]\n"
              << "\n"
              << "Build a BRWT index from sd_vector annotation columns.\n"
              << "\n"
              << "Positional arguments:\n"
              << "  annotation_dir       Directory containing the sd_vector column files\n"
              << "  prefix               Filename prefix to filter column files (e.g. \"col_\")\n"
              << "  output               Output path prefix (produces <output>.brwt and <output>.columns)\n"
              << "  tmp_dir              Optional temporary directory for streaming construction\n"
              << "                       (must not start with '--'; omit for in-memory build)\n"
              << "\n"
              << "Options:\n"
              << "  --threads N          Number of threads [default: hardware concurrency]\n"
              << "  --list FILE          File listing column filenames (one per line), relative\n"
              << "                       to annotation_dir. If omitted, all files matching the\n"
              << "                       prefix in annotation_dir are used.\n"
              << "  --full-nodes N       Number of full (uncompressed) internal nodes [default: 0]\n"
              << "  --partial-nodes N    Number of partially compressed internal nodes [default: 0]\n"
              << "  --linkage-k N        k parameter for greedy linkage clustering [default: 10]\n"
              << "  --linkage-seed N     Random seed for linkage clustering [default: 42]\n"
              << "  --linkage_trivial    Use trivial (sequential) linkage instead of greedy clustering\n"
              << "  --resume             Resume a previously interrupted build\n"
              << "\n"
              << "Examples:\n"
              << "  " << progname << " build ./columns col_ output_index\n"
              << "  " << progname << " build ./columns col_ output_index /tmp/brwt --threads 16\n"
              << "  " << progname << " build ./columns col_ output_index --list files.txt --linkage_trivial\n";
}

void print_query_usage(const char* progname) {
    std::cerr << "Usage: " << progname << " query <brwt_file> <columns_file> <row_ids>\n"
              << "\n"
              << "Query rows from a BRWT index and print matching column names.\n"
              << "\n"
              << "Positional arguments:\n"
              << "  brwt_file            Path to the .brwt index file\n"
              << "  columns_file         Path to the .columns file (column name mapping)\n"
              << "  row_ids              Comma-separated row IDs to query, optionally wrapped\n"
              << "                       in braces, e.g. \"{0,1,2}\" or \"0,1,2\"\n"
              << "\n"
              << "Examples:\n"
              << "  " << progname << " query output_index.brwt output_index.columns \"{0,1,42}\"\n"
              << "  " << progname << " query output_index.brwt output_index.columns 0,1,42\n";
}

void print_stats_usage(const char* progname) {
    std::cerr << "Usage: " << progname << " stats <brwt_file>\n"
              << "\n"
              << "Print statistics of a BRWT index.\n"
              << "\n"
              << "Positional arguments:\n"
              << "  brwt_file            Path to the .brwt index file\n";
}

void print_server_usage(const char* progname) {
    std::cerr << "Usage: " << progname << " server <brwt_file> <columns_file> [options]\n"
              << "\n"
              << "Start an HTTP server to query the BRWT index.\n"
              << "\n"
              << "Positional arguments:\n"
              << "  brwt_file            Path to the .brwt index file\n"
              << "  columns_file         Path to the .columns file (column name mapping)\n"
              << "\n"
              << "Options:\n"
              << "  --port N             Port to listen on [default: 8080]\n";
}

void handle_stats(const std::string& brwt_f) {
    std::ifstream in(brwt_f, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("Failed to open " + brwt_f);
    BRWT brwt; 
    brwt.load(in);
    
    std::cout << "--- BRWT Statistics ---\n"
              << "Number of rows:       " << brwt.num_rows() << "\n"
              << "Number of columns:    " << brwt.num_columns() << "\n"
              << "Number of set bits:   " << brwt.num_relations() << "\n"
              << "Number of nodes:      " << brwt.num_nodes() << "\n"
              << "Average arity:        " << brwt.avg_arity() << "\n"
              << "Shrinking rate:       " << brwt.shrinking_rate() << "\n"
              << "-----------------------\n";
}

void handle_server(const std::string& brwt_f, const std::string& cols_f, unsigned short port) {
    logger->info("Loading BRWT index from {}...", brwt_f);
    std::ifstream in(brwt_f, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("Failed to open " + brwt_f);
    BRWT brwt; 
    brwt.load(in);
    
    logger->info("Loading column names from {}...", cols_f);
    auto names = deserialize_column_names(cols_f);

    mtg::cli::HttpServer server;
    server.config.port = port;

    server.resource["^/query$"]["POST"] = [&](std::shared_ptr<mtg::cli::HttpServer::Response> response, std::shared_ptr<mtg::cli::HttpServer::Request> request) {
        mtg::cli::process_request(response, request, [&](const std::string& req_body) {
            Json::Value req_json = mtg::cli::parse_json_string(req_body);
            if (!req_json.isMember("row_ids") || !req_json["row_ids"].isArray()) {
                throw std::runtime_error("Request must contain 'row_ids' array.");
            }
            
            std::vector<uint64_t> ids;
            for (const auto& id_val : req_json["row_ids"]) {
                ids.push_back(id_val.asUInt64());
            }
            
            auto results = brwt.get_rows(ids);
            
            Json::Value resp_json(Json::objectValue);
            for (size_t i = 0; i < ids.size(); ++i) {
                Json::Value row_res(Json::arrayValue);
                for (auto bit : results[i]) {
                    auto it = std::find_if(names.begin(), names.end(), [&](auto& p){ return p.first == bit; });
                    if (it != names.end()) row_res.append(it->second);
                }
                resp_json[std::to_string(ids[i])] = row_res;
            }
            return Json::writeString(Json::StreamWriterBuilder(), resp_json);
        });
    };

    logger->info("Starting server on port {}...", port);
    server.start();
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_general_usage(argv[0]);
        return 1;
    }
    std::string cmd = argv[1];

    if (cmd == "--help" || cmd == "-h") {
        print_general_usage(argv[0]);
        return 0;
    }

    try {
        if (cmd == "build") {
            if (argc < 5 || std::string(argv[2]) == "--help" || std::string(argv[2]) == "-h") {
                print_build_usage(argv[0]);
                return (argc >= 3 && (std::string(argv[2]) == "--help" || std::string(argv[2]) == "-h")) ? 0 : 1;
            }
            std::string dir = argv[2], pre = argv[3], out = argv[4];
            std::string tmp = (argc > 5 && std::string(argv[5]).find("--") != 0) ? argv[5] : "";
            std::string list_file;
            size_t threads = std::thread::hardware_concurrency();
            size_t full_nodes = 0, partial_nodes = 0;
            size_t linkage_k = 0, linkage_seed = 42;
            bool trivial = false, resume = false;
            for (int i = 5; i < argc; ++i) {
                std::string arg = argv[i];
                if (arg == "--threads") threads = std::stoul(argv[++i]);
                else if (arg == "--list") list_file = argv[++i];
                else if (arg == "--full-nodes") full_nodes = std::stoul(argv[++i]);
                else if (arg == "--partial-nodes") partial_nodes = std::stoul(argv[++i]);
                else if (arg == "--linkage-k") linkage_k = std::stoul(argv[++i]);
                else if (arg == "--linkage-seed") linkage_seed = std::stoul(argv[++i]);
                else if (arg == "--linkage_trivial") trivial = true;
                else if (arg == "--resume") resume = true;
                else if (arg == "--help" || arg == "-h") { print_build_usage(argv[0]); return 0; }
                else if (arg.substr(0, 2) == "--") {
                    std::cerr << "Unknown option: " << arg << "\n\n";
                    print_build_usage(argv[0]);
                    return 1;
                }
            }
            handle_build(dir, pre, out, tmp, list_file, threads, full_nodes, partial_nodes, linkage_k, linkage_seed, trivial, resume);
        } else if (cmd == "query") {
            if (argc < 5 || std::string(argv[2]) == "--help" || std::string(argv[2]) == "-h") {
                print_query_usage(argv[0]);
                return (argc >= 3 && (std::string(argv[2]) == "--help" || std::string(argv[2]) == "-h")) ? 0 : 1;
            }
            handle_query(argv[4], argv[2], argv[3]);
        } else if (cmd == "stats") {
            if (argc < 3 || std::string(argv[2]) == "--help" || std::string(argv[2]) == "-h") {
                print_stats_usage(argv[0]);
                return (argc >= 3 && (std::string(argv[2]) == "--help" || std::string(argv[2]) == "-h")) ? 0 : 1;
            }
            handle_stats(argv[2]);
        } else if (cmd == "server") {
            if (argc < 4 || std::string(argv[2]) == "--help" || std::string(argv[2]) == "-h") {
                print_server_usage(argv[0]);
                return (argc >= 3 && (std::string(argv[2]) == "--help" || std::string(argv[2]) == "-h")) ? 0 : 1;
            }
            std::string brwt_f = argv[2];
            std::string cols_f = argv[3];
            unsigned short port = 8080;
            for (int i = 4; i < argc; ++i) {
                std::string arg = argv[i];
                if (arg == "--port" && i + 1 < argc) port = std::stoul(argv[++i]);
                else if (arg == "--help" || arg == "-h") { print_server_usage(argv[0]); return 0; }
                else {
                    std::cerr << "Unknown option: " << arg << "\n\n";
                    print_server_usage(argv[0]);
                    return 1;
                }
            }
            handle_server(brwt_f, cols_f, port);
        } else {
            std::cerr << "Unknown command: " << cmd << "\n\n";
            print_general_usage(argv[0]);
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
