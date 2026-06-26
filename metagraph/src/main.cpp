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
#include <iomanip>

#include <sdsl/bit_vectors.hpp>
#include <json/json.h>

#include "annotation/binary_matrix/multi_brwt/brwt_builders.hpp"
#include "annotation/binary_matrix/multi_brwt/brwt.hpp"
#include "annotation/binary_matrix/multi_brwt/clustering.hpp"
#include "cli/build_mmap_index.hpp"
#include "cli/mmap_brwt_engine.hpp"
#include "common/logger.hpp"
#include "common/utils/file_utils.hpp"
#include "common/serialization.hpp"
#include "common/vectors/bit_vector.hpp"
#include "common/vectors/bit_vector_adaptive.hpp"
#include "common/vectors/bit_vector_sd.hpp"
#include "common/vectors/bit_vector_sdsl.hpp"
#include "common/threads/threading.hpp"
#include "common/unix_tools.hpp"

// Simple Web Server headers
#include <server_http.hpp>

using namespace mtg;
using namespace mtg::annot::matrix;
using mtg::common::logger;

typedef SimpleWeb::Server<SimpleWeb::HTTP> HttpServer;

namespace {

std::string get_current_date() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d");
    return ss.str();
}

std::string format_time(double total_seconds) {
    int h = static_cast<int>(total_seconds) / 3600;
    int m = (static_cast<int>(total_seconds) % 3600) / 60;
    double s = total_seconds - h * 3600 - m * 60;
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << std::setfill('0');
    if (h > 0) {
        ss << h << ":" << std::setw(2) << m << ":" << std::setw(5) << s;
    } else {
        ss << m << ":" << std::setw(5) << s;
    }
    return ss.str();
}

// Parse the linkage matrix from a file.
// Supports two formats:
//   4-column (original build output):  c1 c2 num_ones node_id
//   variable-column (post-relax / merge output):  node_id child1 child2 [child3 ...]
// The two are distinguished by token count: if exactly 4 tokens and token[3] > token[0]
// we treat it as 4-column build format; otherwise as variable-column.
std::vector<std::vector<uint64_t>> parse_linkage_matrix(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) throw std::runtime_error("Failed to open linkage file: " + filename);
    std::vector<std::vector<uint64_t>> linkage;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::vector<uint64_t> tokens;
        uint64_t v;
        while (ss >> v) tokens.push_back(v);
        if (tokens.size() < 3) continue;
        if (tokens.size() == 4 && tokens[3] > tokens[0] && tokens[3] > tokens[1]) {
            // 4-column format: c1 c2 num_ones node_id  (binary tree, clustering output)
            uint64_t new_id = tokens[3];
            if (new_id >= linkage.size()) linkage.resize(new_id + 1);
            linkage[new_id] = {tokens[0], tokens[1]};
        } else {
            // variable-column format: node_id child1 child2 ...
            uint64_t node_id = tokens[0];
            if (node_id >= linkage.size()) linkage.resize(node_id + 1);
            linkage[node_id].assign(tokens.begin() + 1, tokens.end());
        }
    }
    return linkage;
}

// Compute the linkage matrix and save to file
void BM_BRWTLinkageMatrix_streaming(const std::string& annotation_dir, const std::string& prefix, const std::string& output_file, const std::string& file_list, size_t num_threads, size_t linkage_k, size_t linkage_seed, bool linkage_trivial) {
    std::vector<std::string> file_paths;
    namespace fs = std::filesystem;
    if (!file_list.empty()) {
        std::ifstream list_file(file_list); std::string line;
        while (std::getline(list_file, line)) if (!line.empty()) file_paths.push_back(annotation_dir + "/" + line);
    } else {
        for (const auto& entry : fs::directory_iterator(annotation_dir))
            if (entry.is_regular_file() && entry.path().filename().string().find(prefix) == 0) file_paths.push_back(entry.path().string());
    }
    std::sort(file_paths.begin(), file_paths.end());
    size_t num_columns = file_paths.size();
    LinkageMatrix lm;
    if (linkage_trivial) lm = agglomerative_linkage_trivial(num_columns);
    else {
        std::vector<SparseColumn> samples(num_columns);
        #pragma omp parallel for num_threads(num_threads)
        for (size_t i = 0; i < num_columns; ++i) {
            std::ifstream in(file_paths[i], std::ios::binary);
            sdsl::sd_vector<> sdvec; sdsl::load(sdvec, in);
            samples[i].size = sdvec.size();
            sdsl::sd_vector<>::rank_1_type rank1(&sdvec); uint64_t total_ones = rank1(sdvec.size());
            sdsl::sd_vector<>::select_1_type select_1(&sdvec);
            for (uint64_t r = 1; r <= std::min(total_ones, (uint64_t)10000); ++r) samples[i].set_bits.push_back(select_1(r));
        }
        lm = agglomerative_greedy_linkage_k(std::move(samples), num_threads, linkage_k ? linkage_k : 10, linkage_seed);
    }
    std::ofstream out(output_file);
    for (int i = 0; i < lm.rows(); ++i) out << lm(i, 0) << " " << lm(i, 1) << " " << lm(i, 2) << " " << lm(i, 3) << "\n";
}

// Load columns with metadata mode support
bool load_columns_from_folder_callback(const std::string& folder_path, const std::string& file_list, const std::function<void(uint64_t, const std::string&, std::unique_ptr<bit_vector>&&)>& callback, size_t num_threads, const std::string& prefix, bool metadata_only = false) {
    namespace fs = std::filesystem; std::vector<std::string> file_paths;
    if (!file_list.empty()) {
        std::ifstream list_file(file_list); std::string line;
        while (std::getline(list_file, line)) if (!line.empty()) file_paths.push_back(folder_path + "/" + line);
    } else {
        for (const auto& entry : fs::directory_iterator(folder_path))
            if (entry.is_regular_file() && entry.path().filename().string().find(prefix) == 0) file_paths.push_back(entry.path().string());
    }
    if (file_paths.empty()) return false;
    std::sort(file_paths.begin(), file_paths.end());
    uint64_t column_length = 0;
    {
        std::ifstream first_file(file_paths[0], std::ios::binary);
        if (first_file.is_open()) {
            sdsl::sd_vector<> sdvec; sdsl::load(sdvec, first_file);
            column_length = sdvec.size();
            logger->info("Column length determined from first file: {}", column_length);
        } else return false;
    }
    // If metadata_only, we skip loading columns and just invoke callback with nullptr for column, 
    // allowing caller to gather metadata like column names without incurring loading overhead
    std::atomic<bool> success(true); std::mutex callback_mutex;
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (size_t i = 0; i < file_paths.size(); ++i) {
        if (!success) continue;
        try {
            std::string label = fs::path(file_paths[i]).filename().string();
            if (metadata_only) { std::lock_guard<std::mutex> lock(callback_mutex); callback(i, label, nullptr); continue; }
            std::ifstream file_stream(file_paths[i], std::ios::binary);
            auto sdvec_ptr = std::make_shared<sdsl::sd_vector<>>();
            sdsl::load(*sdvec_ptr, file_stream);

            sdsl::sd_vector<>::rank_1_type rank1(sdvec_ptr.get());
            uint64_t num_ones = rank1(sdvec_ptr->size());
            uint64_t size = sdvec_ptr->size();

            auto column_ptr = std::make_unique<bit_vector_smart>([sdvec_ptr, num_ones](const auto& cb){
                sdsl::sd_vector<>::select_1_type sel(sdvec_ptr.get());
                for(uint64_t r=1; r<=num_ones; ++r) cb(sel(r));
            }, size, num_ones);

            std::lock_guard<std::mutex> lock(callback_mutex); callback(i, label, std::move(column_ptr));
        } catch (...) { success = false; }
    }
    return success;
}

void serialize_column_names(const std::vector<std::pair<uint64_t, std::string>>& column_names, const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    size_t size = column_names.size(); out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    for (const auto& pair : column_names) {
        out.write(reinterpret_cast<const char*>(&pair.first), sizeof(pair.first));
        size_t len = pair.second.size(); out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(pair.second.data(), len);
    }
}

std::vector<std::pair<uint64_t, std::string>> deserialize_column_names(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary); if (!in.is_open()) return {};
    size_t size = 0; in.read(reinterpret_cast<char*>(&size), sizeof(size));
    std::vector<std::pair<uint64_t, std::string>> names; names.reserve(size);
    for (size_t i = 0; i < size; ++i) {
        uint64_t first; in.read(reinterpret_cast<char*>(&first), sizeof(first));
        size_t len; in.read(reinterpret_cast<char*>(&len), sizeof(len));
        std::string second(len, '\0'); in.read(&second[0], len);
        names.emplace_back(first, second);
    }
    return names;
}

void build_BRWT(const std::string columns_folder_path, const std::string prefix, const std::vector<std::vector<uint64_t>>& linkage, const std::string& tmp_path, const std::string& file_list, size_t num_full_nodes, size_t num_partial_nodes, size_t num_threads, const std::string& output_path, bool resume, bool assemble) {
    auto t_start = std::chrono::high_resolution_clock::now();
    std::mutex mu; std::vector<std::pair<uint64_t, std::string>> column_names; std::vector<std::string> column_files;
    // Reconstruct stored_columns level-by-level to ensure dependencies are met
    std::vector<std::vector<uint64_t>> stored_columns(linkage.size());
    size_t total_nodes = linkage.size();
    std::vector<size_t> node_levels(total_nodes, 0);
    size_t max_level = 0;
    for (size_t i = 0; i < total_nodes; ++i) {
        for (uint64_t child : linkage[i]) {
            node_levels[i] = std::max(node_levels[i], node_levels[child] + 1);
        }
        max_level = std::max(max_level, node_levels[i]);
    }

    for (size_t lv = 1; lv <= max_level; ++lv) {
        for (size_t i = 0; i < total_nodes; ++i) {
            if (node_levels[i] != lv) continue;
            for (uint64_t j : linkage[i]) {
                if (linkage[j].empty()) {
                    stored_columns[i].push_back(j);
                } else {
                    for (uint64_t c : stored_columns[j]) {
                        stored_columns[i].push_back(c);
                    }
                }
            }
        }
    }
    auto get_columns = [&](const BRWTBottomUpBuilder::CallColumn& call_column) {
        bool metadata_only = !tmp_path.empty();
        load_columns_from_folder_callback(columns_folder_path, file_list, [&](uint64_t j, const std::string& label, std::unique_ptr<bit_vector>&& column) {
            std::lock_guard<std::mutex> lock(mu);
            if (j >= column_files.size()) column_files.resize(j + 1);
            column_files[j] = columns_folder_path + "/" + label; // Store file path for later loading in build step
            column_names.emplace_back(j, label); // Store column index and name for later serialization
            call_column(j, std::move(column)); // Invoke callback to mark column as loaded (or just metadata if metadata_only)
        }, num_threads, prefix, metadata_only);
    };
    std::string actual_tmp_nodes_dir;
    BRWT root = BRWTBottomUpBuilder::build(get_columns, linkage, tmp_path, stored_columns, column_files, num_full_nodes, num_partial_nodes, num_threads, &actual_tmp_nodes_dir, resume);
    auto t_end = std::chrono::high_resolution_clock::now();
    
    // GOAL 2: Build Metadata Manifest (.info)
    std::filesystem::path manifest_dir = std::filesystem::path(actual_tmp_nodes_dir).parent_path();
    std::string folder_id = std::filesystem::path(actual_tmp_nodes_dir).filename().string();
    std::string info_path = (manifest_dir / (folder_id + "_" + get_current_date() + ".info")).string();
    std::ofstream info(info_path);
    info << "Build Manifest\n==============\n";
    info << "Folder ID: " << folder_id << "\n";
    info << "Columns: " << column_files.size() << "\n";
    info << "Nodes Target: " << (2 * column_files.size() - 1) << "\n";
    info << "Build Time: " << std::chrono::duration<double>(t_end - t_start).count() << " seconds\n";
    info << "Peak RSS: " << (double)get_peak_RSS() / 1e9 << " GB\n";
    info.close();
    logger->info("Build manifest written to: {}", info_path);

    if (assemble) {
        std::ofstream out(output_path + ".brwt", std::ios::binary);
        BRWTBottomUpBuilder::assemble_streaming(out, linkage, stored_columns, actual_tmp_nodes_dir, column_files);
    } else {
        logger->info("Assembly skipped as requested. Nodes stored in: {}", actual_tmp_nodes_dir);
    }
    serialize_column_names(column_names, output_path + ".columns");
}

// ─── merge ────────────────────────────────────────────────────────────────────

void handle_merge(const std::string& brwt_a, const std::string& cols_a,
                  const std::string& brwt_b, const std::string& cols_b,
                  const std::string& output_prefix, size_t num_threads) {
    BRWTBottomUpBuilder::merge(brwt_a, brwt_b, output_prefix, num_threads);

    // Merge .columns metadata from the explicitly provided column files.
    auto names_a = deserialize_column_names(cols_a);
    auto names_b = deserialize_column_names(cols_b);
    if (!names_a.empty() || !names_b.empty()) {
        // Compute offset as max_column_index + 1 over all entries in names_a.
        // We CANNOT use names_a.back().first because build_BRWT inserts column
        // names from parallel worker threads (mutex-protected callbacks), so the
        // insertion order is non-deterministic — back() is not the max index.
        uint64_t offset = 0;
        for (const auto& p : names_a)
            if (p.first + 1 > offset) offset = p.first + 1;
        for (auto& p : names_b) names_a.emplace_back(p.first + offset, p.second);
        serialize_column_names(names_a, output_prefix + ".columns");
        logger->info("Merged .columns written to {}.columns ({} entries)",
                     output_prefix, names_a.size());
    } else {
        logger->warn("[merge] No .columns data found in '{}' or '{}'; output .columns not written.",
                     cols_a, cols_b);
    }
}

// ─── merge-nodes ──────────────────────────────────────────────────────────────

void handle_merge_nodes(const std::string& linkage_a_f, const std::string& node_dir_a,
                        const std::string& linkage_b_f, const std::string& node_dir_b,
                        const std::string& out_node_dir, const std::string& out_linkage,
                        size_t num_threads,
                        const std::string& cols_a_f, const std::string& cols_b_f) {
    BRWTBottomUpBuilder::merge_nodes(linkage_a_f, node_dir_a,
                                     linkage_b_f, node_dir_b,
                                     out_node_dir, out_linkage, num_threads);

    // Merge .columns metadata if both files are present
    auto names_a = deserialize_column_names(cols_a_f);
    auto names_b = deserialize_column_names(cols_b_f);
    if (!names_a.empty() || !names_b.empty()) {
        uint64_t offset = 0;
        for (const auto& p : names_a)
            if (p.first + 1 > offset) offset = p.first + 1;
        for (auto& p : names_b) names_a.emplace_back(p.first + offset, p.second);
        std::filesystem::path out_cols = std::filesystem::path(out_node_dir).parent_path()
                                        / (std::filesystem::path(out_node_dir).filename().string() + ".columns");
        serialize_column_names(names_a, out_cols.string());
        logger->info("Merged .columns written to {} ({} entries)", out_cols.string(), names_a.size());
    }
}

// ──────────────────────────────────────────────────────────────────────────────

void handle_assemble(const std::string& linkage_file, const std::string& node_dir, const std::string& output_file) {
    auto linkage = parse_linkage_matrix(linkage_file);
    size_t num_leaves = 0;
    for (const auto& c : linkage) if (c.empty()) num_leaves++;
    // Reconstruct stored_columns level-by-level to ensure dependencies are met
    std::vector<std::vector<uint64_t>> stored_columns(linkage.size());
    size_t total_nodes = linkage.size();
    std::vector<size_t> node_levels(total_nodes, 0);
    size_t max_level = 0;
    for (size_t i = 0; i < total_nodes; ++i) {
        for (uint64_t child : linkage[i]) {
            node_levels[i] = std::max(node_levels[i], node_levels[child] + 1);
        }
        max_level = std::max(max_level, node_levels[i]);
    }

    for (size_t lv = 1; lv <= max_level; ++lv) {
        for (size_t i = 0; i < total_nodes; ++i) {
            if (node_levels[i] != lv) continue;
            for (uint64_t j : linkage[i]) {
                if (linkage[j].empty()) {
                    stored_columns[i].push_back(j);
                } else {
                    for (uint64_t c : stored_columns[j]) {
                        stored_columns[i].push_back(c);
                    }
                }
            }
        }
    }
    std::ofstream out(output_file + ".brwt", std::ios::binary);
    BRWTBottomUpBuilder::assemble_streaming(out, linkage, stored_columns, node_dir, {});
    logger->info("Assembly complete: {}", output_file + ".brwt");
}

void handle_build(const std::string& annotation_dir, const std::string& prefix, const std::string& output, const std::string& tmp, const std::string& list, size_t threads, size_t full_nodes, size_t partial_nodes, size_t k, size_t seed, bool trivial, bool resume, bool assemble) {
    std::string linkage_file = output + ".linkage";
    bool recompute = !std::filesystem::exists(linkage_file);
    if (!recompute) {
        auto linkage = parse_linkage_matrix(linkage_file);
        size_t linkage_leaves = 0; for (const auto& children : linkage) if (children.empty()) linkage_leaves++;
        size_t current_columns = 0;
        if (!list.empty()) { std::ifstream f(list); std::string line; while (std::getline(f, line)) if (!line.empty()) current_columns++; }
        else { for (const auto& entry : std::filesystem::directory_iterator(annotation_dir)) if (entry.is_regular_file() && entry.path().filename().string().find(prefix) == 0) current_columns++; }
        if (linkage_leaves != current_columns) recompute = true;
    }
    if (recompute) BM_BRWTLinkageMatrix_streaming(annotation_dir, prefix, linkage_file, list, threads, k, seed, trivial);
    auto linkage = parse_linkage_matrix(linkage_file);
    build_BRWT(annotation_dir, prefix, linkage, tmp, list, full_nodes, partial_nodes, threads, output, resume, assemble);
}

void handle_relax(const std::string& input_file, uint64_t max_arity, size_t num_threads, const std::string& output_file) {
    auto t0 = std::chrono::high_resolution_clock::now();

    std::ifstream in(input_file, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("Failed to open " + input_file);

    logger->info("[relax] Loading BRWT from {} ...", input_file);
    BRWT brwt;
    if (!brwt.load(in)) throw std::runtime_error("Failed to load BRWT from " + input_file);
    in.close();

    auto t1 = std::chrono::high_resolution_clock::now();
    logger->info("[relax] Load done in {:.1f}s. Relaxing BRWT (Max Arity: {})...",
                 std::chrono::duration<double>(t1 - t0).count(), max_arity);

    BRWTOptimizer::relax(&brwt, max_arity, num_threads);

    auto t2 = std::chrono::high_resolution_clock::now();
    logger->info("[relax] Relax done in {:.1f}s. Writing output to {} ...",
                 std::chrono::duration<double>(t2 - t1).count(), output_file);

    {
        std::ofstream out(output_file, std::ios::binary);
        if (!out.is_open()) throw std::runtime_error("Failed to open " + output_file);
        brwt.serialize(out);
        out.flush();   // flush to OS buffer
    }  // destructor closes + syncs to disk here — this is the slow part

    auto t3 = std::chrono::high_resolution_clock::now();
    logger->info("[relax] Write+flush done in {:.1f}s. Total: {:.1f}s.",
                 std::chrono::duration<double>(t3 - t2).count(),
                 std::chrono::duration<double>(t3 - t0).count());
}


void handle_relax_nodes(const std::string& linkage_file, const std::string& node_dir, uint64_t max_arity, const std::string& output_linkage, size_t num_threads) {
    logger->info("Parsing linkage matrix from: {}", linkage_file);
    auto linkage = parse_linkage_matrix(linkage_file);
    
    logger->info("Relaxing BRWT Nodes in folder: {} (Max Arity: {})...", node_dir, max_arity);
    BRWTOptimizer::relax_nodes(linkage, node_dir, max_arity, num_threads);
    
    logger->info("Saving updated linkage matrix to: {}", output_linkage);
    std::ofstream out(output_linkage);
    for (size_t i = 0; i < linkage.size(); ++i) {
        if (linkage[i].empty()) continue;
        out << i;
        for (uint64_t child : linkage[i]) out << " " << child;
        out << "\n";
    }
    logger->info("Relaxation complete.");
}

void handle_server_mmap(const std::string& brwt_f, const std::string& cols_f, const std::string& idx_f, uint16_t port) {
    logger->info("Building global topology from {}...", idx_f);
    mtg::cli::MmapNode root = mtg::cli::build_global_topology(idx_f, brwt_f);
    
    logger->info("Initializing EnginePool...");
    auto engine_pool = std::make_shared<mtg::cli::EnginePool>(brwt_f, root);
    
    logger->info("Loading column names from {}...", cols_f);
    auto names = std::make_shared<std::vector<std::pair<uint64_t, std::string>>>(deserialize_column_names(cols_f));

    HttpServer server;
    server.config.port = port;
    server.config.thread_pool_size = std::thread::hardware_concurrency();

    server.resource["^/query$"]["POST"] = [engine_pool, names](auto resp, auto req) {
        try {
            std::string body = req->content.string();
            Json::Value req_json;
            Json::CharReaderBuilder builder;
            std::string errs;
            std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
            
            if (!reader->parse(body.data(), body.data() + body.size(), &req_json, &errs)) {
                throw std::runtime_error("JSON parse error: " + errs);
            }

            if (!req_json.isMember("row_ids") || !req_json["row_ids"].isArray()) {
                throw std::runtime_error("Request must contain 'row_ids' array.");
            }
            
            std::vector<uint64_t> ids;
            for (const auto& id_val : req_json["row_ids"]) {
                ids.push_back(id_val.asUInt64());
            }
            
            auto engine = engine_pool->acquire();
            auto results = engine->get_rows(ids);
            engine_pool->release(engine);
            
            Json::Value resp_json(Json::objectValue);
            for (size_t i = 0; i < ids.size(); ++i) {
                Json::Value row_res(Json::arrayValue);
                for (auto bit : results[i]) {
                    auto it = std::find_if(names->begin(), names->end(), 
                                           [&](auto& p){ return p.first == bit.first; });
                    if (it != names->end()) row_res.append(it->second);
                }
                resp_json[std::to_string(ids[i])] = row_res;
            }
            
            std::string out = Json::writeString(Json::StreamWriterBuilder(), resp_json);
            *resp << "HTTP/1.1 200 OK\r\nContent-Length: " << out.size() << "\r\n\r\n" << out;
            
        } catch (const std::exception& e) {
            std::string msg = std::string("Error: ") + e.what();
            *resp << "HTTP/1.1 400 Bad Request\r\nContent-Length: " << msg.size() << "\r\n\r\n" << msg;
        }
    };

    logger->info("Mmap Server started on http://localhost:{}", port);
    server.start();
}

void handle_server(const std::string& brwt_f, const std::string& cols_f, uint16_t port) {
    logger->info("Loading BRWT index from {}...", brwt_f);
    std::ifstream in(brwt_f, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("Failed to open " + brwt_f);
    
    auto brwt = std::make_shared<BRWT>();
    if (!brwt->load(in)) throw std::runtime_error("Failed to load BRWT");
    
    logger->info("Loading column names from {}...", cols_f);
    auto names = std::make_shared<std::vector<std::pair<uint64_t, std::string>>>(deserialize_column_names(cols_f));

    HttpServer server;
    server.config.port = port;
    server.config.thread_pool_size = std::thread::hardware_concurrency();

    server.resource["^/query$"]["POST"] = [brwt, names](auto resp, auto req) {
        try {
            std::string body = req->content.string();
            Json::Value req_json;
            Json::CharReaderBuilder builder;
            std::string errs;
            std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
            
            if (!reader->parse(body.data(), body.data() + body.size(), &req_json, &errs)) {
                throw std::runtime_error("JSON parse error: " + errs);
            }

            if (!req_json.isMember("row_ids") || !req_json["row_ids"].isArray()) {
                throw std::runtime_error("Request must contain 'row_ids' array.");
            }
            
            std::vector<uint64_t> ids;
            for (const auto& id_val : req_json["row_ids"]) {
                ids.push_back(id_val.asUInt64());
            }
            
            auto results = brwt->get_rows(ids);
            
            Json::Value resp_json(Json::objectValue);
            for (size_t i = 0; i < ids.size(); ++i) {
                Json::Value row_res(Json::arrayValue);
                for (auto bit : results[i]) {
                    auto it = std::find_if(names->begin(), names->end(), 
                                           [&](auto& p){ return p.first == bit; });
                    if (it != names->end()) row_res.append(it->second);
                }
                resp_json[std::to_string(ids[i])] = row_res;
            }
            
            std::string out = Json::writeString(Json::StreamWriterBuilder(), resp_json);
            *resp << "HTTP/1.1 200 OK\r\nContent-Length: " << out.size() << "\r\n\r\n" << out;
            
        } catch (const std::exception& e) {
            std::string msg = std::string("Error: ") + e.what();
            *resp << "HTTP/1.1 400 Bad Request\r\nContent-Length: " << msg.size() << "\r\n\r\n" << msg;
        }
    };

    logger->info("Server started on http://localhost:{}", port);
    server.start();
}

// ─── query-nodes ──────────────────────────────────────────────────────────────────────────────

void handle_query_nodes(const std::string& linkage_file, const std::string& node_dir_str,
                        const std::string& cols_file, const std::string& ids_str,
                        bool log_stats = false) {
    auto linkage = parse_linkage_matrix(linkage_file);
    if (linkage.empty()) throw std::runtime_error("Empty or invalid linkage file: " + linkage_file);

    // Parse row IDs (same comma-separated syntax as `query`)
    std::vector<uint64_t> ids;
    {
        std::string cleaned = ids_str;
        cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), '{'), cleaned.end());
        cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), '}'), cleaned.end());
        std::stringstream ss(cleaned); std::string seg;
        while (std::getline(ss, seg, ',')) if (!seg.empty()) ids.push_back(std::stoull(seg));
    }
    if (ids.empty()) return;

    // Delegate tree traversal to BRWTBottomUpBuilder::query_nodes (has friend access to BRWT).
    auto results = BRWTBottomUpBuilder::query_nodes(ids, linkage, node_dir_str, log_stats);

    // Resolve column IDs to names and print.
    auto names = deserialize_column_names(cols_file);
    for (size_t i = 0; i < ids.size(); ++i) {
        std::cout << "Row " << ids[i] << ": ";
        std::sort(results[i].begin(), results[i].end()); // deterministic output order
        for (uint64_t col_id : results[i]) {
            auto it = std::find_if(names.begin(), names.end(),
                                   [&](auto& p){ return p.first == col_id; });
            if (it != names.end()) std::cout << it->second << " ";
            else                   std::cout << "col_" << col_id << " ";
        }
        std::cout << "\n";
    }
}

// ───────────────────────────────────────────────────────────────────────────────

void handle_query(const std::string& ids_str, const std::string& brwt_f, const std::string& cols_f) {
    std::vector<uint64_t> ids;
    std::string cleaned = ids_str;
    cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), '{'), cleaned.end());
    cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), '}'), cleaned.end());
    std::stringstream ss(cleaned); std::string segment;
    while (std::getline(ss, segment, ',')) if (!segment.empty()) ids.push_back(std::stoull(segment));

    std::ifstream in(brwt_f, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("query: cannot open BRWT file: " + brwt_f);

    BRWT brwt;
    if (!brwt.load(in))
        throw std::runtime_error("query: failed to load BRWT from: " + brwt_f
            + " (file may be corrupt or wrong format)");

    logger->info("[query] Loaded {} columns, {} rows from {}",
                 brwt.num_columns(), brwt.num_rows(), brwt_f);

    // Validate row IDs before querying to avoid assert/segfault inside slice_rows
    for (uint64_t id : ids) {
        if (id >= brwt.num_rows())
            throw std::runtime_error("query: row id " + std::to_string(id)
                + " out of range (index has " + std::to_string(brwt.num_rows()) + " rows)");
    }

    auto results = brwt.get_rows(ids);
    auto names   = deserialize_column_names(cols_f);

    for (size_t i = 0; i < ids.size(); ++i) {
        std::cout << "Row " << ids[i] << ": ";
        for (auto bit : results[i]) {
            auto it = std::find_if(names.begin(), names.end(), [&](auto& p){ return p.first == bit; });
            if (it != names.end()) std::cout << it->second << " ";
            else                   std::cout << "col_" << bit << " ";
        }
        std::cout << "\n";
    }
}

void handle_benchmark(const std::string& brwt_f, const std::string& cols_f) {
    auto t_start = std::chrono::high_resolution_clock::now();

    std::ifstream in(brwt_f, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("benchmark: cannot open BRWT file: " + brwt_f);

    BRWT brwt;
    if (!brwt.load(in))
        throw std::runtime_error("benchmark: failed to load BRWT from: " + brwt_f);

    auto names = deserialize_column_names(cols_f);

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    size_t rss_bytes = get_curr_RSS();

    std::cout << format_time(elapsed) << ", " << rss_bytes / 1024 << "\n";
}


// ─── query-columns (monolithic .brwt) ────────────────────────────────────────
// Usage: query-columns <brwtFile> <columnsFile> <outDir> --name_list <nameListFile>
//
// Looks up each column name in <columnsFile>, extracts its full row annotation
// from the BRWT, and writes <outDir>/<name>.sd (sdsl::sd_vector<> binary format).

void handle_query_columns(const std::string& brwt_f,
                          const std::string& cols_f,
                          const std::string& out_dir_s,
                          const std::vector<std::string>& col_names) {
    // Load the BRWT index.
    std::ifstream in(brwt_f, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("query-columns: cannot open BRWT file: " + brwt_f);
    BRWT brwt;
    if (!brwt.load(in))
        throw std::runtime_error("query-columns: failed to load BRWT from: " + brwt_f);
    logger->info("[query-columns] Loaded {} columns, {} rows from {}",
                 brwt.num_columns(), brwt.num_rows(), brwt_f);

    // Build name → column_id map from the .columns file.
    auto all_names = deserialize_column_names(cols_f);
    std::unordered_map<std::string, uint64_t> name_to_id;
    for (const auto& [cid, cname] : all_names)
        name_to_id[cname] = cid;

    // Resolve each requested name.
    std::vector<std::pair<uint64_t, std::string>> requests;
    for (const auto& name : col_names) {
        auto it = name_to_id.find(name);
        if (it == name_to_id.end()) {
            logger->warn("[query-columns] Column '{}' not found in .columns file, skipping.", name);
            continue;
        }
        requests.emplace_back(it->second, name);
    }

    if (requests.empty()) {
        logger->warn("[query-columns] No valid column names resolved.");
        return;
    }

    BRWTBottomUpBuilder::query_columns(brwt, requests, out_dir_s);
    logger->info("[query-columns] Done. {} columns written to {}", requests.size(), out_dir_s);
}

// ─── query-columns-nodes (node-folder format) ─────────────────────────────────
// Usage: query-columns-nodes <linkageFile> <nodeDir> <columnsFile> <outDir>
//                            --name_list <nameListFile>
//
// Same as query-columns but operates on the node-folder format produced by
// `build --no-assemble`.  No assembly step is required.

void handle_query_columns_nodes(const std::string& linkage_f,
                                const std::string& node_dir_s,
                                const std::string& cols_f,
                                const std::string& out_dir_s,
                                const std::vector<std::string>& col_names) {
    auto linkage = parse_linkage_matrix(linkage_f);
    if (linkage.empty())
        throw std::runtime_error("query-columns-nodes: empty or invalid linkage: " + linkage_f);

    // Build name → column_id map.
    auto all_names = deserialize_column_names(cols_f);
    std::unordered_map<std::string, uint64_t> name_to_id;
    for (const auto& [cid, cname] : all_names)
        name_to_id[cname] = cid;

    std::vector<std::pair<uint64_t, std::string>> requests;
    for (const auto& name : col_names) {
        auto it = name_to_id.find(name);
        if (it == name_to_id.end()) {
            logger->warn("[query-columns-nodes] Column '{}' not found in .columns file, skipping.", name);
            continue;
        }
        requests.emplace_back(it->second, name);
    }

    if (requests.empty()) {
        logger->warn("[query-columns-nodes] No valid column names resolved.");
        return;
    }

    BRWTBottomUpBuilder::query_columns_nodes(requests, linkage, node_dir_s, out_dir_s);
    logger->info("[query-columns-nodes] Done. {} columns written to {}", requests.size(), out_dir_s);
}

// ─── root-append (monolithic .brwt) ───────────────────────────────────────────
// Extends the BRWT matrix by appending (target_len - num_rows) zero rows at the
// end, writing the result to a new .brwt file.

void handle_root_append(const std::string& input_brwt,
                        uint64_t target_len,
                        const std::string& output_brwt) {
    BRWTBottomUpBuilder::root_append(input_brwt, target_len, output_brwt);
}

// ─── root-append-nodes (node-folder format) ────────────────────────────────────
// Same as root-append but operates on the node-folder format.
// Backs up <root_id> as ori_node_<root_id> then overwrites <root_id>.

void handle_root_append_nodes(const std::string& linkage_f,
                              const std::string& node_dir_s,
                              uint64_t target_len) {
    auto linkage = parse_linkage_matrix(linkage_f);
    if (linkage.empty())
        throw std::runtime_error("root-append-nodes: empty or invalid linkage: " + linkage_f);
    BRWTBottomUpBuilder::root_append_nodes(linkage, node_dir_s, target_len);
}

void show_usage(const std::string& program_name) {
    std::cerr << "Multi-BRWT Optimized Build Tool (Adaptive Strategy Engine)\n"
              << "Version: 0.3.0\n\n"
              << "Usage: " << program_name << " <command> [arguments]\n\n"
              << "Commands:\n"
              << "  build <annotationDir> <prefix> <outputFile> [tmpDir] [flags]\n"
              << "      Build a Multi-BRWT from .sd files. If tmpDir is provided, uses disk-assisted build.\n"
              << "      Flags:\n"
              << "        --file_list <file>    Path to a file containing the list of input .sd files.\n"
              << "        --threads <N>         Total CPU threads for internal math (Default: Max).\n"
              << "        --full-nodes <N>      Max parallel merges for sparse nodes (Default: --threads).\n"
              << "        --partial-nodes <N>   Max parallel merges for dense nodes (Default: RAM-aware).\n"
              << "        --linkage_trivial     Use sequential/trivial linkage instead of greedy.\n"
              << "        --resume              Detect existing nodes in tmpDir and repair corrupted ones.\n"
              << "        --no-assemble         Finish merges and exit without creating final single file.\n\n"
              << "  assemble <linkageFile> <nodeDir> <outputFile>\n"
              << "      Stitch existing nodes from nodeDir into a single .brwt file.\n\n"
              << "  merge <brwtFileA> <columnFileA> <brwtFileB> <columnFileB> <outputPrefix> [--threads <N>]\n"
              << "      Merge two monolithic .brwt files (same row length) into one.\n"
              << "      Both .columns files must be provided explicitly; the merged result is written\n"
              << "      to <outputPrefix>.brwt and <outputPrefix>.columns.\n\n"
              << "  merge-nodes <linkageA> <nodeDirA> <linkageB> <nodeDirB> <outNodeDir> <outLinkage> [flags]\n"
              << "      Merge two node-folder BRWT representations (same row length) into one.\n"
              << "      Flags:\n"
              << "        --cols_a <file>   .columns file for index A (optional).\n"
              << "        --cols_b <file>   .columns file for index B (optional).\n"
              << "        --threads <N>     CPU threads (Default: Max).\n\n"
              << "  relax <inputBRWT> <maxArity> <outputBRWT> [--threads <N>]\n"
              << "      Flatten a monolithic .brwt file to optimize space and speed.\n\n"
              << "  relax-nodes <linkageFile> <nodeDir> <maxArity> <outputLinkageFile> [--threads <N>]\n"
              << "      Flatten a tree stored as a folder of nodes (Low RAM mode).\n\n"
              << "  query <brwtFile> <columnsFile> <rowIds>\n"
              << "      Query row bitsets from a single .brwt file (e.g., \"0,100,50084136181\").\n\n"
              << "  query-nodes <linkageFile> <nodeDir> <columnsFile> <rowIds> [--log]\n"
              << "      Query row bitsets from a BRWT stored as a nodes folder.\n"
              << "      <linkageFile>  : linkage matrix file (.linkage) used during build.\n"
              << "      <nodeDir>      : folder containing node files (node_<id> files + root <id> file).\n"
              << "      <columnsFile>  : column-name index file (.columns).\n"
              << "      <rowIds>       : comma-separated row IDs (e.g., \"0,100,50084136181\").\n"
              << "      --log          : print query stats (nodes loaded, hits, etc.) to stderr.\n\n"
              << "  query-columns <brwtFile> <columnsFile> <outDir> --name_list <nameListFile>\n"
              << "      Extract column annotations from a monolithic .brwt file.\n"
              << "      Each named column is written as <outDir>/<name>.sd (sdsl::sd_vector format).\n"
              << "      <nameListFile> : plain-text file with one column name per line.\n\n"
              << "  query-columns-nodes <linkageFile> <nodeDir> <columnsFile> <outDir> --name_list <nameListFile>\n"
              << "      Extract column annotations from a node-folder BRWT (no assembly needed).\n"
              << "      Each named column is written as <outDir>/<name>.sd (sdsl::sd_vector format).\n"
              << "      <nameListFile> : plain-text file with one column name per line.\n\n"
              << "  root-append <inputBrwt> <targetLen> <outputBrwt>\n"
              << "      Extend a monolithic .brwt matrix to targetLen rows by appending zeros.\n"
              << "      Writes a new .brwt file; input is unchanged.\n\n"
              << "  root-append-nodes <linkageFile> <nodeDir> <targetLen>\n"
              << "      Extend a node-folder BRWT matrix to targetLen rows by appending zeros.\n"
              << "      Backs up original root as ori_node_<root_id>; overwrites <root_id>.\n\n"
              << "  build-mmap-index <inputFile.brwt> <outputFile.brwt.idx> [--output-log <logFile>]\n"
              << "      Generate a sidecar index for O(1) random access to BRWT nodes.\n\n"
              << "  server-mmap <brwtFile> <columnsFile> <indexFile> [--port <P>]\n"
              << "      Start a REST API server for row queries using an mmap index.\n\n"
              << "  server <brwtFile> <columnsFile> [--port <P>]\n"
              << "      Start a REST API server for row queries.\n\n"
              << "  benchmark <brwtFile> <columnsFile>\n"
              << "      Load BRWT and columns file to measure loading time and RAM usage.\n";
}


} // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        show_usage(argv[0]);
        return 1;
    }
    std::string cmd = argv[1];
    try {
        if (cmd == "build") {
            if (argc < 5) {
                show_usage(argv[0]);
                return 1;
            }
            std::string dir = argv[2], pre = argv[3], out = argv[4], tmp = (argc > 5 && std::string(argv[5]).find("--") != 0) ? argv[5] : "";
            size_t threads = std::thread::hardware_concurrency();
            size_t full_nodes = 0, partial_nodes = 0;
            bool trivial = false, resume = false, assemble = true;
            for(int i=5; i<argc; ++i) {
                if (std::string(argv[i]) == "--threads") threads = std::stoul(argv[++i]);
                else if (std::string(argv[i]) == "--full-nodes") full_nodes = std::stoul(argv[++i]);
                else if (std::string(argv[i]) == "--partial-nodes") partial_nodes = std::stoul(argv[++i]);
                else if (std::string(argv[i]) == "--linkage_trivial") trivial = true;
                else if (std::string(argv[i]) == "--resume") resume = true;
                else if (std::string(argv[i]) == "--no-assemble") assemble = false;
            }
            handle_build(dir, pre, out, tmp, "", threads, full_nodes, partial_nodes, 0, 42, trivial, resume, assemble);
        } else if (cmd == "merge") {
            // merge <brwtA> <columnFileA> <brwtB> <columnFileB> <outputPrefix> [--threads <N>]
            if (argc < 7) { show_usage(argv[0]); return 1; }
            size_t threads = std::thread::hardware_concurrency();
            for (int i = 7; i < argc; ++i)
                if (std::string(argv[i]) == "--threads" && i+1 < argc) threads = std::stoul(argv[++i]);
            handle_merge(argv[2], argv[3], argv[4], argv[5], argv[6], threads);
        } else if (cmd == "merge-nodes") {
            // merge-nodes <linkA> <ndirA> <linkB> <ndirB> <outDir> <outLinkage> [flags]
            if (argc < 8) { show_usage(argv[0]); return 1; }
            size_t threads = std::thread::hardware_concurrency();
            std::string cols_a, cols_b;
            for (int i = 8; i < argc; ++i) {
                std::string a = argv[i];
                if (a == "--threads" && i+1 < argc) threads = std::stoul(argv[++i]);
                else if (a == "--cols_a" && i+1 < argc) cols_a = argv[++i];
                else if (a == "--cols_b" && i+1 < argc) cols_b = argv[++i];
            }
            handle_merge_nodes(argv[2], argv[3], argv[4], argv[5], argv[6], argv[7],
                               threads, cols_a, cols_b);
        } else if (cmd == "assemble") {
            if (argc < 5) {
                show_usage(argv[0]);
                return 1;
            }
            handle_assemble(argv[2], argv[3], argv[4]);
        } else if (cmd == "relax") {
            if (argc < 5) {
                show_usage(argv[0]);
                return 1;
            }
            std::string input = argv[2];
            uint64_t arity = std::stoull(argv[3]);
            std::string output = argv[4];
            size_t threads = std::thread::hardware_concurrency();
            for(int i=5; i<argc; ++i) {
                if (std::string(argv[i]) == "--threads") threads = std::stoul(argv[++i]);
            }
            handle_relax(input, arity, threads, output);
        } else if (cmd == "relax-nodes") {
            if (argc < 6) {
                show_usage(argv[0]);
                return 1;
            }
            std::string linkage_f = argv[2];
            std::string nodes_d = argv[3];
            uint64_t arity = std::stoull(argv[4]);
            std::string out_linkage = argv[5];
            size_t threads = std::thread::hardware_concurrency();
            for(int i=6; i<argc; ++i) {
                if (std::string(argv[i]) == "--threads") threads = std::stoul(argv[++i]);
            }
            handle_relax_nodes(linkage_f, nodes_d, arity, out_linkage, threads);
        } else if (cmd == "query") {
            if (argc < 5) {
                show_usage(argv[0]);
                return 1;
            }
            handle_query(argv[4], argv[2], argv[3]);
        } else if (cmd == "query-nodes") {
            // query-nodes <linkageFile> <nodeDir> <columnsFile> <rowIds> [--log]
            if (argc < 6) {
                show_usage(argv[0]);
                return 1;
            }
            bool log_stats = false;
            for (int i = 6; i < argc; ++i) {
                if (std::string(argv[i]) == "--log") log_stats = true;
            }
            handle_query_nodes(argv[2], argv[3], argv[4], argv[5], log_stats);
        } else if (cmd == "query-columns") {
            // query-columns <brwtFile> <columnsFile> <outDir> --name_list <nameListFile>
            if (argc < 6) { show_usage(argv[0]); return 1; }
            std::string name_list_file;
            for (int i = 5; i < argc; ++i) {
                if (std::string(argv[i]) == "--name_list" && i + 1 < argc)
                    name_list_file = argv[++i];
            }
            if (name_list_file.empty()) {
                std::cerr << "Error: query-columns requires --name_list <file>\n";
                show_usage(argv[0]); return 1;
            }
            std::vector<std::string> col_names;
            { std::ifstream lf(name_list_file); if (!lf.is_open()) throw std::runtime_error("Cannot open name list: " + name_list_file);
              std::string ln; while (std::getline(lf, ln)) if (!ln.empty()) col_names.push_back(ln); }
            handle_query_columns(argv[2], argv[3], argv[4], col_names);
        } else if (cmd == "query-columns-nodes") {
            // query-columns-nodes <linkageFile> <nodeDir> <columnsFile> <outDir> --name_list <nameListFile>
            if (argc < 7) { show_usage(argv[0]); return 1; }
            std::string name_list_file;
            for (int i = 6; i < argc; ++i) {
                if (std::string(argv[i]) == "--name_list" && i + 1 < argc)
                    name_list_file = argv[++i];
            }
            if (name_list_file.empty()) {
                std::cerr << "Error: query-columns-nodes requires --name_list <file>\n";
                show_usage(argv[0]); return 1;
            }
            std::vector<std::string> col_names;
            { std::ifstream lf(name_list_file); if (!lf.is_open()) throw std::runtime_error("Cannot open name list: " + name_list_file);
              std::string ln; while (std::getline(lf, ln)) if (!ln.empty()) col_names.push_back(ln); }
            handle_query_columns_nodes(argv[2], argv[3], argv[4], argv[5], col_names);
        } else if (cmd == "root-append") {
            // root-append <inputBrwt> <targetLen> <outputBrwt>
            if (argc < 5) { show_usage(argv[0]); return 1; }
            handle_root_append(argv[2], std::stoull(argv[3]), argv[4]);
        } else if (cmd == "root-append-nodes") {
            // root-append-nodes <linkageFile> <nodeDir> <targetLen>
            if (argc < 5) { show_usage(argv[0]); return 1; }
            handle_root_append_nodes(argv[2], argv[3], std::stoull(argv[4]));
        } else if (cmd == "build-mmap-index") {
            if (argc < 4) { show_usage(argv[0]); return 1; }
            std::string log_file = "";
            if (argc >= 6 && std::string(argv[4]) == "--output-log") {
                log_file = argv[5];
            }
            mtg::cli::handle_build_mmap_index(argv[2], argv[3], log_file);
        } else if (cmd == "server-mmap") {
            if (argc < 5) {
                show_usage(argv[0]);
                return 1;
            }
            std::string brwt_f = argv[2];
            std::string cols_f = argv[3];
            std::string idx_f = argv[4];
            uint16_t port = 8080;
            for(int i=5; i<argc; ++i) {
                if (std::string(argv[i]) == "--port" && i+1 < argc) port = std::stoul(argv[++i]);
            }
            handle_server_mmap(brwt_f, cols_f, idx_f, port);
        } else if (cmd == "server") {
            if (argc < 4) {
                show_usage(argv[0]);
                return 1;
            }
            std::string brwt_f = argv[2];
            std::string cols_f = argv[3];
            uint16_t port = 8080;
            for(int i=4; i<argc; ++i) {
                if (std::string(argv[i]) == "--port" && i+1 < argc) port = std::stoul(argv[++i]);
            }
            handle_server(brwt_f, cols_f, port);
        } else if (cmd == "benchmark") {
            if (argc < 4) { show_usage(argv[0]); return 1; }
            handle_benchmark(argv[2], argv[3]);
        } else {
            show_usage(argv[0]);
            return 1;
        }
    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << "\n"; return 1; }
    return 0;
}
