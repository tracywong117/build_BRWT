#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <filesystem>
#include <sdsl/sd_vector.hpp>
#include <sdsl/bit_vectors.hpp>

namespace fs = std::filesystem;

// Usage: ./gen_sdsl_vectors <num_rows> <num_vectors> <output_dir> <prefix> <sparsity>
int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <num_rows> <num_vectors> <output_dir> <prefix> <sparsity (0.0-1.0)>\n";
        return 1;
    }

    uint64_t num_rows = std::stoull(argv[1]);
    int num_vectors = std::stoi(argv[2]);
    std::string output_dir = argv[3];
    std::string prefix = argv[4];
    double sparsity = std::stod(argv[5]);

    if (!fs::exists(output_dir)) {
        fs::create_directories(output_dir);
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution d(sparsity);

    // To store ground truth
    std::string truth_file = output_dir + "/" + "_" + prefix + "_truth.txt";
    std::ofstream truth(truth_file);

    for (int i = 0; i < num_vectors; ++i) {
        sdsl::bit_vector bv(num_rows, 0);
        
        std::string col_name = prefix + "_" + std::to_string(i);

        for (uint64_t j = 0; j < num_rows; ++j) {
            if (d(gen)) {
                bv[j] = 1;
                // Write to truth file: row_id col_name
                truth << j << " " << col_name << "\n";
            }
        }

        sdsl::sd_vector<> sd_vec(bv);
        std::string filename = output_dir + "/" + col_name + ".sd";
        
        // Save using sdsl structure serialization
        if (!sdsl::store_to_file(sd_vec, filename)) {
            std::cerr << "Error writing file " << filename << "\n";
            return 1;
        }
        std::cout << "Generated " << filename << "\n";
    }

    truth.close();
    return 0;
}
