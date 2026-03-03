import subprocess
import os
import sys
import time
import shutil

# Configuration
METAGRAPH_BIN = "../build/metagraph"
# METAGRAPH_BIN="./metagraph"
GEN_VECTORS_BIN = "./gen_sdsl_vectors"
DATA_DIR = "benchmark_data"

# Benchmark Parameters
# Increased size to make I/O and bitwise math optimizations visible.
NUM_ROWS = 10_000_000  # 10M rows 
NUM_COLS = 1000        # 1000 columns
SPARSITY = 0.01        # 1% density
THREADS = os.cpu_count() or 1

def run_command(cmd):
    print(f"Executing: {cmd}")
    start = time.time()
    subprocess.check_call(cmd, shell=True)
    return time.time() - start

def setup():
    # if os.path.exists(DATA_DIR):
        # shutil.rmtree(DATA_DIR)
    # os.makedirs(DATA_DIR)

    # Compile generator if missing
    if not os.path.exists(GEN_VECTORS_BIN):
        print("Compiling gen_sdsl_vectors...")
        subprocess.check_call("bash compile_test.sh", shell=True)

def main():
    if not os.path.exists(METAGRAPH_BIN):
        print(f"Error: metagraph binary not found at {METAGRAPH_BIN}")
        print("Please build the project first.")
        return

    setup()

    print(f"\n--- Benchmark Configuration ---")
    print(f"Rows:      {NUM_ROWS:,}")
    print(f"Columns:   {NUM_COLS:,}")
    print(f"Threads:   {THREADS}")
    print(f"Sparsity:  {SPARSITY}")
    print(f"--------------------------------\n")

    # 1. Generate Vectors
    # print(f"Step 1: Generating {NUM_COLS} input columns...")
    # gen_time = run_command(f"{GEN_VECTORS_BIN} {NUM_ROWS} {NUM_COLS} {DATA_DIR} col {SPARSITY}")
    # print(f"Generation took: {gen_time:.2f} seconds\n")

    # 2. Build BRWT
    print(f"Step 2: Building BRWT (Trivial Linkage)...")
    # We use trivial linkage to specifically isolate the performance of 
    # the leaf loading, bitwise ORing, and internal tree construction logic.
    build_cmd = (
        f"{METAGRAPH_BIN} build {DATA_DIR} col {DATA_DIR}/bench_new.brwt "
        f"--linkage_trivial --threads {THREADS}"
    )
    
    build_time = run_command(build_cmd)

    print(f"\n--- Performance Result ---")
    print(f"Total Build Time: {build_time:.2f} seconds")
    print(f"Throughput:       {NUM_COLS / build_time:.2f} columns/sec")
    print(f"--------------------------")
    
    # Clean up large data
    # print("\nCleaning up benchmark data...")
    # shutil.rmtree(DATA_DIR)

if __name__ == "__main__":
    main()
