import subprocess
import os
import sys
import time
import shutil
import random

# Configuration
METAGRAPH_BIN = "../build/metagraph_DNA"
# METAGRAPH_BIN = "../build_original_optimized/metagraph_DNA"
GEN_VECTORS_BIN = "./gen_sdsl_vectors"
TMP_DIR = "/media/data/tracy/build_BRWT/test/tmp"
DATA_DIR = "benchmark_data"
OUTPUT_FILE = "bench_new.brwt"

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
    # Compile generator if missing
    if not os.path.exists(GEN_VECTORS_BIN):
        print("Compiling gen_sdsl_vectors...")
        subprocess.check_call("bash compile_test.sh", shell=True)

def verify_correctness():
    truth_file = os.path.join(DATA_DIR, f"_col_truth.txt")
    if not os.path.exists(truth_file):
        print("Truth file not found, skipping verification.")
        return

    print("\n--- Verifying Correctness ---")
    
    # Load sample truth data (first 5 distinct rows found)
    truth_sample = {} # row_id -> set of column names
    with open(truth_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            row_id_str, col_name = parts
            row_id = int(row_id_str)
            if row_id not in truth_sample:
                if len(truth_sample) >= 5:
                    continue
                truth_sample[row_id] = set()
            truth_sample[row_id].add(col_name + ".sd")
            if len(truth_sample) >= 5 and row_id not in truth_sample:
                break

    for row_id, expected_cols in truth_sample.items():
        # Query format: metagraph query <brwtFile> <columnsFile> <rowIds>
        # Note: we wrap row_id in {} as expected by the query handler
        query_cmd = f"{METAGRAPH_BIN} query {DATA_DIR}/{OUTPUT_FILE}.brwt {DATA_DIR}/{OUTPUT_FILE}.columns '{{{row_id}}}'"
        print(f"  Checking Row {row_id}...")
        try:
            output = subprocess.check_output(query_cmd, shell=True).decode()
            print(f"    Query Output: {output.strip()}")
            
            # Parse output: "Row {id}: col1.sd col2.sd ..."
            actual_cols = set()
            for line in output.splitlines():
                if line.startswith(f"Row {row_id}:"):
                    actual_part = line.split(":", 1)[1].strip()
                    actual_cols = set(actual_part.split())
                    break
            
            if actual_cols == expected_cols:
                print(f"    [PASS] Found {len(actual_cols)} matching columns.")
            else:
                print(f"    [FAIL] Row {row_id} mismatch!")
                print(f"      Expected ({len(expected_cols)}): {sorted(list(expected_cols))[:10]}...")
                print(f"      Actual   ({len(actual_cols)}): {sorted(list(actual_cols))[:10]}...")
                sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"    [ERROR] Query failed: {e}")
            sys.exit(1)
    
    print("Correctness verification passed successfully.")

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
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Step 1: Generating {NUM_COLS} input columns...")
        run_command(f"{GEN_VECTORS_BIN} {NUM_ROWS} {NUM_COLS} {DATA_DIR} col {SPARSITY}")
    else:
        print(f"Step 1: Using existing benchmark data in {DATA_DIR}")

    # 2. Build BRWT
    print(f"\nStep 2: Building BRWT (Trivial Linkage)...")
    # We use trivial linkage to specifically isolate the performance of 
    # the leaf loading, bitwise ORing, and internal tree construction logic.
    build_cmd = (
        f"{METAGRAPH_BIN} build {DATA_DIR} col {DATA_DIR}/{OUTPUT_FILE} {TMP_DIR} "
        f"--linkage_trivial --threads {THREADS}"
    )
    
    build_time = run_command(build_cmd)

    # 3. Verify
    verify_correctness()

    print(f"\n--- Performance Result ---")
    print(f"Total Build Time: {build_time:.2f} seconds")
    print(f"Throughput:       {NUM_COLS / build_time:.2f} columns/sec")
    print(f"--------------------------")

if __name__ == "__main__":
    main()
