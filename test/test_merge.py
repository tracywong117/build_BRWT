import subprocess
import os
import sys
import shutil

METAGRAPH_BIN = "../build/metagraph" # Default assumption
GEN_VECTORS_BIN = "./gen_sdsl_vectors"

def run_command(cmd):
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)

def main():
    if len(sys.argv) > 1:
        global METAGRAPH_BIN
        METAGRAPH_BIN = sys.argv[1]

    # Try finding metagraph if not valid
    if not os.path.exists(METAGRAPH_BIN):
        # check parent parent build etc.
        possibilities = [
            "../build/metagraph",
            "../build_static/metagraph",
            "./build/metagraph",
            "metagraph"
        ]
        for p in possibilities:
            if os.path.exists(p):
                METAGRAPH_BIN = p
                break
        else:
            print(f"Error: metagraph binary not found at {METAGRAPH_BIN}.")
            sys.exit(1)

    print(f"Using metagraph binary at: {METAGRAPH_BIN}")

    # Compile gen_sdsl_vectors if needed
    if not os.path.exists(GEN_VECTORS_BIN):
        print("Compiling gen_sdsl_vectors...")
        run_command("bash compile_test.sh")

    # Parameters (A and B must have the same number of rows for merge)
    N = 200 # Row count
    COLS_A = 80
    COLS_B = 120
    SPARSITY = 0.02
    
    DATA_DIR = "test_merge_data"
    # Clean up any existing data at the start to ensure clean test
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR)
    os.makedirs(f"{DATA_DIR}/a")
    os.makedirs(f"{DATA_DIR}/b")

    # 1. Generate Tree A Vectors (N rows, COLS_A cols)
    print("Generating tree A vectors...")
    run_command(f"{GEN_VECTORS_BIN} {N} {COLS_A} {DATA_DIR}/a a {SPARSITY}")
    
    # 2. Generate Tree B Vectors (N rows, COLS_B cols)
    print("Generating tree B vectors...")
    run_command(f"{GEN_VECTORS_BIN} {N} {COLS_B} {DATA_DIR}/b b {SPARSITY}")

    # 3. Build Tree A BRWT
    print("Building Tree A BRWT...")
    run_command(f"{METAGRAPH_BIN} build {DATA_DIR}/a a {DATA_DIR}/a_brwt --linkage_trivial")

    # 4. Build Tree B BRWT
    print("Building Tree B BRWT...")
    run_command(f"{METAGRAPH_BIN} build {DATA_DIR}/b b {DATA_DIR}/b_brwt --linkage_trivial")

    # 5. Merge A and B
    print("Merging BRWT A and B...")
    run_command(f"{METAGRAPH_BIN} merge {DATA_DIR}/a_brwt.brwt {DATA_DIR}/a_brwt.columns {DATA_DIR}/b_brwt.brwt {DATA_DIR}/b_brwt.columns {DATA_DIR}/merged")

    # ---------------------------------------------------------
    # Size Report
    # ---------------------------------------------------------
    def get_size(path):
        if os.path.exists(path):
            return os.path.getsize(path)
        return -1

    print("\n--- File Sizes (Bytes) ---")
    print(f"Tree A BRWT:  {get_size(f'{DATA_DIR}/a_brwt.brwt')}")
    print(f"Tree B BRWT:  {get_size(f'{DATA_DIR}/b_brwt.brwt')}")
    print(f"Merged BRWT:  {get_size(f'{DATA_DIR}/merged.brwt')}")
    
    # ---------------------------------------------------------
    # Verification Logic
    # ---------------------------------------------------------
    
    def load_truth_from_files(files):
        """Loads truth from a list of files. Returns dict: row -> set(cols)"""
        truth = {}
        for filepath in files:
            if not os.path.exists(filepath):
                print(f"Warning: Truth file not found: {filepath}")
                continue
            with open(filepath) as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue
                    r = int(parts[0])
                    c = parts[1]
                    if not c.endswith(".sd"):
                        c += ".sd"
                    
                    if r not in truth: truth[r] = set()
                    truth[r].add(c)
        return truth

    def verify_brwt(label, brwt_file, columns_file, expected_truth, num_rows):
        print(f"\n--- Verifying {label} ---")
        if not os.path.exists(brwt_file):
            print(f"File not found: {brwt_file}")
            return False
            
        batch_size = 50
        all_correct = True
        
        for start in range(0, num_rows, batch_size):
            end = min(start + batch_size, num_rows)
            batch_ids = list(range(start, end))
            batch_arg = "{" + ",".join(map(str, batch_ids)) + "}"
            
            cmd = f"{METAGRAPH_BIN} query {brwt_file} {columns_file} \"{batch_arg}\""
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = proc.communicate()
            
            if proc.returncode != 0:
                print(f"Query execution failed: {stderr}")
                return False
                
            # Parse output: "Row <id>: col1.sd col2.sd ..."
            row_data = {}
            for line in stdout.splitlines():
                line = line.strip()
                if line.startswith("Row "):
                    parts = line.split(":")
                    if len(parts) < 1: continue
                    try:
                        r_str = parts[0].split()[1]
                        r_id = int(r_str)
                        if len(parts) > 1:
                            cols = set(parts[1].strip().split())
                        else:
                            cols = set()
                        cols.discard("")
                        row_data[r_id] = cols
                    except IndexError:
                        pass
            
            # Check correctness
            for r_id in batch_ids:
                exp = expected_truth.get(r_id, set())
                got = row_data.get(r_id, set())
                
                if got != exp:
                    print(f"Mismatch at Row {r_id}!")
                    print(f"  Expected: {sorted(list(exp))}")
                    print(f"  Got:      {sorted(list(got))}")
                    all_correct = False
                    return False 

        if all_correct:
            print(f"SUCCESS: {label} is correct.")
            return True
        else:
            print(f"FAILURE: {label} has errors.")
            return False

    # Load truths
    truth_a = load_truth_from_files([f"{DATA_DIR}/a/_a_truth.txt"])
    truth_b = load_truth_from_files([f"{DATA_DIR}/b/_b_truth.txt"])
    truth_merged = load_truth_from_files([f"{DATA_DIR}/a/_a_truth.txt", f"{DATA_DIR}/b/_b_truth.txt"])

    # 1. Verify Tree A
    if not verify_brwt("Tree A", f"{DATA_DIR}/a_brwt.brwt", f"{DATA_DIR}/a_brwt.columns", truth_a, N):
        print("Tree A verification failed.")
        sys.exit(1)

    # 2. Verify Tree B
    if not verify_brwt("Tree B", f"{DATA_DIR}/b_brwt.brwt", f"{DATA_DIR}/b_brwt.columns", truth_b, N):
        print("Tree B verification failed.")
        sys.exit(1)

    # 3. Verify Merged (Current Merge)
    if not verify_brwt("Merged BRWT (Current)", f"{DATA_DIR}/merged.brwt", f"{DATA_DIR}/merged.columns", truth_merged, N):
        print("Merged BRWT verification failed.")
        sys.exit(1)

    print("\nALL TESTS PASSED.")

if __name__ == "__main__":
    main()
