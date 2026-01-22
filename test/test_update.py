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
        print(f"Error: metagraph binary not found at {METAGRAPH_BIN}.")
        sys.exit(1) # Let it crash if user doesn't care

    # Compile gen_sdsl_vectors if needed
    if not os.path.exists(GEN_VECTORS_BIN):
        print("Compiling gen_sdsl_vectors...")
        run_command("bash compile_test.sh")

    # Parameters
    N = 100 # Old rows
    M = 100  # New rows added
    OLD_COLS = 100
    NEW_COLS = 100
    SPARSITY = 0.01
    
    DATA_DIR = "test_data"
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR)
    os.makedirs(f"{DATA_DIR}/old")
    os.makedirs(f"{DATA_DIR}/new")

    # 1. Generate Old Vectors (N rows)
    print("Generating old vectors...")
    run_command(f"{GEN_VECTORS_BIN} {N} {OLD_COLS} {DATA_DIR}/old old {SPARSITY}")
    
    # 2. Generate New Vectors (N + M rows)
    print("Generating new vectors...")
    # NOTE: These new vectors represent NEW experiments. 
    # But they cover the full range of rows (N + M). 
    # The first N rows correspond to old rows.
    run_command(f"{GEN_VECTORS_BIN} {N + M} {NEW_COLS} {DATA_DIR}/new new {SPARSITY}")

    # 3. Build Old BRWT
    print("Building Old BRWT...")
    run_command(f"{METAGRAPH_BIN} build {DATA_DIR}/old old {DATA_DIR}/old_brwt --linkage_trivial")

    # 4. Build New BRWT
    print("Building New BRWT...")
    run_command(f"{METAGRAPH_BIN} build {DATA_DIR}/new new {DATA_DIR}/new_brwt --linkage_trivial")

    # 5. Merge
    print("Merging BRWTs...")
    run_command(f"{METAGRAPH_BIN} update {DATA_DIR}/old_brwt.brwt {DATA_DIR}/new_brwt.brwt {DATA_DIR}/merged.brwt")

    # 6. Relax (Updated BRWT)
    print("Relaxing Merged BRWT...")
    run_command(f"{METAGRAPH_BIN} relax {DATA_DIR}/merged.brwt 7 {DATA_DIR}/relaxed_merged.brwt") # MaxArity=2

    # ---------------------------------------------------------
    # Size Report
    # ---------------------------------------------------------
    def get_size(path):
        if os.path.exists(path):
            return os.path.getsize(path)
        return -1

    print("\n--- File Sizes (Bytes) ---")
    print(f"Old BRWT:           {get_size(f'{DATA_DIR}/old_brwt.brwt')}")
    print(f"New BRWT:           {get_size(f'{DATA_DIR}/new_brwt.brwt')}")
    print(f"Updated BRWT:       {get_size(f'{DATA_DIR}/merged.brwt')}")
    print(f"Relaxed Updated:    {get_size(f'{DATA_DIR}/relaxed_merged.brwt')}")
    
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
                    # Append .sd extension if missing, to match metagraph query output
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
            
        batch_size = 100  # Larger batch size
        all_correct = True
        
        # We only query rows present in expected_rows (optimization) or all rows?
        # Let's query all rows up to num_rows to ensure no false positives
        
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
                
            # Parse output
            # Expected format: "Row <id>: col1.sd col2.sd ..."
            row_data = {}
            for line in stdout.splitlines():
                line = line.strip()
                if line.startswith("Row "):
                    # print(line)
                    parts = line.split(":")
                    if len(parts) < 1: continue
                    try:
                        r_str = parts[0].split()[1]
                        r_id = int(r_str)
                        if len(parts) > 1:
                            cols = set(parts[1].strip().split())
                        else:
                            cols = set()
                        # remove empty strings if any
                        cols.discard("")
                        row_data[r_id] = cols
                    except IndexError:
                        pass
            
            # Check correctness
            for r_id in batch_ids:
                exp = expected_truth.get(r_id, set())
                got = row_data.get(r_id, set())
                # if got == exp:
                #     print(f" Row {r_id} correct.")
                #     print(f" Expected: {sorted(list(exp))}")
                #     print(f" Got:      {sorted(list(got))}")
                
                if got != exp:
                    print(f"Mismatch at Row {r_id}!")
                    print(f"  Expected: {sorted(list(exp))}")
                    print(f"  Got:      {sorted(list(got))}")
                    all_correct = False
                    # Don't spam: return immediately on failure usually better for debugging, 
                    # but maybe show a few?
                    return False 

        if all_correct:
            print(f"SUCCESS: {label} is correct.")
            return True
        else:
            print(f"FAILURE: {label} has errors.")
            return False

    # Load truths
    old_truth_file = f"{DATA_DIR}/old/_old_truth.txt"
    new_truth_file = f"{DATA_DIR}/new/_new_truth.txt"
    
    truth_old_only = load_truth_from_files([old_truth_file])
    truth_new_only = load_truth_from_files([new_truth_file])
    truth_merged = load_truth_from_files([old_truth_file, new_truth_file])

    # 1. Verify Old
    if not verify_brwt("Old BRWT", f"{DATA_DIR}/old_brwt.brwt", f"{DATA_DIR}/old_brwt.columns", truth_old_only, N):
        print("Skipping further verification steps due to failure.")
        sys.exit(1)

    # 2. Verify New
    if not verify_brwt("New BRWT", f"{DATA_DIR}/new_brwt.brwt", f"{DATA_DIR}/new_brwt.columns", truth_new_only, N + M):
         print("Skipping further verification steps due to failure.")
         sys.exit(1)

    # 3. Verify Merged
    if not verify_brwt("Merged BRWT", f"{DATA_DIR}/merged.brwt", f"{DATA_DIR}/merged.columns", truth_merged, N + M):
         sys.exit(1)

    print("\nALL TESTS PASSED.")

if __name__ == "__main__":
    main()
