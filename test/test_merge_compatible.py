import subprocess
import os
import sys
import shutil
import filecmp

METAGRAPH_BIN = "../build/metagraph"
DATA_DIR = "test_merge_data"

def run_command(cmd):
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)

def main():
    if len(sys.argv) > 1:
        global METAGRAPH_BIN
        METAGRAPH_BIN = sys.argv[1]

    # Try finding metagraph if not valid
    if not os.path.exists(METAGRAPH_BIN):
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

    # Ensure source data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory {DATA_DIR} not found. Please run test_merge.py first to generate data.")
        sys.exit(1)

    # 1. Back up original merged files if not already backed up
    orig_brwt = f"{DATA_DIR}/merged_original.brwt"
    orig_cols = f"{DATA_DIR}/merged_original.columns"
    
    current_brwt = f"{DATA_DIR}/merged.brwt"
    current_cols = f"{DATA_DIR}/merged.columns"

    if not os.path.exists(orig_brwt):
        if not os.path.exists(current_brwt):
            print(f"Error: original merged file {current_brwt} not found.")
            sys.exit(1)
        print("Backing up original merged.brwt to merged_original.brwt...")
        shutil.copyfile(current_brwt, orig_brwt)
        
    if not os.path.exists(orig_cols):
        if not os.path.exists(current_cols):
            print(f"Error: original columns file {current_cols} not found.")
            sys.exit(1)
        print("Backing up original merged.columns to merged_original.columns...")
        shutil.copyfile(current_cols, orig_cols)

    # 2. Re-build Tree A and Tree B using the specified (optimized) binary
    print("Building Tree A BRWT with new binary...")
    run_command(f"{METAGRAPH_BIN} build {DATA_DIR}/a a {DATA_DIR}/a_brwt --linkage_trivial")

    print("Building Tree B BRWT with new binary...")
    run_command(f"{METAGRAPH_BIN} build {DATA_DIR}/b b {DATA_DIR}/b_brwt --linkage_trivial")

    # 3. Merge using the specified (optimized) binary
    # This will overwrite merged.brwt and merged.columns
    print("Merging BRWT A and B with new binary...")
    run_command(f"{METAGRAPH_BIN} merge {DATA_DIR}/a_brwt.brwt {DATA_DIR}/a_brwt.columns {DATA_DIR}/b_brwt.brwt {DATA_DIR}/b_brwt.columns {DATA_DIR}/merged")

    # 4. Compare files
    print("\n--- Comparing Merged Files ---")
    
    brwt_match = filecmp.cmp(orig_brwt, current_brwt, shallow=False)
    
    import struct
    def parse_columns_file(filepath):
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, 'rb') as f:
                size_data = f.read(8)
                if not size_data:
                    return []
                size = struct.unpack('Q', size_data)[0]
                entries = []
                for _ in range(size):
                    col_id = struct.unpack('Q', f.read(8))[0]
                    str_len = struct.unpack('Q', f.read(8))[0]
                    name = f.read(str_len).decode('utf-8')
                    entries.append((col_id, name))
                return sorted(entries)
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            return None

    orig_cols_parsed = parse_columns_file(orig_cols)
    new_cols_parsed = parse_columns_file(current_cols)
    cols_match = (orig_cols_parsed == new_cols_parsed)

    print(f"merged.brwt match (byte-for-byte): {brwt_match}")
    print(f"merged.columns match (logical):    {cols_match}")

    # 5. Load truths and run query verification
    print("\n--- Verifying Row Query Correctness ---")
    
    def load_truth_from_files(files):
        truth = {}
        for filepath in files:
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

    # Determine N (rows) dynamically by loading truth file length or a default
    truth_merged = load_truth_from_files([f"{DATA_DIR}/a/_a_truth.txt", f"{DATA_DIR}/b/_b_truth.txt"])
    N = max(truth_merged.keys()) + 1

    def verify_brwt(label, brwt_file, columns_file, expected_truth, num_rows):
        batch_size = 50
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
            
            for r_id in batch_ids:
                exp = expected_truth.get(r_id, set())
                got = row_data.get(r_id, set())
                if got != exp:
                    print(f"Mismatch at Row {r_id}!")
                    print(f"  Expected: {sorted(list(exp))}")
                    print(f"  Got:      {sorted(list(got))}")
                    return False
        return True

    success = verify_brwt("Merged BRWT (New)", current_brwt, current_cols, truth_merged, N)
    if success:
        print("SUCCESS: Merged BRWT queries are 100% correct.")
    else:
        print("FAILURE: Mismatches found in queries.")
        sys.exit(1)

    if brwt_match and cols_match:
        print("\nCOMPATIBILITY TEST PASSED: Identical output files (and logical column matches) and correct queries.")
    else:
        if not brwt_match:
            print("\nWARNING: Binary files merged.brwt are NOT byte-for-byte identical.")
            print(f"Original size: {os.path.getsize(orig_brwt)}, New size: {os.path.getsize(current_brwt)}")
        if not cols_match:
            print("\nWARNING: merged.columns logically do NOT match.")
            print(f"Original columns size: {len(orig_cols_parsed) if orig_cols_parsed else 0}, New: {len(new_cols_parsed) if new_cols_parsed else 0}")
        print("Compatibility test status: DIFFERENCE DETECTED.")
        sys.exit(2)

if __name__ == "__main__":
    main()

