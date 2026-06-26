"""
test_relax_compatible.py
=========================
Assumes test_relax.py has already been run (test_relax_data/ exists with
relaxed.brwt saved as the baseline).

This script:
  1. Backs up relaxed.brwt → relaxed_original.brwt  (once, on first run)
  2. Re-builds the BRWT from the SAME existing vectors using the new binary
  3. Re-relaxes with the new binary → relaxed.brwt
  4. Compares byte-for-byte with relaxed_original.brwt
  5. Verifies query correctness against original BRWT (source of truth)
"""
import subprocess
import os
import sys
import shutil
import filecmp

METAGRAPH_BIN = "../build/metagraph"
DATA_DIR      = "test_relax_data"
MAX_ARITY     = 4   # Must match what test_relax.py used

def run_command(cmd):
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)

def query_brwt(bin_path, brwt_file, cols_file, row_ids, batch_size=50):
    results = {}
    for start in range(0, len(row_ids), batch_size):
        batch = row_ids[start:start + batch_size]
        batch_arg = "{" + ",".join(map(str, batch)) + "}"
        cmd = f"{bin_path} query {brwt_file} {cols_file} \"{batch_arg}\""
        proc = subprocess.Popen(cmd, shell=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Query failed: {stderr}")
        for line in stdout.splitlines():
            line = line.strip()
            if not line.startswith("Row "): continue
            parts = line.split(":", 1)
            try:
                r_id = int(parts[0].split()[1])
                cols = set(parts[1].strip().split()) if len(parts) > 1 else set()
                cols.discard("")
                results[r_id] = cols
            except (IndexError, ValueError):
                pass
    return results

def main():
    if len(sys.argv) > 1:
        global METAGRAPH_BIN
        METAGRAPH_BIN = sys.argv[1]

    if not os.path.exists(METAGRAPH_BIN):
        for p in ["../build/metagraph", "../build_static/metagraph", "./build/metagraph", "metagraph"]:
            if os.path.exists(p):
                METAGRAPH_BIN = p
                break
        else:
            print(f"Error: metagraph binary not found at {METAGRAPH_BIN}.")
            sys.exit(1)

    print(f"Using metagraph binary at: {METAGRAPH_BIN}")

    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} not found. Please run test_relax.py first.")
        sys.exit(1)

    orig_relaxed    = f"{DATA_DIR}/relaxed_original.brwt"
    current_relaxed = f"{DATA_DIR}/relaxed.brwt"

    # 1. Backup baseline on first run
    if not os.path.exists(orig_relaxed):
        if not os.path.exists(current_relaxed):
            print(f"Error: {current_relaxed} not found. Run test_relax.py first.")
            sys.exit(1)
        print("Backing up relaxed.brwt → relaxed_original.brwt ...")
        shutil.copyfile(current_relaxed, orig_relaxed)

    # 2. Re-build BRWT from the SAME existing vectors with the new binary
    #    Use the same sorted file list for deterministic column ordering.
    print("\nRe-building BRWT with new binary...")
    file_list = f"{DATA_DIR}/col_filelist.txt"
    if not os.path.exists(file_list):
        sd_files = sorted([
            os.path.join(DATA_DIR, "cols", f)
            for f in os.listdir(f"{DATA_DIR}/cols")
            if f.endswith(".sd")
        ])
        with open(file_list, "w") as fl:
            for p in sd_files:
                fl.write(p + "\n")
    run_command(f"{METAGRAPH_BIN} build {DATA_DIR}/cols col {DATA_DIR}/original_brwt"
                f" --file_list {file_list}")

    # 3. Re-relax with the new binary
    print(f"\nRe-relaxing BRWT (max_arity={MAX_ARITY}) with new binary...")
    run_command(f"{METAGRAPH_BIN} relax {DATA_DIR}/original_brwt.brwt {MAX_ARITY}"
                f" {DATA_DIR}/relaxed.brwt")

    # 4. Byte-for-byte comparison
    print("\n--- Comparing Relaxed BRWT Files ---")
    brwt_match = filecmp.cmp(orig_relaxed, current_relaxed, shallow=False)
    print(f"relaxed.brwt match (byte-for-byte): {brwt_match}")
    if not brwt_match:
        print(f"  Baseline size: {os.path.getsize(orig_relaxed)}")
        print(f"  New size:      {os.path.getsize(current_relaxed)}")

    # 5. Query correctness: new relaxed.brwt must match original_brwt row-by-row
    print("\n--- Verifying Query Correctness (new relaxed vs original) ---")
    cols_file = f"{DATA_DIR}/original_brwt.columns"

    # Infer N from the columns file (check the total column count, not rows)
    # We query all rows that appear in any column — safest is to just query a range.
    # We derive N from the file count in the SD directory.
    sd_count = len([f for f in os.listdir(f"{DATA_DIR}/cols") if f.endswith(".sd")])
    # N is unknown here — we stored vectors with sparsity 0.02 over N=500 rows.
    # Use a conservative upper bound from the cols count, or store N in a file.
    # Better: load N from a metadata file if it exists.
    metadata_file = f"{DATA_DIR}/metadata.txt"
    if os.path.exists(metadata_file):
        with open(metadata_file) as mf:
            N = int(mf.read().strip())
    else:
        # Fallback: N hard-coded to match test_relax.py default
        N = 500
    all_rows = list(range(N))

    orig_results = query_brwt(METAGRAPH_BIN, f"{DATA_DIR}/original_brwt.brwt",
                               cols_file, all_rows)
    new_results  = query_brwt(METAGRAPH_BIN, current_relaxed, cols_file, all_rows)

    query_ok = True
    for r in all_rows:
        exp    = orig_results.get(r, set())
        result = new_results.get(r, set())
        if exp != result:
            print(f"Mismatch at Row {r}: expected={sorted(exp)}, got={sorted(result)}")
            query_ok = False
            break

    if query_ok:
        print("SUCCESS: New relaxed BRWT queries are 100% correct.")
    else:
        print("FAILURE: Query mismatches found.")
        sys.exit(1)

    # 6. Final verdict
    if brwt_match:
        print("\nCOMPATIBILITY TEST PASSED: Byte-for-byte identical and queries correct.")
    else:
        print("\nWARNING: relaxed.brwt differs from baseline (byte-for-byte).")
        print("Query results are still correct but the binary output changed.")
        print("Status: DIFFERENCE DETECTED (exit 2).")
        sys.exit(2)

if __name__ == "__main__":
    main()
