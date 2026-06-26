import subprocess
import os
import sys
import shutil

METAGRAPH_BIN = "../build/metagraph"
GEN_VECTORS_BIN = "./gen_sdsl_vectors"

def run_command(cmd):
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)

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

    if not os.path.exists(GEN_VECTORS_BIN):
        print("Compiling gen_sdsl_vectors...")
        run_command("bash compile_test.sh")

    # Parameters
    N        = 500        # Row count
    COLS     = 200        # Number of columns
    SPARSITY = 0.02
    MAX_ARITY = 4         # relax max arity

    DATA_DIR = "test_relax_data"

    # -------------------------------------------------------------------
    # Fresh setup: always regenerate data so we own the ground truth
    # -------------------------------------------------------------------
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR)
    os.makedirs(f"{DATA_DIR}/cols")

    # Persist N so test_relax_compatible.py can read it
    with open(f"{DATA_DIR}/metadata.txt", "w") as mf:
        mf.write(str(N))

    # 1. Generate vectors
    print("Generating vectors...")
    run_command(f"{GEN_VECTORS_BIN} {N} {COLS} {DATA_DIR}/cols col {SPARSITY}")

    # 2. Build BRWT with a sorted file list (deterministic column ordering)
    print("Building BRWT...")
    file_list = f"{DATA_DIR}/col_filelist.txt"
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

    # 3. Relax the BRWT
    print(f"Relaxing BRWT (max_arity={MAX_ARITY})...")
    run_command(f"{METAGRAPH_BIN} relax {DATA_DIR}/original_brwt.brwt {MAX_ARITY}"
                f" {DATA_DIR}/relaxed.brwt")

    # -------------------------------------------------------------------
    # Size Report
    # -------------------------------------------------------------------
    def get_size(path):
        return os.path.getsize(path) if os.path.exists(path) else -1

    print("\n--- File Sizes (Bytes) ---")
    print(f"Original BRWT: {get_size(f'{DATA_DIR}/original_brwt.brwt')}")
    print(f"Relaxed  BRWT: {get_size(f'{DATA_DIR}/relaxed.brwt')}")

    # -------------------------------------------------------------------
    # Verification strategy:
    #   - Query every row from the ORIGINAL BRWT to get ground truth.
    #   - Query the RELAXED BRWT and compare: results must be identical.
    #   (relax only restructures the tree; query answers must not change)
    # -------------------------------------------------------------------

    def query_brwt(brwt_file, cols_file, row_ids, batch_size=50):
        """Returns dict: row_id -> set(column_names)"""
        results = {}
        for start in range(0, len(row_ids), batch_size):
            batch = row_ids[start:start + batch_size]
            batch_arg = "{" + ",".join(map(str, batch)) + "}"
            cmd = f"{METAGRAPH_BIN} query {brwt_file} {cols_file} \"{batch_arg}\""
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

    all_rows = list(range(N))
    cols_file = f"{DATA_DIR}/original_brwt.columns"

    print("\n--- Querying Original BRWT (ground truth) ---")
    original_results = query_brwt(f"{DATA_DIR}/original_brwt.brwt", cols_file, all_rows)
    print(f"Got results for {len(original_results)} rows.")

    print("\n--- Verifying Relaxed BRWT matches Original ---")
    relaxed_results = query_brwt(f"{DATA_DIR}/relaxed.brwt", cols_file, all_rows)
    ok = True
    for r in all_rows:
        exp = original_results.get(r, set())
        got = relaxed_results.get(r, set())
        if exp != got:
            print(f"Mismatch at Row {r}: expected={sorted(exp)}, got={sorted(got)}")
            ok = False
            break
    if ok:
        print("SUCCESS: Relaxed BRWT query results are identical to original.")
    else:
        sys.exit(1)

    print("\nALL TESTS PASSED.")
    print(f"\nIMPORTANT: relaxed.brwt saved at {DATA_DIR}/relaxed.brwt")
    print("Run test_relax_compatible.py BEFORE rebuilding to compare against it.")

if __name__ == "__main__":
    main()
