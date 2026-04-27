import subprocess
import os
import sys
import time
import shutil
import random

# Configuration
METAGRAPH_BIN = "../build_static/metagraph"
# METAGRAPH_BIN = "../build_original_optimized/metagraph"
GEN_VECTORS_BIN = "./gen_sdsl_vectors"
TMP_DIR = "/media/data/tracy/build_BRWT/test/tmp"
DATA_DIR = "benchmark_data"
OUTPUT_FILE = "bench_new.brwt"

# Optional cross-check: verify `query-nodes` returns the same rows as `query`.
VERIFY_QUERY_NODES = True

# Benchmark Parameters
# Increased size to make I/O and bitwise math optimizations visible.
NUM_ROWS = 10_000_000  # 10M rows 
NUM_COLS = 1000        # 1000 columns
SPARSITY = 0.01        # 1% density
THREADS = os.cpu_count() or 1


def parse_query_output(output):
    """Parse all `Row <id>: col1 col2 ...` lines into {row_id: set(columns)}."""
    result = {}
    for line in output.splitlines():
        if not line.startswith("Row "):
            continue
        colon_idx = line.index(":")
        rid = int(line[4:colon_idx])
        cols_part = line[colon_idx + 1:].strip()
        result[rid] = set(cols_part.split()) if cols_part else set()
    return result

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
    truth_file = os.path.join(DATA_DIR, "_col_truth.txt")
    if not os.path.exists(truth_file):
        print("Truth file not found, skipping verification.")
        return

    print("\n--- Verifying Correctness (comprehensive) ---")

    linkage_file = os.path.join(DATA_DIR, f"{OUTPUT_FILE}.linkage")
    columns_file = os.path.join(DATA_DIR, f"{OUTPUT_FILE}.columns")
    brwt_file = os.path.join(DATA_DIR, f"{OUTPUT_FILE}.brwt")

    do_query_nodes_check = (
        VERIFY_QUERY_NODES
        and os.path.exists(linkage_file)
        and os.path.exists(columns_file)
        and os.path.isdir(TMP_DIR)
    )
    if VERIFY_QUERY_NODES and not do_query_nodes_check:
        print("  [WARN] Skipping query-nodes verification (missing linkage/columns/tmp-dir).")

    # Phase 1: Build a complete truth index from the truth file.
    # truth_index: row_id -> set of column filenames
    print("  Loading truth file (this may take a moment)...")
    truth_index = {}   # row_id -> set(col_name.sd, ...)
    all_row_ids = set()
    load_start = time.time()
    with open(truth_file, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) != 2:
                continue
            row_id = int(parts[0])
            col_name = parts[1] + ".sd"
            all_row_ids.add(row_id)
            if row_id not in truth_index:
                truth_index[row_id] = set()
            truth_index[row_id].add(col_name)
    load_elapsed = time.time() - load_start
    print(f"  Loaded {len(truth_index):,} rows with set bits from truth file in {load_elapsed:.1f}s")

    # Phase 2: Select a diverse set of rows to verify.
    NUM_VERIFY = 100
    sorted_row_ids = sorted(truth_index.keys())

    selected_rows = set()

    # (a) Boundary rows: first and last rows that have any bits set
    selected_rows.add(sorted_row_ids[0])
    selected_rows.add(sorted_row_ids[-1])

    # (b) Rows with the most columns set (stress-test dense rows)
    by_count = sorted(truth_index.items(), key=lambda x: len(x[1]), reverse=True)
    for row_id, _ in by_count[:5]:
        selected_rows.add(row_id)

    # (c) Rows with the fewest columns set (sparse rows, often just 1 column)
    by_count_asc = sorted(truth_index.items(), key=lambda x: len(x[1]))
    for row_id, _ in by_count_asc[:5]:
        selected_rows.add(row_id)

    # (d) Rows with zero columns set (should return empty)
    #     Pick some row IDs that do NOT appear in the truth file
    zero_rows_added = 0
    for candidate in range(NUM_ROWS):
        if candidate not in all_row_ids:
            selected_rows.add(candidate)
            zero_rows_added += 1
            if zero_rows_added >= 5:
                break

    # (e) Uniformly spaced rows across the entire range
    step = max(1, len(sorted_row_ids) // 20)
    for i in range(0, len(sorted_row_ids), step):
        selected_rows.add(sorted_row_ids[i])

    # (f) Random sample to fill up to NUM_VERIFY
    remaining = NUM_VERIFY - len(selected_rows)
    if remaining > 0:
        pool = [r for r in sorted_row_ids if r not in selected_rows]
        if len(pool) > remaining:
            selected_rows.update(random.sample(pool, remaining))
        else:
            selected_rows.update(pool)

    selected_rows = sorted(selected_rows)
    print(f"  Verifying {len(selected_rows)} rows "
          f"(boundary, densest, sparsest, zero-column, evenly-spaced, random)...")

    # Phase 3: Query in batches and compare.
    BATCH_SIZE = 50  # rows per query invocation
    passed = 0
    failed = 0
    query_nodes_mismatches = 0
    query_nodes_checks = 0
    errors = []

    for batch_start in range(0, len(selected_rows), BATCH_SIZE):
        batch = selected_rows[batch_start:batch_start + BATCH_SIZE]
        ids_str = ",".join(str(r) for r in batch)
        query_cmd = (
            f"{METAGRAPH_BIN} query "
            f"{brwt_file} "
            f"{columns_file} "
            f"'{{{ids_str}}}'"
        )
        try:
            output = subprocess.check_output(query_cmd, shell=True).decode()
        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] Query command failed: {e}")
            sys.exit(1)

        actual_map = parse_query_output(output)

        if do_query_nodes_check:
            query_nodes_cmd = (
                f"{METAGRAPH_BIN} query-nodes "
                f"{linkage_file} "
                f"{TMP_DIR} "
                f"{columns_file} "
                f"'{{{ids_str}}}'"
            )
            try:
                nodes_output = subprocess.check_output(query_nodes_cmd, shell=True).decode()
            except subprocess.CalledProcessError as e:
                print(f"  [ERROR] query-nodes command failed: {e}")
                sys.exit(1)

            nodes_map = parse_query_output(nodes_output)
            for row_id in batch:
                query_nodes_checks += 1
                query_cols = actual_map.get(row_id, set())
                node_cols = nodes_map.get(row_id, set())
                if query_cols != node_cols:
                    query_nodes_mismatches += 1
                    missing = query_cols - node_cols
                    extra = node_cols - query_cols
                    detail = (
                        f"Row {row_id}: query-nodes mismatch vs query "
                        f"(query={len(query_cols)}, query-nodes={len(node_cols)})"
                    )
                    if missing:
                        detail += f"; missing-from-query-nodes {len(missing)}: {sorted(missing)[:5]}"
                    if extra:
                        detail += f"; extra-in-query-nodes {len(extra)}: {sorted(extra)[:5]}"
                    errors.append(detail)
                    if len(errors) <= 10:
                        print(f"    [FAIL] {detail}")
                

        # Compare each row in this batch
        for row_id in batch:
            expected = truth_index.get(row_id, set())
            actual = actual_map.get(row_id, set())

            if actual == expected:
                passed += 1
            else:
                failed += 1
                missing = expected - actual
                extra = actual - expected
                detail = f"Row {row_id}: expected {len(expected)} cols, got {len(actual)} cols"
                if missing:
                    detail += f"; missing {len(missing)}: {sorted(missing)[:5]}"
                if extra:
                    detail += f"; extra {len(extra)}: {sorted(extra)[:5]}"
                errors.append(detail)
                if len(errors) <= 10:
                    print(f"    [FAIL] {detail}")

    # Phase 4: Summary
    print(f"\n  Verification Summary: {passed}/{passed + failed} rows passed")
    if do_query_nodes_check:
        print(
            f"  query-nodes parity: "
            f"{query_nodes_checks - query_nodes_mismatches}/{query_nodes_checks} rows matched query"
        )

    total_failures = failed + query_nodes_mismatches
    if total_failures:
        if len(errors) > 10:
            print(f"  (showing first 10 of {len(errors)} failures)")
        print("  CORRECTNESS CHECK FAILED!")
        sys.exit(1)
    else:
        print("  All rows verified successfully. ✓")

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
