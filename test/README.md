# BRWT Update Implementation Notes

## Overview
This document describes the implementation of the `update` command in Metagraph, which allows merging a new BRWT matrix into an existing one. This is useful for incremental updates where new experiments (columns) are added to an existing index.

## Constraints
*   **Old BRWT**: $N$ rows.
*   **New BRWT**: $N + M$ rows.
*   **Alignment**: The first $N$ rows of the new BRWT must correspond exactly to the rows of the old BRWT. The tail $M$ rows are new additions.

## Core Algorithm (`BRWT::update_merge`)

The implementation is located in `src/annotation/binary_matrix/multi_brwt/brwt.cpp`.

### 1. Row Extension (Padding)
The core challenge is that the `Old` BRWT has fewer rows than the `New` BRWT. The BRWT structure relies on `nonzero_rows_` (a bit vector) at each node to map rows to the ranks passed down to children.

*   **Observation**: If we append zeros to the `nonzero_rows_` vector of the root node, the *ranks* of the existing set bits (ones) do **not** change.
*   **Mechanism**:
    1.  We verify that `new.num_rows() >= old.num_rows()`.
    2.  We create a new bit vector of size $N+M$.
    3.  We copy the bits from the old $N$-sized vector into the start of the new one. The remaining $M$ bits are initialized to 0.
    4.  We replace the root `nonzero_rows_` of the `Old` BRWT with this new padded vector.
    5.  **Result**: The `Old` BRWT now technically has $N+M$ rows, but it is "empty" (all zeros) for the new $M$ rows. Its internal child structure remains completely valid and untouched.

### 2. Concatenation
Once the `Old` BRWT is padded to match the row dimensions of the `New` BRWT, the problem reduces to a standard column concatenation.

*   We use the existing `BRWTBottomUpBuilder::concatenate` function.
*   We create a list of submatrices: `[Old_Padded, New]`.
*   The builder merges them into a single BRWT structure.

## CLI Integration (`main.cpp`)

The functionality is exposed via the `update` command:
`metagraph update <old_brwt> <new_brwt> <output_brwt>`

### 1. BRWT Merge
*   Loads both old and new BRWT files.
*   Calls `old_brwt->update_merge(*new_brwt)`.
*   Serializes the result to disk (`.brwt`).

### 2. Metadata Update (`.columns` file)
The column names file maps internal column indices to string labels (e.g., experiment IDs). This must also be merged.

*   Loads `old.columns` and `new.columns`.
*   The `concatenate` operation appends the new columns *after* the old columns.
*   Therefore, the indices of the **New** columns shift by `old_num_columns`.
*   **Logic**:
    ```cpp
    for (auto& pair : new_columns) {
        pair.index += old_num_columns;
    }
    merged_columns = old_columns + new_columns;
    ```
*   Serializes the merged list to the output `.columns` file.

## Testing
The correctness is verified by `test/test_update.py`, which:
1.  Generates random bit vectors.
2.  Builds separate BRWTs.
3.  Runs the `update` command.
4.  Queries the resulting merged BRWT and compares the results against the ground truth sets of columns for every row.

## Compile `gen_sdsl_vectors`
```sh
g++ -std=c++17 -DNDEBUG -O3 \
    -I /home/p2-admin/include \
    -L /home/p2-admin/lib \
    gen_sdsl_vectors.cpp \
    -lsdsl -ldivsufsort -ldivsufsort64 \
    -o gen_sdsl_vectors

./gen_sdsl_vectors <num_rows> <num_vectors> <output_dir> <prefix> <sparsity>
```
