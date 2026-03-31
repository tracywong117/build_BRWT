# Technical Specification: Optimized Multi-BRWT Construction

This implementation addresses the memory and performance bottlenecks of the original MetaGraph BRWT builder, specifically targeting datasets exceeding 200GB on machines with limited RAM (e.g., 64GB).

## 1. Memory Management (The "Streaming" Architecture)

The primary cause of failure in large-scale builds was the "RAM Wall" during the final tree assembly. The original implementation attempted to load all compressed nodes into RAM simultaneously to link pointers.

### Recursive Streaming Assembly
We replaced the in-memory assembly with a **Depth-First Streaming Assembler**.
*   **Shallow Node Persistence**: During the bottom-up merge, intermediate nodes are serialized to disk in a "shallow" format. Only the node's specific `RangePartition` and `nonzero_rows` (bit vector) are saved. Child pointers are omitted to prevent recursive data duplication.
*   **DFS Serialization**: The final `.brwt` file is written by traversing the linkage tree recursively. The assembler loads exactly one node's metadata from disk, writes it to the final stream, recurses into its children, and immediately clears the parent from RAM.
*   **Result**: RAM usage remains constant regardless of tree depth or total node count.

### Transformed Child Persistence
A critical fix was implemented to ensure bit-identity. In a BRWT, children are transformed (sub-indexed) relative to their parents during a merge.
*   **The Fix**: After each successful `concatenate` operation, the updated, sub-indexed children are immediately written back to the temporary directory, overwriting their previous state. This ensures the streaming assembler loads the transformed data rather than the raw inputs.

## 2. Speed Optimization (Hybrid OR Engine)

The bitwise OR operation is the "Hot Path" of the build process. We optimized `compute_or` to dynamically adapt to data density.

### Path A: Sparse Scatter (The "Two-Pointer" Logic)
*   **Target**: Sparse leaves and lower-level internal nodes (Density < 20%).
*   **Implementation**: Utilizes the SDSL `sd_vector` internal index to jump directly between set bits (`1`s).
*   **Efficiency**: Completely skips scanning the billions of zeros in the bitmap, performing a "scatter" operation into the result buffer.

### Path B: Dense Word-OR (SIMD Acceleration)
*   **Target**: Dense internal nodes at the top of the tree.
*   **Implementation**: Uses `dynamic_cast` to detect uncompressed `bit_vector_stat` objects and obtain a raw `uint64_t*` pointer.
*   **Efficiency**: Bypasses the virtual function overhead of `get_int()` (which originally cost 100 million calls per column). By accessing memory directly, the compiler can use **AVX2 instructions** to perform bitwise OR on 256 bits in a single CPU cycle.

## 3. Lazy Data Loading

To further reduce the RAM footprint during the initial leaf preparation phase, we implemented **Lazy SD-Vector Loading**.

*   **Mechanism**: Modified the loader callback to wrap the loaded `sdsl::sd_vector<>` directly into a `bit_vector_sd` object.
*   **Optimization**: This skips the redundant uncompression of input columns into raw bitmaps.
*   **RAM Savings**: For a column with 6.4 billion bits, this saves ~800MB of RAM per thread. In a multi-threaded build, this prevents OOM crashes before the first merge even begins.

## 4. Operational Visibility

To allow for predictable execution on long-running 200GB+ builds, real-time monitoring was integrated:
*   **Tree Depth Analysis**: The builder analyzes the linkage matrix before starting to calculate the total number of levels.
*   **Progress Logs**: Outputs `[STEP] Building Level X/Y` so the user can estimate completion time.
*   **RSS Monitoring**: Logs `[MEM] Process RSS: Z GB` at every level, providing early warning if the system is approaching physical RAM limits.

## 5. Verification & Accuracy

*   **Bit-Identity**: The optimized streamer is verified to produce a `.brwt` file that is 100% bit-identical to the output of the original MetaGraph implementation.
*   **Query Compatibility**: Because the optimizations target the **build process** rather than the **data format**, the resulting files are fully compatible with existing MetaGraph query engines without any code changes.
