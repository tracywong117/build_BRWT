# Minimal MetaGraph BRWT

This repository provides a streamlined implementation of the **Binary Relation Wavelet Tree (BRWT)** data structure, derived from the original [MetaGraph](https://github.com/metagraph-dev/metagraph) project.

## Overview

Unlike the full MetaGraph suite, which supports various graph annotations and representations, this version is aggressively pruned to focus exclusively on:
- **BRWT Construction**: Building compressed binary matrices from annotation bit vectors.
- **Querying**: Efficiently retrieving column labels for given rows.
- **Relaxation**: Reducing the size of the BRWT by relaxing the constraint on the number of set bits (arity).
- **Serving**: Exposing the BRWT via a lightweight HTTP API.
- **Manipulation**: Concatenating and updating existing BRWT files.

## Build Instructions

### Prerequisites
- CMake >= 3.19
- GCC (supporting C++17) or Clang
- Zlib

### Compile
```bash
git clone --recursive https://github.com/tracywong117/build_BRWT.git
cd build_brwt
git submodule update --init --recursive
cd metagraph
cmake .
make -j$(nproc)
```
This will produce a single executable named `metagraph`.

## Usage

### 1. Build a BRWT
Construct a BRWT from a directory of column annotations.
```bash
./metagraph build <annotation_dir> <prefix> <output_file> [options]
```
- `<annotation_dir>`: Directory containing input files.
- `<prefix>`: Prefix of files to include.
- `<output_file>`: Path to write the output BRWT.

### 2. Query a BRWT
Find which columns contain specific rows.
```bash
./metagraph query <brwt_file> <columns_file> <row_ids>
```
- `<row_ids>`: A set of row IDs, e.g., `"{1, 2, 3}"`.

### 3. Relax a BRWT
Relax the BRWT to reduce its size by allowing nodes to have lower arity.
```bash
./metagraph relax <brwt_file> <max_arity> <output_file>
```

### 4. Serve
Start an HTTP server to query the BRWT.
```bash
./metagraph serve <brwt_file> <columns_file> <port>
```

### 5. Update / Concatenate
Merge two BRWTs.
```bash
./metagraph update <old_brwt> <new_brwt> <output_file>
./metagraph concat <brwt1> <brwt2> <output_file>
```

## License and Attribution

This project is a derivative work of **MetaGraph**.
- **Original Authors**: See `AUTHORS` file.
- **License**: This code is distributed under the same license as the original Metagraph project. See `LICENSE` and `COPYRIGHT` for details.
