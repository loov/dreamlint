#!/usr/bin/env bash
set -euo pipefail

# Configure with CMake to produce compile_commands.json.
cmake -S /project -B /project/build \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    >/dev/null

# Run scip-clang over the compilation database.
cd /project
scip-clang --compdb-path=build/compile_commands.json \
    --index-output-path=index.scip

ls -l /project/index.scip
