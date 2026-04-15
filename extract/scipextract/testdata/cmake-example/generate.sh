#!/usr/bin/env bash
# Regenerates index.scip using a Docker container with cmake + scip-clang.
# Run from any directory; the script resolves paths relative to itself.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE=dreamlint-cmake-scip:latest

docker build --platform=linux/amd64 -t "$IMAGE" "$HERE"

# Clean stale build/ before running so CMake reconfigures against a fresh
# tree and compile_commands.json points at the mounted paths.
rm -rf "$HERE/build"

docker run --rm --platform=linux/amd64 \
    -v "$HERE":/project \
    "$IMAGE"

rm -rf "$HERE/build"

echo
echo "Wrote $HERE/index.scip ($(wc -c <"$HERE/index.scip") bytes)"
