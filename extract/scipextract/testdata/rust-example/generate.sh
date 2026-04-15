#!/usr/bin/env bash
# Regenerates index.scip with rust-analyzer inside a Docker container.
# Run from any directory; the script resolves paths relative to itself.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE=dreamlint-rust-scip:latest

docker build -t "$IMAGE" "$HERE"

rm -rf "$HERE/target" "$HERE/Cargo.lock"

docker run --rm \
    -v "$HERE":/project \
    "$IMAGE"

rm -rf "$HERE/target" "$HERE/Cargo.lock"

echo
echo "Wrote $HERE/index.scip ($(wc -c <"$HERE/index.scip") bytes)"
