#!/usr/bin/env bash
# Regenerates index.scip with scip-typescript inside a Docker container.
# Run from any directory; the script resolves paths relative to itself.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE=dreamlint-typescript-scip:latest

docker build -t "$IMAGE" "$HERE"

rm -rf "$HERE/node_modules" "$HERE/dist" "$HERE/package-lock.json"

docker run --rm \
    -v "$HERE":/project \
    "$IMAGE"

rm -rf "$HERE/node_modules" "$HERE/dist" "$HERE/package-lock.json"

echo
echo "Wrote $HERE/index.scip ($(wc -c <"$HERE/index.scip") bytes)"
