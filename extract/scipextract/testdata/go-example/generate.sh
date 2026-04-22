#!/usr/bin/env bash
# Regenerates index.scip with scip-go inside a Docker container.
# Run from any directory; the script resolves paths relative to itself.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE=dreamlint-go-scip:latest

docker build -t "$IMAGE" "$HERE"

docker run --rm \
    -v "$HERE":/project \
    "$IMAGE"

echo
echo "Wrote $HERE/index.scip ($(wc -c <"$HERE/index.scip") bytes)"
