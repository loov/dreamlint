#!/usr/bin/env bash
# Regenerates index.scip with scip-python inside a Docker container.
# Run from any directory; the script resolves paths relative to itself.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE=dreamlint-python-scip:latest

docker build --platform=linux/amd64 -t "$IMAGE" "$HERE"

docker run --rm --platform=linux/amd64 \
    -v "$HERE":/project \
    "$IMAGE"

echo
echo "Wrote $HERE/index.scip ($(wc -c <"$HERE/index.scip") bytes)"
