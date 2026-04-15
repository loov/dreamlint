#!/usr/bin/env bash
set -euo pipefail

cd /project

# Fetch dependencies and populate the build graph that rust-analyzer needs.
cargo check --offline >/dev/null 2>&1 || cargo check >/dev/null

rust-analyzer scip .

ls -l /project/index.scip
