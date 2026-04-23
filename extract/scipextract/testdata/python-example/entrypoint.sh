#!/usr/bin/env bash
set -euo pipefail

cd /project

scip-python index --output=index.scip --project-name=example --project-version=0.1.0 .

ls -l /project/index.scip
