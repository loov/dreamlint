#!/usr/bin/env bash
set -euo pipefail

cd /project

scip-go --output=index.scip

ls -l /project/index.scip
