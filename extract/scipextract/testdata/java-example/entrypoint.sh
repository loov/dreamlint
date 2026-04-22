#!/usr/bin/env bash
set -euo pipefail

cd /project

scip-java index --output=index.scip

ls -l /project/index.scip
