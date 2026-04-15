#!/usr/bin/env bash
set -euo pipefail

cd /project

# Install dev dependencies (typescript + scip-typescript) locally. Silent
# install; progress output would clutter the generate.sh log.
npm install --silent --no-audit --no-fund

npx --yes scip-typescript index

ls -l /project/index.scip
