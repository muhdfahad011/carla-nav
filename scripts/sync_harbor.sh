#!/bin/bash
# Copy the package into the Harbor task build context.
set -e
cd "$(dirname "$0")/.."
rm -rf harbor_task/environment/rlnav
cp -r rlnav harbor_task/environment/rlnav
cp pyproject.toml harbor_task/environment/pyproject.toml
echo "Synced rlnav into harbor_task/environment"
