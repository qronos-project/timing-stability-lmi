#!/bin/sh

# Create a ZIP file including all submodules.
# Requirement: All submodules properly initialized (git submodule update --init --recursive)
set -e
cd "$(dirname "$0")"
./git-archive-all/git-archive-all.sh --format zip -- source-complete.zip
ls -l --si source-complete.zip
