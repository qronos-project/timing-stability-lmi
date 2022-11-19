#!/bin/bash
# Common functionality for all shell scripts
set -e
cd "$(dirname "$0")"
test -f hyst/README.md || { echo "Cannot find the hyst submodule, try 'git submodule update --init --recursive'"; exit 1; }
test -f mpmath/README.rst || { echo "Cannot find the mpmath submodule, try 'git submodule update --init --recursive'"; exit 1; }
echo "GIT Revision: "
git describe --always --dirty
