#!/bin/bash
set -e
git clean -xi
./run_experiments_docker.sh "$@"
rm output.tar.gz || true
tar --exclude=systems.pickle --exclude='*.pyc' --exclude=output.tar.gz --exclude-vcs -czvf output.tar.gz .
