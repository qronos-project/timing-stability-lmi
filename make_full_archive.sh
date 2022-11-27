#!/bin/bash
set -e
git clean -xi || echo "Warning: git clean failed, possibly because you are running from the output archive and not from a GIT repository. Please cleanup the previous output files yourself."
./run_experiments_docker.sh "$@"
rm output.tar.gz || true
tar --exclude=systems.pickle --exclude='*.pyc' --exclude=output.tar.gz --exclude-vcs -czvf output.tar.gz .
