#!/bin/bash
set -ex
# Runs unittests and creates a test coverage report in src/python_coverage.
# Start this script with '--no-cache' if something fails.


cd "$(dirname "$0")"
. common.sh

# Run container as current user, except if CONTAINER_UID is given.
export CONTAINER_UID=${CONTAINER_UID:-$UID}

docker-compose build $1
docker-compose run web /src/python_test_coverage.sh
