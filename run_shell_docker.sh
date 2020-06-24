#!/bin/bash
set -ex
echo "This script runs a shell in a docker container containing all necessary dependencies."
echo "The ./src directory is available in the container at /src."
echo "Run with '--no-cache' if something fails"
echo "Try: python3 -m qronos.run_experiments"

cd "$(dirname "$0")"
. common.sh

# Run container as current user, except if CONTAINER_UID is given.
export CONTAINER_UID=${CONTAINER_UID:-$UID}

docker-compose build $1
docker-compose run web bash
