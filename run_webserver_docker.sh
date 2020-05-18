#!/bin/bash
set -ex
echo "This script runs a IPython notebook webserver."
echo "Follow the messages on how to start it"
echo "Run this script with '--no-cache' if something fails"


cd "$(dirname "$0")"
. common.sh

# Run container as current user, except if CONTAINER_UID is given.
export CONTAINER_UID=${CONTAINER_UID:-$UID}

docker-compose build $1
docker-compose run --service-ports web
