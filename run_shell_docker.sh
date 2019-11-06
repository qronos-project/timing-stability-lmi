#!/bin/sh
set -ex
echo "This script runs a shell in a docker container containing all necessary dependencies."
echo "The ./src directory is available in the container at /src."
echo "Run with '--no-cache' if something fails"
echo "Try: python3 run.py"
docker-compose build $1
docker-compose run web bash
