#!/bin/sh
set -ex
echo "This script runs a IPython notebook webserver."
echo "Follow the messages on how to start it"
echo "Run this script with '--no-cache' if something fails"
docker-compose build $1
docker-compose run --service-ports web
