#!/bin/sh
set -ex
# run this script with '--no-cache' if something fails
docker-compose build $1
docker-compose run web python3 -m unittest discover
