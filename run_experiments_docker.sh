#!/bin/sh
set -ex
echo "This script runs the experiments and prints the log and the LaTeX table of results."
echo "All output is also saved to ./logfile.txt"
echo "Run this script with '--no-cache' if something fails"
docker-compose build $1
docker-compose run web python3 -m qronos.lis.analyze 2>&1 | tee ./logfile.txt
