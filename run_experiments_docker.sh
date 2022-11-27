#!/bin/bash
set -e
set -o pipefail
IFS=$'\n\t'

test -f hyst/Dockerfile || { echo "The hyst submodule is missing, or you are not in the right directory"; exit 1; }

cd "$(dirname "$0")"
. common.sh

echo "This script runs the experiments and prints the log and the LaTeX table of results."
echo "All output is also saved to ./logfile.txt"
echo "Run this script with '--no-cache' if something fails"


# Run container as current user, except if CONTAINER_UID is given.
export CONTAINER_UID=${CONTAINER_UID:-$UID}

if [ "$1" = "--no-cache" ]; then
	docker-compose build --no-cache
	echo "Please restart the script without --no-cache to actually run the experiments."
	exit
fi
docker-compose build
# -T: prevent artifacts (special terminal characters) in the logfile
# python -u: unbuffered stdin/stdout to avoid print delays on the console
docker-compose run -T web python3 -u -m qronos.run_experiments $@ 2>&1 | tee ./logfile.txt
echo
echo done.
