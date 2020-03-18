#!/bin/bash
set -e
cd "$(dirname $0)"
COVERAGE=python3-coverage
# note: some configuration is read from .coveragerc
COVERAGE_RUN="$COVERAGE run"
export PYTHON_COVERAGE=1 # switch on coverage measurement workarounds in the python scripts
ls -ld .
mkdir -p python_coverage
rm -r python_coverage/
$COVERAGE erase
# $COVERAGE_RUN some_manual_script.py
$COVERAGE_RUN -m unittest discover -v
$COVERAGE combine
$COVERAGE report
$COVERAGE html
echo "A test coverage report can be found in $(pwd)/python_coverage/index.html"
