#!/usr/bin/env bash

set -e

THEANO_FLAGS='gcc.cxxflags="-march=core2"' nosetests "$@"

if [[ "$RUN_PYLINT" == "true" ]]; then
    . ./scripts/lint.sh
fi
