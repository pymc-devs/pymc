#!/usr/bin/env bash

set -e

THEANO_FLAGS="floatX=${FLOATX},gcc.cxxflags='-march=core2'" pytest -v --cov=pymc3 "$@"

if [[ "$RUN_PYLINT" == "true" ]]; then
    . ./scripts/lint.sh
fi
