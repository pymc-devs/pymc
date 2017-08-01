#!/usr/bin/env bash

set -e

if [[ "$RUN_PYLINT" == "true" ]]; then
    . ./scripts/lint.sh
fi

THEANO_FLAGS="floatX=${FLOATX},gcc.cxxflags='-march=core2'" pytest -v --cov=pymc3 "$@"

