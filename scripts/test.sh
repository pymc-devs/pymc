#!/usr/bin/env bash

set -e

if [[ "$RUN_PYLINT" == "true" ]]; then
    . ./scripts/lint.sh
fi

_FLOATX=${FLOATX:=float64}
THEANO_FLAGS="floatX=${_FLOATX},gcc.cxxflags='-march=core2'" pytest -v --cov=pymc3 --cov-report=xml "$@" --cov-report term
