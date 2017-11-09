#!/usr/bin/env bash

set -e

if [[ "$BUILD_DOCS" == "true" ]]; then
    . ./scripts/lint.sh
fi

if [[ "$RUN_PYLINT" == "true" ]]; then
    travis-sphinx -n build
fi

THEANO_FLAGS="floatX=${FLOATX},gcc.cxxflags='-march=core2'" pytest -v --cov=pymc3 "$@"

