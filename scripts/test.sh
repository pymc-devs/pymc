#!/usr/bin/env bash

set -e

if [[ "$BUILD_DOCS" == "true" ]]; then
    travis-sphinx -n build
    travis-sphinx -c "docs.pymc.io" deploy
fi

if [[ "$RUN_PYLINT" == "true" ]]; then
    . ./scripts/lint.sh
fi

THEANO_FLAGS="floatX=${FLOATX},gcc.cxxflags='-march=core2'" pytest -v --cov=pymc3 "$@"

