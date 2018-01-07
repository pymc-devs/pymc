#!/usr/bin/env bash

if [[ "$BUILD_DOCS" == "true" ]]; then
    travis-sphinx build -n -s docs/source
fi

if [[ "$RUN_PYLINT" == "true" ]]; then
    . ./scripts/lint.sh
fi

theano-cache purge
THEANO_FLAGS="floatX=${FLOATX},gcc.cxxflags='-march=core2'" pytest -v --cov=pymc3 "$@"
