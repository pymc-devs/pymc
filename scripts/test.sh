#!/usr/bin/env bash

set -e

if [[ "$BUILD_DOCS" == "true" ]]; then
    travis-sphinx -n build

    if [[ "${TRAVIS_PULL_REQUEST}" = "false" ]]; then
        travis-sphinx deploy;
    fi
fi

if [[ "$RUN_PYLINT" == "true" ]]; then
    . ./scripts/lint.sh
fi

THEANO_FLAGS="floatX=${FLOATX},gcc.cxxflags='-march=core2'" pytest -v --cov=pymc3 "$@"

