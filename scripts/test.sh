#!/usr/bin/env bash

set -e

_FLOATX=${FLOATX:=float64}
PYTENSOR_FLAGS="floatX=${_FLOATX}" pytest -v --cov=pymc --cov-report=xml "$@" --cov-report term
