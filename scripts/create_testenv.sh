#!/usr/bin/env bash

set -e # fail on first error

PYTHON_VERSION=${PYTHON_VERSION:-3.4} # if no python specified, use 3.4

conda create -n testenv --yes pip python=${PYTHON_VERSION}

source activate testenv

conda install --yes jupyter pyzmq numpy scipy nose matplotlib pandas Cython patsy statsmodels joblib
if [ ${PYTHON_VERSION} == "2.7" ]; then
  conda install --yes mock enum34;
fi

pip install --no-deps numdifftools
pip install git+https://github.com/Theano/Theano.git
pip install git+https://github.com/mahmoudimus/nose-timer.git

python setup.py build_ext --inplace
