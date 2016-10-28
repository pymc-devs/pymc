#!/usr/bin/env bash

set -e # fail on first error

PYTHON_VERSION=${PYTHON_VERSION:-3.5} # if no python specified, use 3.5

if [[ "$*" != *--global* ]]
then
  conda create -n testenv --yes pip python=${PYTHON_VERSION}
  source activate testenv
fi

conda install --yes pyqt=4.11.4 jupyter pyzmq numpy scipy nose matplotlib pandas Cython patsy statsmodels joblib coverage
if [ ${PYTHON_VERSION} == "2.7" ]; then
  conda install --yes mock enum34;
fi

pip install tqdm
pip install --no-deps numdifftools
pip install git+https://github.com/Theano/Theano.git
pip install git+https://github.com/mahmoudimus/nose-timer.git

python setup.py build_ext --inplace
