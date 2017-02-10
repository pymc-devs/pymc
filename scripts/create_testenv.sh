#!/usr/bin/env bash

set -ex # fail on first error, print commands

while test $# -gt 0
do
    case "$1" in
        --global)
            GLOBAL=1
            ;;
        --no-setup)
            NO_SETUP=1
            ;;
    esac
    shift
done

PYTHON_VERSION=${PYTHON_VERSION:-3.6} # if no python specified, use 3.6

if [ -z ${GLOBAL} ]
then
    conda create -n testenv --yes pip python=${PYTHON_VERSION}
    source activate testenv
fi

pip install jupyter
conda install --yes pyqt matplotlib --channel conda-forge
conda install --yes pyzmq numpy scipy nose pandas Cython patsy statsmodels joblib coverage mkl-service
if [ ${PYTHON_VERSION} == "2.7" ]; then
    conda install --yes mock enum34;
fi

pip install --upgrade pip
pip install tqdm
pip install nose_parameterized
pip install --no-deps numdifftools
pip install git+https://github.com/Theano/Theano.git
pip install git+https://github.com/mahmoudimus/nose-timer.git

if [ -z ${NO_SETUP} ]
then
    python setup.py build_ext --inplace
fi
