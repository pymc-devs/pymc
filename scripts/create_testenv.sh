#!/usr/bin/env bash

set -ex # fail on first error, print commands

while test $# -gt 0; do
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

command -v conda >/dev/null 2>&1 || {
  echo "Requires conda but it is not installed.  Run install_miniconda.sh." >&2
  exit 1
}

ENVNAME="${ENVNAME:-testenv}"         # if no ENVNAME is specified, use testenv
PYTHON_VERSION=${PYTHON_VERSION:-3.6} # if no python specified, use 3.6

if [ -z ${GLOBAL} ]; then
  if conda env list | grep -q ${ENVNAME}; then
    echo "Environment ${ENVNAME} already exists, keeping up to date"
  else
    conda create -n ${ENVNAME} --yes pip python=${PYTHON_VERSION}
  fi
  source activate ${ENVNAME}
fi
pip install --upgrade pip

conda install --yes mkl-service
conda install --yes -c conda-forge python-graphviz

# Travis env is unable to import cached mpl sometimes https://github.com/pymc-devs/pymc3/issues/3423
pip install --no-cache-dir --force-reinstall -e .
pip install --no-cache-dir --force-reinstall -r requirements-dev.txt

# Install untested, non-required code (linter fails without them)
pip install ipython ipywidgets

#  Install editable using the setup.py
if [ -z ${NO_SETUP} ]; then
  python setup.py build_ext --inplace
fi
