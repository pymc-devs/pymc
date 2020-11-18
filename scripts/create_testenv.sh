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

if [ -z ${GLOBAL} ]; then
  source $(dirname $(dirname $(which conda)))/etc/profile.d/conda.sh
  if conda env list | grep -q ${ENVNAME}; then
    echo "Environment ${ENVNAME} already exists, keeping up to date"
    conda activate ${ENVNAME}
    mamba env update -f environment-dev.yml
  else
    conda config --add channels conda-forge
    conda config --set channel_priority strict
    conda install -c conda-forge mamba --yes
    mamba env create -f environment-dev.yml
    conda activate ${ENVNAME}
  fi
fi

#  Install editable using the setup.py
if [ -z ${NO_SETUP} ]; then
  python setup.py build_ext --inplace
fi
