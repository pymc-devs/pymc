#!/usr/bin/env bash

${PYTHON_VERSION:=-3.5} # if no python specified, use 3

if [ ${PYTHON_VERSION} == "2.7" ]; then
  wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh;
else
  wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
fi

if [ ${PYTHON_VERSION} == "2.7" ]; then
  INSTALL_FOLDER="$HOME/minconda2"
else
  INSTALL_FOLDER="$HOME/minconda3"
fi

# install miniconda to home folder
bash miniconda.sh -b -p $INSTALL_FOLDER
export PATH="$INSTALL_FOLDER/bin:$PATH"
conda update -q conda
