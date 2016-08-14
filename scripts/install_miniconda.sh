#!/usr/bin/env bash


set -e # fail on first error

${PYTHON_VERSION:=-3.5} # if no python specified, use 3

echo "using miniconda for python-${PYTHON_VERSION}"

if [ "$(uname)" == "Darwin" ]; then
  URL_OS="MacOSX"
elif [ "$(expr substr "$(uname -s)" 1 5)" == "Linux" ]; then
  URL_OS="Linux"
elif [ "$(expr substr "$(uname -s)" 1 10)" == "MINGW32_NT" ]; then
  URL_OS="Windows"
fi

echo "detected operating system: $URL_OS"

if [ ${PYTHON_VERSION} == "2.7" ]; then
  echo "Installing miniconda for python 2.7"
  wget http://repo.continuum.io/miniconda/Miniconda-latest-$URL_OS-x86_64.sh -O miniconda.sh;
  INSTALL_FOLDER="$HOME/minconda2"
else
  echo "Installing miniconda for python 3.5"
  wget http://repo.continuum.io/miniconda/Miniconda3-latest-$URL_OS-x86_64.sh -O miniconda.sh;
  INSTALL_FOLDER="$HOME/minconda3"
fi


# install miniconda to home folder
bash miniconda.sh -b -p $INSTALL_FOLDER
export PATH="$INSTALL_FOLDER/bin:$PATH"
conda update -q conda
