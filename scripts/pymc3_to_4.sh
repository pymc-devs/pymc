#!/bin/bash

if [ -z "$1" ]
  then
    echo "Script to automatically convert code (*.py and *.ipynb) from PyMC3 to 4.0. Use with care."
    echo "Usage: pymc3_to_4.sh <path>"
    exit 1
fi

declare -a replace_strings=(
    "s/az.from_pymc3/pm.to_inference_data/g"
    "s/arviz.from_pymc3/pm.to_inference_data/g"
    "s/pymc3/pymc/g"
    "s/PyMC3/PyMC/g"
    "s/from theano import tensor as tt/import aesara.tensor as at/g"
    "s/import theano\.tensor as tt/import aesara.tensor as at/g"
    "s/tt\./at./g"
    "s/aet/at/g"
    "s/studenat/studentt/g"
    "s/theano/aesara/g"
    "s/Theano/Aesara/g"
    "s/pm\.sample()/pm.sample(return_inferencedata=False)/g"
    "s/, return_inferencedata\=True//g"
    "s/return_inferencedata\=True, //g"
    "s/return_inferencedata\=True//g"
)

for replace in "${replace_strings[@]}"; do
    find $1 -name "*.ipynb" -type f -exec sed -i -e "$replace" {} \;
    find $1 -name "*.py" -type f -exec sed -i -e "$replace" {} \;
done
