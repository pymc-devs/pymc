#!/usr/bin/env bash

which pylint
which python
pylint --rcfile=.pylintrc pymc3
