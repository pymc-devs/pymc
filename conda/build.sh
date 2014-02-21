#!/bin/bash

if [ "$(uname)" == "Darwin" ]; then
    export CFLAGS="-arch x86_64"
    export FFLAGS="-static -ff2c -arch x86_64"
    export LDFLAGS="-Wall -undefined dynamic_lookup -bundle -arch x86_64"
fi

$PYTHON setup.py build
$PYTHON setup.py install
