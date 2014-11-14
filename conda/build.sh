#!/bin/bash
if [ "$(uname)" == "Darwin" ]; then
    # export LDFLAGS="-Wall -undefined dynamic_lookup -bundle -arch x86_64"
    cp /usr/local/Cellar/gcc/4.9.1/lib/gcc/x86_64-apple-darwin13.3.0/4.9.1/libgfortran*.*a .
    LDFLAGS="-undefined dynamic_lookup -bundle -Wl,-search_paths_first -L$(pwd) $LDFLAGS"
fi

$PYTHON setup.py build
$PYTHON setup.py install
