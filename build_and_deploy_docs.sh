#!/bin/sh

pushd docs/source
make html
ghp-import -c docs.pymc.io -n -p _build/html/
popd
