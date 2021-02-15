#!/bin/sh

git --git-dir=docs/source/pymc-examples/.git --work-tree=docs/source/pymc-examples checkout fast-docs-build
git submodule update --remote
pushd docs/source
make html
ghp-import -c docs.pymc.io -n -p _build/html/
popd
