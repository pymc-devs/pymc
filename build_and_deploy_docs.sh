#!/bin/sh

latesttag=$(git describe --tags `git rev-list --tags --max-count=1`)
echo checking out ${latesttag}
git checkout ${latesttag}
git submodule update --init --recursive
pushd docs/source
make html
ghp-import -c docs.pymc.io -n -p _build/html/
popd
