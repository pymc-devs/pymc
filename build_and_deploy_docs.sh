#!/bin/bash

pushd docs
bash convert_nbs_to_md.sh
popd
mkdocs build --clean
cp -R docs/source/_build/html site/manual
mkdocs gh-deploy
