#!/bin/bash

pushd docs
bash convert_nbs_to_md.sh
popd
mkdocs build --clean
mkdocs gh-deploy
