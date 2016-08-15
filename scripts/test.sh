#!/usr/bin/env bash

THEANO_FLAGS='gcc.cxxflags="-march=core2"' nosetests "$@"
