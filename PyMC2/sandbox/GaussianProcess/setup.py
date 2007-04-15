#/usr/bin/env python  
try:
    import setuptools
except ImportError:
    pass
from numpy.distutils.core import setup, Extension

fcov = Extension(name='fcov',sources=['GaussianProcess/cov_funs/fcov.f'])
futils = Extension(name='futils',sources=['GaussianProcess/cov_funs/futils.f'])

distrib = setup(
name="GaussianProcess",
version="0.0",
description = "GaussianProcess version 0.0",
license="Academic Free License",
author="Anand Patil",
packages=["GaussianProcess", "GaussianProcess.cov_funs"],
ext_modules = [futils]
)
