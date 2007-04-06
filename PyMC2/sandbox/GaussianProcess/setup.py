#/usr/bin/env python  
try:
    import setuptools
except ImportError:
    pass
from numpy.distutils.core import setup, Extension

fcov = Extension(name='fcov',sources=['GaussianProcess/cov_funs/fcov.f'])

distrib = setup(
name="GaussianProcess",
version="0.1",
description = "GaussianProcess version 0.1",
license="Academic Free License",
author="Anand Patil",
# url="trichech.us",
packages=["GaussianProcess", "GaussianProcess.cov_funs"],
ext_modules = [fcov]
)
