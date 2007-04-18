#/usr/bin/env python  

try:
    import setuptools
except ImportError:
    pass

from numpy.distutils.misc_util import Configuration
from numpy.distutils.system_info import get_info
config = Configuration('PyMC2',parent_package=None,top_path=None)

# Compile flib (fortran source for statistical distributions.)
config.add_extension(name='flib',sources=['PyMC2/flib.f'])

# If optimized lapack/ BLAS libraries are present, compile distributions that involve linear algebra against those.
lapack_info = get_info('lapack_opt',0)
if lapack_info:
    config.add_extension(name='flib_blas',sources=['PyMC2/flib_blas.f'],extra_info=lapack_info)

# Try to compile the Pyrex version of LazyFunction
try:
    config.add_extension(name='PyrexLazyFunction',sources=['PyMC2/PyrexLazyFunction.c'])
except:
    pass

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(  version="2.0",
            description = "PyMC version 2.0",
            license="Academic Free License",
            packages=["PyMC2", "PyMC2.database", "PyMC2.examples", "PyMC2.MultiModelInference", "PyMC2/tests"],
            url="trichech.us",
            **(config.todict()))


# #/usr/bin/env python  
# try:
#     import setuptools
# except ImportError:
#     pass
# from numpy.distutils.core import setup, Extension
# 
# # Compile flib (fortran source for statistical distributions.)
# flib = Extension(name='flib',sources=['PyMC2/flib.f'])
# flib_blas = Extension(name='flib_blas',sources=['PyMC2/flib_blas.f'])
# try:
#     PyrexLazyFunction = Extension(name='PyrexLazyFunction',sources=['PyMC2/PyrexLazyFunction.c'])
#     ext_modules = [flib, flib_blas, PyrexLazyFunction]
# except:
#     ext_modules = [flib, flib_blas]
# 
# distrib = setup(
# name="PyMC2",
# version="2.0",
# description = "PyMC version 2.0",
# license="Academic Free License",
# url="trichech.us",
# packages=["PyMC2", "PyMC2.database", "PyMC2/tests", "PyMC2.examples", 
# "PyMC2.MultiModelInference"],
# ext_modules = ext_modules
# )
