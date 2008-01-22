#/usr/bin/env python  

try:
    import setuptools
except ImportError:
    pass

from numpy.distutils.misc_util import Configuration
from numpy.distutils.system_info import get_info
config = Configuration('pymc',parent_package=None,top_path=None)

# If optimized lapack/ BLAS libraries are present, compile distributions that involve linear algebra against those.
# TODO: Use numpy's lapack_lite if optimized BLAS are not present.


lapack_info = get_info('lapack_opt',1)
if lapack_info:
    print 'Compiling everything'
    config.add_extension(name='flib',sources=['PyMC/flib.f',
    'PyMC/histogram.f', 'PyMC/flib_blas.f', 'PyMC/math.f', 'PyMC/gibbsit.f'], extra_info=lapack_info)
else:
    print 'Not compiling flib_blas'
    config.add_extension(name='flib',sources=['PyMC/flib.f', 'PyMC/histogram.f', 'PyMC/math.f'])
    
# Try to compile the Pyrex version of LazyFunction
config.add_extension(name='LazyFunction',sources=['PyMC/LazyFunction.c'])
config.add_extension(name='Container_values', sources='PyMC/Container_values.c')

config_dict = config.todict()
try:
    config_dict.pop('packages')
except:
    pass

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(  version="2.0",
            description = "PyMC version 2.0",
            license="Academic Free License",
            packages=["pymc", "pymc/database", "pymc/examples", "pymc/MultiModelInference", "pymc/tests"],
            url="trichech.us",
            **(config_dict))

