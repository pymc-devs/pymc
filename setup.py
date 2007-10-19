#/usr/bin/env python  

try:
    import setuptools
except ImportError:
    pass

from numpy.distutils.misc_util import Configuration
from numpy.distutils.system_info import get_info
config = Configuration('PyMC2',parent_package=None,top_path=None)

# If optimized lapack/ BLAS libraries are present, compile distributions that involve linear algebra against those.
# TODO: Use numpy's lapack_lite if optimized BLAS are not present.
try:
    lapack_info = get_info('lapack_opt',1)
    config.add_extension(name='flib',sources=['PyMC2/flib.f',
    'PyMC2/histogram.f', 'PyMC2/flib_blas.f', 'PyMC2/math.f', 'PyMC2/gibbsit.f'], extra_info=lapack_info)
except:
    config.add_extension(name='flib',sources=['PyMC2/flib.f', 'PyMC2/histogram.f', 'PyMC2/math.f'])
    
    
# Try to compile the Pyrex version of LazyFunction
config.add_extension(name='LazyFunction',sources=['PyMC2/LazyFunction.c'])
config.add_extension(name='Container_values', sources='PyMC2/Container_values.c')

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
            packages=["PyMC2", "PyMC2/database", "PyMC2/examples", "PyMC2/MultiModelInference", "PyMC2/tests"],
            url="trichech.us",
            **(config_dict))

