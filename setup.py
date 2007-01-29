#/usr/bin/env python  
try:
    import setuptools
except ImportError:
    pass
from numpy.distutils.core import setup, Extension

# Compile flib (fortran source for statistical distributions.)
flib = Extension(name='PyMC.flib',sources=['PyMC/flib.f'])

# Compile base objects in C
PyMCObjects = Extension(name='PyMC.PyMCObjects', sources = [	'PyMC/PyMCObjects/PyMCBase.c',
																'PyMC/PyMCObjects/Parameter.c',
																'PyMC/PyMCObjects/Node.c',
																'PyMC/PyMCObjects/RemoteProxy.c',
																'PyMC/PyMCObjects/PyMCObjects.c'])

distrib = setup(
    name="PyMC",
    version="2.0",
    description = "PyMC version 2.0",
    license="Academic Free License",
    url="trichech.us",
    packages=["PyMC", "PyMC.database", "PyMC.tests"],
    ext_modules = [PyMCObjects,flib]    
)
