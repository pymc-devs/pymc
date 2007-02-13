#/usr/bin/env python  
try:
    import setuptools
except ImportError:
    pass
from numpy.distutils.core import setup, Extension

# Compile flib (fortran source for statistical distributions.)
flib = Extension(name='flib',sources=['PyMC2/flib.f'])

try:    
    # Compile base objects in C
    PyMCObjects = Extension(name='PyMCObjects', 
        sources = ['PyMC2/PyMCObjects/Parameter.c',
        'PyMC2/PyMCObjects/Node.c',
        'PyMC2/PyMCObjects/PyMCObjects.c'])
    ext_modules = [PyMCObjects, flib]

except:
    print '\n'+60*'*'
    print 'Not able to compile C objects, falling back to pure python.'
    print 60*'*'+'\n'
    
    ext_modules = [flib]

distrib = setup(
name="PyMC2",
version="2.0",
description = "PyMC version 2.0",
license="Academic Free License",
url="trichech.us",
packages=["PyMC2", "PyMC2.database", "PyMC2.tests", "PyMC2.examples", 
"PyMC2.MultiModelInference"],
ext_modules = ext_modules
)
    
