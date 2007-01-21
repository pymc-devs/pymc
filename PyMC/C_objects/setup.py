# do python setup.py build
# then the module PyMCObjects is in build/lib.<system>

from distutils.core import setup, Extension

PyMCObjects = Extension('PyMCObjects', sources = ['PyMCBase.c','Parameter.c','Node.c','PyMCObjects.c'])

setup (name = 'PyMCObjects',
       version = '1.0',
       description = 'C versions of the basic PyMC objects',
       ext_modules = [PyMCObjects])