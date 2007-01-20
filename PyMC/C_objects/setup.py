# do python setup.py build
# then the module PyMCObjects is in build/lib.<system>

from distutils.core import setup, Extension

PyMCObjects = Extension('PyMCObjects', sources = ['PyMCObjects.c'])

setup (name = 'PyMCObjects',
       version = '1.0',
       description = 'Lalala',
       ext_modules = [PyMCObjects])