#!/usr/bin/env python
from codecs import open
from os.path import realpath, dirname, join
from setuptools import setup, find_packages
import sys
import re

DISTNAME = 'pymc3'
DESCRIPTION = "Probabilistic Programming in Python: Bayesian Modeling and Probabilistic Machine Learning with Theano"
AUTHOR = 'PyMC Developers'
AUTHOR_EMAIL = 'pymc.devs@gmail.com'
URL = "http://github.com/pymc-devs/pymc3"
LICENSE = "Apache License, Version 2.0"

classifiers = ['Development Status :: 5 - Production/Stable',
               'Programming Language :: Python',
               'Programming Language :: Python :: 3',
               'Programming Language :: Python :: 3.5',
               'Programming Language :: Python :: 3.6',
               'License :: OSI Approved :: Apache Software License',
               'Intended Audience :: Science/Research',
               'Topic :: Scientific/Engineering',
               'Topic :: Scientific/Engineering :: Mathematics',
               'Operating System :: OS Independent']

PROJECT_ROOT = dirname(realpath(__file__))

# Get the long description from the README file
with open(join(PROJECT_ROOT, 'README.rst'), encoding='utf-8') as buff:
    LONG_DESCRIPTION = buff.read()

REQUIREMENTS_FILE = join(PROJECT_ROOT, 'requirements.txt')

with open(REQUIREMENTS_FILE) as f:
    install_reqs = f.read().splitlines()

test_reqs = ['pytest', 'pytest-cov']

def get_version():
    VERSIONFILE = join('pymc3', '__init__.py')
    lines = open(VERSIONFILE, 'rt').readlines()
    version_regex = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in lines:
        mo = re.search(version_regex, line, re.M)
        if mo:
            return mo.group(1)
    raise RuntimeError('Unable to find version in %s.' % (VERSIONFILE,))

if __name__ == "__main__":
    setup(name=DISTNAME,
          version=get_version(),
          maintainer=AUTHOR,
          maintainer_email=AUTHOR_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          long_description=LONG_DESCRIPTION,
          packages=find_packages(),
          package_data={'docs': ['*']},
          include_package_data=True,
          classifiers=classifiers,
          install_requires=install_reqs,
          tests_require=test_reqs,
          test_suite='nose.collector')
