#!/usr/bin/env python
from os.path import realpath, dirname, join
from setuptools import setup, find_packages
import sys


DISTNAME = 'pymc3'
DESCRIPTION = "PyMC3"
LONG_DESCRIPTION = """Bayesian estimation, particularly using Markov chain Monte Carlo (MCMC), is an increasingly relevant approach to statistical estimation. However, few statistical software packages implement MCMC samplers, and they are non-trivial to code by hand. ``pymc3`` is a python package that implements the Metropolis-Hastings algorithm as a python class, and is extremely flexible and applicable to a large suite of problems. ``pymc3`` includes methods for summarizing output, plotting, goodness-of-fit and convergence diagnostics."""
MAINTAINER = 'Thomas Wiecki'
MAINTAINER_EMAIL = 'thomas.wiecki@gmail.com'
AUTHOR = 'John Salvatier and Christopher Fonnesbeck'
AUTHOR_EMAIL = 'chris.fonnesbeck@vanderbilt.edu'
URL = "http://github.com/pymc-devs/pymc3"
LICENSE = "Apache License, Version 2.0"
VERSION = "3.1"

classifiers = ['Development Status :: 5 - Production/Stable',
               'Programming Language :: Python',
               'Programming Language :: Python :: 2',
               'Programming Language :: Python :: 3',
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3.4',
               'Programming Language :: Python :: 3.5',
               'Programming Language :: Python :: 3.6',
               'License :: OSI Approved :: Apache Software License',
               'Intended Audience :: Science/Research',
               'Topic :: Scientific/Engineering',
               'Topic :: Scientific/Engineering :: Mathematics',
               'Operating System :: OS Independent']

PROJECT_ROOT = dirname(realpath(__file__))
REQUIREMENTS_FILE = join(PROJECT_ROOT, 'requirements.txt')

with open(REQUIREMENTS_FILE) as f:
    install_reqs = f.read().splitlines()

if sys.version_info < (3, 4):
    install_reqs.append('enum34')

test_reqs = ['pytest', 'pytest-cov']
if sys.version_info[0] == 2:  # py3 has mock in stdlib
    test_reqs.append('mock')


if __name__ == "__main__":
    setup(name=DISTNAME,
          version=VERSION,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
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
