#!/usr/bin/env python
from setuptools import setup
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
VERSION = "3.0"

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

with open('requirements.txt') as f:
    install_reqs = f.read().splitlines()

if sys.version_info < (3, 4):
    install_reqs.append('enum34')

test_reqs = ['nose']
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
          packages=['pymc3',
                    'pymc3.backends',
                    'pymc3.distributions',
                    'pymc3.examples',
                    'pymc3.glm',
                    'pymc3.step_methods',
                    'pymc3.step_methods.hmc',
                    'pymc3.tuning',
                    'pymc3.tests',
                    'pymc3.variational',
                    'pymc3.external',
                    'pymc3.gp',
                    'docs',
                    '.',
          ],
          classifiers=classifiers,
          install_requires=install_reqs,
          tests_require=test_reqs,
          extras_require={'edward': ['edward>=1.1.6']},
          test_suite='nose.collector')
