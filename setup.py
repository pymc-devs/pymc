from setuptools import setup


DISTNAME = 'pymc'
DESCRIPTION = "PyMC 3"
LONG_DESCRIPTION    = """Bayesian estimation, particularly using Markov chain Monte Carlo (MCMC), is an increasingly relevant approach to statistical estimation. However, few statistical software packages implement MCMC samplers, and they are non-trivial to code by hand. ``pymc`` is a python package that implements the Metropolis-Hastings algorithm as a python class, and is extremely flexible and applicable to a large suite of problems. ``pymc`` includes methods for summarizing output, plotting, goodness-of-fit and convergence diagnostics."""
MAINTAINER = 'John Salvatier'
MAINTAINER_EMAIL = 'jsalvati@u.washington.edu'
AUTHOR = 'John Salvatier and Christopher Fonnesbeck'
AUTHOR_EMAIL = 'chris.fonnesbeck@vanderbilt.edu'
URL = "http://github.com/pymc-devs/pymc"
LICENSE = "BSD"
VERSION = "3.0"

classifiers = ['Development Status :: 3 - Alpha',
               'Programming Language :: Python',
               'License :: OSI Approved :: BSD License',
               'Intended Audience :: Science/Research',
               'Topic :: Scientific/Engineering',
               'Topic :: Scientific/Engineering :: Mathematics',
               'Operating System :: OS Independent']

if __name__ == "__main__":

    setup(name=DISTNAME,
          version=VERSION,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          long_description=LONG_DESCRIPTION,
          packages=['pymc', 'pymc.distributions',
                    'pymc.step_methods', 'pymc.tuning', 'pymc.tests'],
          classifiers=classifiers,
          install_requires=['theano', 'numpy', 'scipy', 'numdifftools'])
