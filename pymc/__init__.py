"""
Markov Chain Monte Carlo sampling toolkit.

Bayesian estimation, particularly using Markov chain Monte Carlo (MCMC), is an increasingly relevant approach to statistical estimation. However, few statistical software packages implement MCMC samplers, and they are non-trivial to code by hand. pymc is a python package that implements the Metropolis-Hastings algorithm as a python class, and is extremely flexible and applicable to a large suite of problems. pymc includes methods for summarizing output, plotting, goodness-of-fit and convergence diagnostics.

pymc only requires NumPy. All other dependencies such as matplotlib, SciPy, pytables, sqlite or mysql are optional.

"""

__version__ = '2.1alpha'

try:
    import numpy
except ImportError:
    raise ImportError, 'NumPy does not seem to be installed. Please see the user guide.'

# Core modules
from threadpool import *
try:
    import Container_values
    del Container_values
except ImportError:
    raise ImportError, 'You seem to be importing PyMC from inside its source tree. Please change to another directory and try again.'
from Node import *
from Container import *
from PyMCObjects import *
from InstantiationDecorators import *
from CommonDeterministics import *
from distributions import *
from Model import *
from StepMethods import *
from MCMC import *
from NormalApproximation import *

from tests import test

# Utilities modules
import utils
import CommonDeterministics
from CircularStochastic import *
import distributions
import gp

# Optional modules
try:
    from diagnostics import *
except ImportError:
    pass

try:
    import ScipyDistributions
except ImportError:
    pass

try:
    import parallel
except ImportError:
    pass

try:
    import sandbox
except ImportError:
    pass

try:
    import graph
except ImportError:
    pass

try:
    import Matplot
except:
    pass

