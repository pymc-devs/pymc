"""
Markov Chain methods in Python.

A toolkit of stochastic methods for biometric analysis. Features
a Metropolis-Hastings MCMC sampler and both linear and unscented 
(non-linear) Kalman filters.

Pre-requisite modules: numpy, matplotlib
Required external components: TclTk

"""

__version__ = '2.0rc2'

# Core modules
try:
    import Container_values
    del Container_values
except ImportError:
    raise ImportError, 'You seem to be importing PyMC from inside its source tree.\n\t\t Please change to another directory and try again.'
from Node import *
from Container import *
from PyMCObjects import *
from Model import *
from distributions import *
from InstantiationDecorators import *
from NormalApproximation import *
from MCMC import *
from StepMethods import *
from diagnostics import *
from CommonDeterministics import *

from tests import test

# Utilities modules
import utils
import CommonDeterministics
import distributions
import gp

# Optional modules
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
except ImportError:
    pass

