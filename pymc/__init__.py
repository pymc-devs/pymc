"""
Markov Chain Monte Carlo sampling toolkit.

Bayesian estimation, particularly using Markov chain Monte Carlo (MCMC), is an increasingly relevant approach to statistical estimation. However, few statistical software packages implement MCMC samplers, and they are non-trivial to code by hand. pymc is a python package that implements the Metropolis-Hastings algorithm as a python class, and is extremely flexible and applicable to a large suite of problems. pymc includes methods for summarizing output, plotting, goodness-of-fit and convergence diagnostics.

pymc only requires NumPy. All other dependencies such as matplotlib, SciPy, pytables, or sqlite are optional.

"""

__version__ = '2.2grad'

try:
    import numpy
except ImportError:
    raise ImportError('NumPy does not seem to be installed. Please see the user guide.')

# Core modules
from .threadpool import *
import os
import pymc
if os.getcwd().find(os.path.abspath(os.path.split(os.path.split(pymc.__file__)[0])[0]))>-1:
    from .six import print_
    print_('\n\tWarning: You are importing PyMC from inside its source tree.')
from .Node import *
from .Container import *
from .PyMCObjects import *
from .InstantiationDecorators import *
from .CommonDeterministics import *
from .NumpyDeterministics import *
from .distributions import *
from .Model import *
from .StepMethods import *
from .MCMC import *
from .NormalApproximation import *



from .tests import test

# Utilities modules
from . import utils
append = utils.append
from . import CommonDeterministics
from . import NumpyDeterministics
from .CircularStochastic import *
from . import distributions
from . import gp

# Optional modules
try:
    from .diagnostics import *
except ImportError:
    pass

try:
    from . import ScipyDistributions
except ImportError:
    pass

try:
    import parallel
except ImportError:
    pass

try:
    from . import sandbox
except ImportError:
    pass

try:
    from . import graph
except ImportError:
    pass

try:
    from . import Matplot
except:
    pass

