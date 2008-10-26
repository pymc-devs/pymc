"""
Markov Chain methods in Python.

A toolkit of stochastic methods for biometric analysis. Features
a Metropolis-Hastings MCMC sampler and both linear and unscented 
(non-linear) Kalman filters.

Pre-requisite modules: numpy, matplotlib
Required external components: TclTk

"""

# Make sure you're not importing from inside the source tree
import os
split_path = os.getcwd().split('/')
for i in xrange(len(split_path)):
    if split_path[i] == 'pymc':
        this_path = '/'.join(split_path[:i+1])
        file_list = os.listdir(this_path)
        if 'INSTALL.txt' in file_list:
            f = file(this_path + '/' + 'INSTALL.txt')
            f.readline()
            pream = f.readline()
            f.close()
            if pream == 'PyMC Installation Instructions\n':
                raise RuntimeError, 'You seem to be trying to import PyMC from inside its source tree.\n\t\t Please change to another directory and try again.'
        

# Core modules
from Node import *
from Container import *
from PyMCObjects import *
from Model import *
from distributions import *
from InstantiationDecorators import *
from NormalApproximation import *
from MCMC import *
from StepMethods import *
from convergencediagnostics import *
from CommonDeterministics import *
from Data import *

from tests import test

# Utilities modules
import utils
import MultiModelInference
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

