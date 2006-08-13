"""
Markov Chain methods in Python.

A toolkit of stochastic methods for biometric analysis. Features
a Metropolis-Hastings MCMC sampler and both linear and unscented 
(non-linear) Kalman filters.

Pre-requisite modules: numpy, matplotlib
Required external components: TclTk

"""

__author__ = "Christopher Fonnesbeck <chris@trichech.us>"
__version__ = "1.0"

__modules__ = ["MCMC", "TimeSeries"]

for mod in __modules__:
    exec "from %s import *" % mod
del mod
