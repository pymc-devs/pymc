"""
Markov chain Monte Carlo methods in Python.
"""

__author__ = "Christopher Fonnesbeck <chris@trichech.us>"
__version__ = "1.3"

__modules__ = ["MCMC", "TimeSeries", "Backends", "Extensions", "Tests"]

for mod in __modules__:
    exec "from %s import *" % mod
del mod
