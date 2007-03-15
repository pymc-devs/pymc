"""
Markov chain Monte Carlo methods in Python.
"""

__author__ = "Christopher Fonnesbeck <chris@trichech.us>"
__version__ = "1.2"

__modules__ = ["MCMC", "TimeSeries"]

for mod in __modules__:
    exec "from %s import *" % mod
del mod
