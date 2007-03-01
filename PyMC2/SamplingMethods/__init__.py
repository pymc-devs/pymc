"""
Sampling Methods for PyMC
"""

__modules__ = ["SamplingMethods", "GibbsSampler"]

for mod in __modules__:
    exec "from %s import *" % mod
del mod
