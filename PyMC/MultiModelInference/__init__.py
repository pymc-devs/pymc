"""
PyMC multi-model inference support
"""

__modules__ = ["ModelPosterior",
				"RJMCMC"]

for mod in __modules__:
    exec "from %s import *" % mod
del mod
