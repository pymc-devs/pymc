"""
PyMC multi-model inference support
"""

# __modules__ = ["RJMCMC", "ModelPosterior"]
__modules__ = ["ModelPosterior"]

for mod in __modules__:
    exec "from %s import *" % mod
del mod
