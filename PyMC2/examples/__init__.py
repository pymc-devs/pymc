"""
Examples for PyMC.
"""

__modules__ = ["DisasterModel", "DisasterModel2", "model_1", "model_2", "model_3", "model_for_joint"]

for mod in __modules__:
    exec "from %s import *" % mod
del mod
