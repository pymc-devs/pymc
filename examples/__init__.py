"""
Examples for PyMC.
"""

__modules__ = ["DisasterModel"]

for mod in __modules__:
    exec "from %s import *" % mod
del mod
