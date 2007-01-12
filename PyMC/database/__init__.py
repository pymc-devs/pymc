"""
Database backends for PyMC.
"""

__author__ = "David Huard <david.huard@gmail.com>"
__version__ = "0.1"

__modules__ = ["memory_trace", "hdf5"]

for mod in __modules__:
    exec "from %s import *" % mod
del mod
