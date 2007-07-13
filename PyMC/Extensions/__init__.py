"""
Extension modules for PyMC.

"""

#__author__ = ["Chris Fonnesbeck"]
#__version__ = "1.3"

__modules__ = ["ClosedCapture", "OpenCapture"]


available_modules = []
for mod in __modules__:
    try:
        exec "from %s import *" % mod
        available_modules.append(mod)
    except ImportError, msg:
        pass
del mod

