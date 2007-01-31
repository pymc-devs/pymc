"""
Database backends for PyMC.
"""

__author__ = "David Huard <david.huard@gmail.com>"
__version__ = "0.1"

__modules__ = ["memory_trace", "hdf5", "no_trace", "txt"]

# David- I haven't been able to get pytables working on my machine,
# unfortunately, that's why I've made the database imports optional.
# 
# -A

for mod in __modules__:
    try:
        exec "from %s import *" % mod
    except ImportError, msg:
        print "Database module " + mod + " could not be loaded: "
        print msg
del mod
