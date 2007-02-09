"""
Database backends for PyMC.
"""

__author__ = "David Huard <david.huard@gmail.com>"
__version__ = "0.1"

__modules__ = ["ram", "hdf5", "no_trace", "txt", "sqlite", "mysql"]

# David- I haven't been able to get pytables working on my machine,
# unfortunately, that's why I've made the database imports optional.
# 
# -A
# Alright. Since this will be a fairly common situation, I think it would be 
# alright to let the import fail silently. We could maybe offer a status 
# fonction that would tell the user what exactly has been loaded succesfully. 

for mod in __modules__:
    try:
        exec "from %s import *" % mod
    except ImportError, msg:
        print "Database module " + mod + " could not be loaded: "
        print msg
del mod
