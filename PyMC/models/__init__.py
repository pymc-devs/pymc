"""Models used to test PyMC new design."""

__modules__ = ["model_1", "model_2", "model_3", "model_for_joint"]

for mod in __modules__:
    exec "from %s import *" % mod
del mod
