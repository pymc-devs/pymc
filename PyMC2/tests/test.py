#!/Library/Frameworks/Python.framework/Versions/2.4/bin/python

from PyMC2 import Model
from PyMC2.examples import model_1

M = Model(model_1,dbase = 'memory_trace')

