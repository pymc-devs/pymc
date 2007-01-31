#!/Library/Frameworks/Python.framework/Versions/2.4/bin/python

from PyMC import Model
from PyMC.examples import model_1

M = Model(model_1,dbase = 'memory_trace')

