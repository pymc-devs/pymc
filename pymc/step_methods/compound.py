'''
Created on Mar 7, 2011

@author: johnsalvatier
'''
from ..quickclass import *

@quickclass(object)
def CompoundStep(methods):
    methods = list(methods)
    tune = False
    def step(point):
        for method in methods:
            point = method.step(point)
        return point
    return locals()
