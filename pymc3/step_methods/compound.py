'''
Created on Mar 7, 2011

@author: johnsalvatier
'''


class CompoundStep(object):
    """Step method composed of a list of several other step methods applied in sequence."""

    def __init__(self, methods):
        self.methods = list(methods)

    def step(self, point):
        for method in self.methods:
            point = method.step(point)
        return point
