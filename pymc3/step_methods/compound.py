'''
Created on Mar 7, 2011

@author: johnsalvatier
'''


class CompoundStep(object):
    """Step method composed of a list of several other step methods applied in sequence."""

    def __init__(self, methods):
        self.methods = list(methods)
        self.generates_stats = any(method.generates_stats for method in self.methods)
        self.stats_dtypes = []
        for method in self.methods:
            if method.generates_stats:
                self.stats_dtypes.extend(method.stats_dtypes)
        assert all(m.nparticles == methods[0].nparticles for m in methods), "number of particles should be consistent"

    def step(self, point):
        if self.generates_stats:
            states = []
            for method in self.methods:
                if method.generates_stats:
                    point, state = method.step(point)
                    states.extend(state)
                else:
                    point = method.step(point)
            return point, states
        else:
            for method in self.methods:
                point = method.step(point)
            return point

    @property
    def nparticles(self):
        return self.methods[0].nparticles

    @nparticles.setter
    def nparticles(self, value):
        for method in self.methods:
            method.nparticles = value

    @property
    def min_nparticles(self):
        return max(method.min_nparticles for method in self.methods)