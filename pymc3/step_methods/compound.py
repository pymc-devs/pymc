'''
Created on Mar 7, 2011

@author: johnsalvatier
'''

def CompoundStep(methods):
    support = list(set(hasattr(m, 'nparticles') for m in methods))
    assert len(support) == 1, "All methods must all support/not support particles"
    if support[0]:
        nparticles = set(m.nparticles for m in methods if hasattr(m, 'nparticles'))
        if len(nparticles) > 1:
            raise ValueError("number of particles should be consistent, step methods have {} nparticles".format(nparticles))
        return _CompoundParticleStep(methods)
    else:
        return _CompoundStep(methods)


class _CompoundStep(object):
    """Step method composed of a list of several other step methods applied in sequence."""

    def __init__(self, methods):
        self.methods = list(methods)
        self.generates_stats = any(method.generates_stats for method in self.methods)
        self.stats_dtypes = []
        for method in self.methods:
            if method.generates_stats:
                self.stats_dtypes.extend(method.stats_dtypes)

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


class _CompoundParticleStep(_CompoundStep):

    @property
    def nparticles(self):
        return self.methods[0].nparticles

    @nparticles.setter
    def nparticles(self, value):
        for method in self.methods:
            method.nparticles = value

    @property
    def min_nparticles(self):
        return max(m.min_nparticles for m in self.methods)