'''
Created on Mar 7, 2011

@author: johnsalvatier
'''

from ..core import *

class CompoundStep(object):
    """Step method composed of a list of several other step methods applied in sequence."""
    def __init__(self, methods):
        self.methods = list(methods)

    def step(self, point):
        for method in self.methods:
            point = method.step(point)
        return point


class SingleComponentStep(CompoundStep):
    """Step method that is applied to each random variable separately.  Step methods are applied in sequence."""
    def __init__(self, vars=None, **kwargs):
        model = modelcontext(kwargs.get('model', None))
        if vars is None:
            vars = model.vars

        if 'step_method' in kwargs:
            self.step_method = kwargs.pop('step_method')
        elif not hasattr(self, 'step_method'):
            raise ArgumentError("Either supply step_method argument or use an inherited class like SingleComponentSlice or SingleComponentMetropolis.")

        methods = [self.step_method(vars=[v], **kwargs) for v in vars]

        super(SingleComponentStep, self).__init__(methods)
