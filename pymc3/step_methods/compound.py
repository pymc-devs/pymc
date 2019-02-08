'''
Created on Mar 7, 2011

@author: johnsalvatier
'''
import numpy as np


class CompoundStep:
    """Step method composed of a list of several other step
    methods applied in sequence."""

    def __init__(self, methods):
        self.methods = list(methods)
        self.generates_stats = any(
            method.generates_stats for method in self.methods)
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
            # Model logp can only be the logp of the _last_ state, if there is
            # one. Pop all others (if dict), or set to np.nan (if namedtuple).
            for state in states[:-1]:
                if isinstance(state, dict):
                    state.pop('model_logp', None)
                elif isinstance(state, namedtuple):
                    state = state._replace(logp=np.nan)
            return point, states
        else:
            for method in self.methods:
                point = method.step(point)
            return point

    def warnings(self):
        warns = []
        for method in self.methods:
            if hasattr(method, 'warnings'):
                warns.extend(method.warnings())
        return warns

    def stop_tuning(self):
        for method in self.methods:
            method.stop_tuning()

    @property
    def vars_shape_dtype(self):
        dtype_shapes = {}
        for method in self.methods:
            dtype_shapes.update(method.vars_shape_dtype)
        return dtype_shapes
