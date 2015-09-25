from ..core import *
from .compound import CompoundStep
from ..vartypes import *

import numpy as np
from numpy.random import uniform
from numpy import log, isfinite
from enum import IntEnum, unique

__all__ = ['ArrayStep', 'ArrayStepShared', 'metrop_select', 'SamplerHist',
             'Competence', 'Constant']

# TODO Add docstrings to ArrayStep
@unique
class Competence(IntEnum):
    """Enum for charaterizing competence classes of step methods.
    Values include:
    0: incompatible
    1: compatible
    2: preferred
    3: ideal
    """
    incompatible = 0
    compatible   = 1
    preferred    = 2
    ideal        = 3

class BlockedStep(object):
    def __new__(cls, *args, **kwargs):
        blocked = kwargs.get('blocked')
        if blocked is None:
            # Try to look up default value from class
            blocked = getattr(cls, 'default_blocked', True)
            kwargs['blocked'] = blocked

        model = modelcontext(kwargs.get('model'))

        # vars can either be first arg or a kwarg
        if 'vars' not in kwargs and len(args) >= 1:
            vars = args[0]
            args = args[1:]
        elif 'vars' in kwargs:
            vars = kwargs.pop('vars')
        else: # Assume all model variables
            vars = model.vars

        #get the actual inputs from the vars
        vars = inputvars(vars) 

        if not blocked and len(vars) > 1:
            # In this case we create a separate sampler for each var
            # and append them to a CompoundStep
            steps = []
            for var in vars:
                step = super(BlockedStep, cls).__new__(cls)
                # If we don't return the instance we have to manually
                # call __init__
                step.__init__([var], *args, **kwargs)
                steps.append(step)

            return CompoundStep(steps)
        else:
            return super(BlockedStep, cls).__new__(cls)
            
    @staticmethod
    def competence(var):
        return Competence.incompatible
        
    @classmethod
    def _competence(cls, vars):
        return [cls.competence(var) for var in np.atleast_1d(vars)]
        

class ArrayStep(BlockedStep):
    def __init__(self, vars, fs, allvars=False, blocked=True):
        self.vars = vars
        self.ordering = ArrayOrdering(vars)
        self.fs = fs
        self.allvars = allvars
        self.blocked = blocked

    def step(self, point):
        bij = DictToArrayBijection(self.ordering, point)

        inputs = list(map(bij.mapf, self.fs))
        if self.allvars:
            inputs += [point]

        apoint = self.astep(bij.map(point), *inputs)
        return bij.rmap(apoint)


class ArrayStepShared(BlockedStep):
    """Faster version of ArrayStep that requires the substep method that does not wrap the functions the step method uses.

    Works by setting shared variables before using the step. This eliminates the mapping and unmapping overhead as well 
    as moving fewer variables around.
    """
    def __init__(self, vars, shared, blocked=True):
        """
        Parameters
        ----------
        vars : list of sampling variables
        shared : dict of theano variable -> shared variable
        blocked : Boolean (default True)
        """
        self.vars = vars
        self.ordering = ArrayOrdering(vars)
        self.shared = { str(var) : shared for var, shared in shared.items() }
        self.blocked = blocked

    def step(self, point):
        for var, share in self.shared.items():
            share.container.storage[0] = point[var]

        bij = DictToArrayBijection(self.ordering, point)

        apoint = self.astep(bij.map(point))
        return bij.rmap(apoint)

def metrop_select(mr, q, q0):
    # Perform rejection/acceptance step for Metropolis class samplers

    # Compare acceptance ratio to uniform random number
    if isfinite(mr) and log(uniform()) < mr:
        # Accept proposed value
        return q
    else:
        # Reject proposed value
        return q0


class SamplerHist(object):
    def __init__(self):
        self.metrops = []

    def acceptr(self):
        return np.minimum(np.exp(self.metrops), 1)

class Constant(ArrayStep):
    """
    Dummy sampler that returns the current value at every iteration. Useful for
    fixing parameters at a particular value.
    
    Parameters
    ----------
    vars : list
        List of variables for sampler.
    model : PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).
    """

    def __init__(self, vars, model=None, **kwargs):

        model = modelcontext(model)

        self.model = model

        super(Constant, self).__init__(vars, [model.fastlogp], **kwargs)

    def astep(self, q0, logp):
        
        return q0