import pymc3 as pm
import theano

from .arraystep import ArrayStepShared
from ..blocking import ArrayOrdering, DictToArrayBijection
from ..model import modelcontext
from ..theanof import inputvars, make_shared_replacements


class ParticleStep(ArrayStepShared):
    """
    Defined as a controller for a population of walkers/particles which all have individual parameter/logp values.
    This implementation assumes nothing about what is happening in the actual proprietary `astep`
    To use a ParticleStep, write an astep method to except and return a 1d array of variables.
    The higher level `step` method will control where the variables go and what reshapes to perform
    """

    def __new__(cls, *args, **kwargs):
        obj = ArrayStepShared.__new__(cls, *args[1:], **kwargs)
        return obj

    def __init__(self, nparticles=None, vars=None, model=None, mode=None, **kwargs):
        model = modelcontext(model)
        if vars is not None and len(vars) != len(model.vars):
            raise NotImplementedError(
                'ParticleStep sampling is not yet implemented along with other sampling techniques!')
        vars = model.vars
        vars = inputvars(vars)
        shared = make_shared_replacements(vars, model)
        self.mode = mode
        self.model = model
        self.vars = vars
        self.dim = sum([i.dsize for i in self.vars])
        if nparticles is None:
            self.nparticles = (self.dim*2) + 2
        else:
            self.nparticles = nparticles
        self.ordering = ArrayOrdering(vars, self.nparticles)
        self.shared = {str(var): shared for var, shared in shared.items()}
        self.blocked = True
        self.t_func = logp(model.logpt, vars, shared)

    def astep(self, point_array):
        raise NotImplementedError("This is a ParticleStep template, it doesn't do anything!")


    def step(self, point):
        for var, share in self.shared.items():
            share.container.storage[0] = point[var]

        bij = DictToArrayBijection(self.ordering, point)
        apoint = self.astep(bij.map(point))
        return bij.rmap(apoint)


def logp(logp, vars, shared):
    [logp0], inarray0 = pm.join_nonshared_inputs([logp], vars, shared)
    tensor_type = inarray0.type
    f = theano.function([inarray0], logp0)
    f.trust_input = True
    return f
