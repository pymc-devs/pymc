from .arraystep import ArrayStepShared, ArrayStep, Competence
from ..model import modelcontext
from ..theanof import inputvars, make_shared_replacements
from ..blocking import ArrayOrdering, DictToArrayBijection
from emcee import EnsembleSampler
import numpy as np

class ParticleStep(ArrayStepShared):
    def __new__(cls, *args, **kwargs):
        obj = ArrayStepShared.__new__(cls, *args[1:], **kwargs)
        return obj

    def __init__(self, nparticles=None, model=None, mode=None, **kwargs):
        model = modelcontext(model)
        vars = model.vars
        vars = inputvars(vars)
        shared = make_shared_replacements(vars, model)
        self.mode = mode
        self.model = model
        self.vars = vars
        if nparticles is None:
            self.nparticles = (len(self.vars)*2) + 1
        else:
            self.nparticles = nparticles
        self.ordering = ArrayOrdering(vars, self.nparticles)
        self.shared = {str(var): shared for var, shared in shared.items()}
        self.blocked = True

    def step(self, point):
        for var, share in self.shared.items():
            share.container.storage[0] = point[var]

        bij = DictToArrayBijection(self.ordering, point)
        apoint = self.astep(bij.map(point))
        return bij.rmap(apoint)


class EmceeSamplerStep(ParticleStep):
    default_blocked = True

    def __init__(self, nparticles=None, model=None, mode=None, **kwargs):
        super(EmceeSamplerStep, self).__init__(nparticles, model, mode, **kwargs)
        dim = len(self.model.vars)
        t_func = self.model.makefn([self.model.logpt])
        def lnpostfn(p):
            s = t_func(*p)
            return s[0]
        self.emcee_sampler = EnsembleSampler(self.nparticles, dim, lnpostfn, **kwargs)
        self.names = []


    def flatten_args(self):
        pass

    def setup_step(self, point_array):
        assert point_array.shape == (self.emcee_sampler.dim, self.nparticles)
        self.sample_generator = self.emcee_sampler.sample(point_array.T, storechain=True, iterations=self._draws)

    def astep(self, point_array):
        if not hasattr(self, 'sample_generator'):
            self.setup_step(point_array)
        q, lnprob, state = self.sample_generator.next()
        return q.T