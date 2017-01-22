import numpy as np
from pymc3 import ParticleStep


class EmceeEnsemble(ParticleStep):
    default_blocked = True

    def __init__(self, nparticles=None, vars=None, model=None, mode=None, **kwargs):
        super(EmceeEnsemble, self).__init__(nparticles, vars, model, mode, **kwargs)

        def lnprob(p):
            s = self.t_func(p)
            if np.isnan(s):
                return -np.inf
            return s
        from emcee import EnsembleSampler
        self.emcee_sampler = EnsembleSampler(self.nparticles, self.dim, lnprob, **kwargs)

    def setup_step(self, point_array):
        assert point_array.shape == (self.nparticles, self.emcee_sampler.dim)
        self.sample_generator = self.emcee_sampler.sample(point_array, storechain=False, iterations=self._draws)

    def astep(self, point_array):
        if not hasattr(self, 'sample_generator'):
            self.setup_step(point_array)
        q, lnprob, state = self.sample_generator.next()
        return q


