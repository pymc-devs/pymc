import numpy as np
from pymc3 import ParticleStep

try:
    import emcee


    class EmceeEnsemble(ParticleStep):
        default_blocked = True

        def __init__(self, nparticles=None, vars=None, model=None, mode=None, **kwargs):
            super(EmceeEnsemble, self).__init__(nparticles, vars, model, mode, **kwargs)
            self.sample_generator = None

            def lnprob(p):
                s = self.t_func(p)
                if np.isnan(s):
                    return -np.inf
                return s

            self.emcee_sampler = emcee.EnsembleSampler(self.nparticles, self.dim, lnprob, **kwargs)


        def setup_step(self, point_array):
            assert point_array.shape == (self.nparticles, self.emcee_sampler.dim)
            self.sample_generator = self.emcee_sampler.sample(point_array, storechain=False, iterations=100000)
            # very hacky way to make emcee keep going without `iterations` information

        def astep(self, point_array):
            if self.sample_generator is None:
                self.setup_step(point_array)
            try:
                q, lnprob, state = self.sample_generator.next()
            except StopIteration:
                self.sample_generator = None
                return self.astep(point_array)
            return q


except ImportError:
    raise ImportError("emcee is required to make all these EmceeEnsembleSamplers work! `pip install emcee`")



