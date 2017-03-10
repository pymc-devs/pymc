import numpy as np
from pymc3 import ParticleStep

try:
    import emcee


    class EmceeEnsemble(ParticleStep):
        """
        ParticleStep function that uses the external library emcee.
        Please refer to emcee documentation for emcee_kwargs
        """

        default_blocked = True
        generates_stats = False

        def __init__(self, vars=None, model=None, emcee_kwargs=None, **kwargs):
            super(EmceeEnsemble, self).__init__(vars, model, **kwargs)
            self.sample_generator = None
            self.min_nparticles = self.dim * 2 + 2
            if emcee_kwargs is None:
                self.emcee_kwargs = {}
            else:
                self.emcee_kwargs = emcee_kwargs

        def lnprob(self, p):
            s = self.t_func(p)
            if np.isnan(s):
                return -np.inf
            return s

        def setup_step(self, point_array):
            self.emcee_sampler = emcee.EnsembleSampler(self.nparticles, self.dim, self.lnprob, **self.emcee_kwargs)
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



