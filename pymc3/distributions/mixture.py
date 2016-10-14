import numpy as np
import theano.tensor as tt

from ..math import logsumexp
from .dist_math import bound
from .distribution import Discrete, Distribution
from .distributions.continuous import get_tau_sd


def all_discrete(comp_dists):
    if isinstance(comp_dists, Distribution):
        return isinstance(comp_dists, Discrete)
    else:
        return all(isinstance(comp_dist, Discrete) for comp_dist in comp_dists)


class Mixture(Distribution):
    def __init__(self, w, comp_dists, *args, **kwargs):
        shape = kwargs.pop('shape', ())

        if all_discrete(comp_dists):
            dtype = kwargs.pop('dtype', 'int64')
            defaults = kwargs.pop('defaults', ['mode'])
        else:
            dtype = kwargs.pop('dtype', 'float64')
            defaults = kwargs.pop('defaults', ['mean', 'mode'])

            self.mean = (w * comp_dists.mean).sum(axis=-1)

        super(Mixture, self).__init__(shape, dtype, defaults=defaults,
                                      *args, **kwargs)
        
        self.w = w
        self.comp_dists = comp_dists

        comp_modes = self._comp_modes()
        comp_mode_logps = self.logp(comp_modes)
        self.mode = comp_modes[tt.argmax(comp_modes, axis=-1)]
    
    def _comp_logp(self, value):
        comp_dists = self.comp_dists
        
        try:
            value_ = value if value.ndim > 1 else tt.shape_padright(value)

            return comp_dists.logp(value_)
        except AttributeError:
            return tt.stack([comp_dist.logp(value) for comp_dist in comp_dists],
                            axis=1)

    def _comp_modes(self):
        try:
            return self.comp_dists.mode()
        except AttributeError:
            return tt.stack([comp_dist.mode for comp_dist in comp_dists],
                            axis=1)

    def _comp_samples(self, point=None, size=None, repeat=None):
        try:
            samples = self.comp_dists.random(point=point, size=size, repeat=repeat)
        except AttributeError:
            samples = np.column_stack([comp_dist.random(point=point, size=size, repeat=repeat)
                                       for comp_dist in self.comp_dists])

        return np.squeeze(samples)
        
    def logp(self, value):
        w = self.w
        
        return bound(logsumexp(tt.log(w) + self._comp_logp(value), axis=-1).sum(),
                     w >= 0, w <= 1, tt.allclose(w.sum(axis=-1), 1))

    def random(self, point=None, size=None, repeat=None):
        w = draw_values([self.w], point=point)

        w_samples = generate_samples(np.random.multinomial, 1, w,
                                     broadcast_shape=w.shape[:-1] or (1,),
                                     dist_shape=self.shape,
                                     size=size)
        comp_samples = self._comp_samples(point=point, size=size, repeat=repeat)

        return comp_samples[np.squeeze(w_samples == 1)]

    
class NormalMixture(Mixture):
    def __init__(self, w, mu, *args, **kwargs):
        _, sd = get_tau_sd(tau=kwargs.pop('tau', None),
                           sd=kwargs.pop('sd', None))
        
        super(NormalMixture, self).__init__(w, pm.Normal.dist(mu, sd=sd),
                                            *args, **kwargs)
