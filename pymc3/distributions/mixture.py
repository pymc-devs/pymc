import numpy as np
import theano.tensor as tt

from .distribution import Continuous
from .distributions.continuous import get_tau_sd


class Mixture(Continuous):
    def __init__(self, w, comp_dists, *args, **kwargs):
        super(Mixture, self).__init__(*args, **kwargs)
        
        self.w = w
        self.comp_dists = comp_dists
        
        try:
            self.mean = (w * comp_dists.mean).sum()
        except AttributeError:
            pass
    
    def _comp_logp(self, value):
        comp_dists = self.comp_dists
        
        try:
            value_ = value if value.ndim > 1 else tt.shape_padright(value)

            return comp_dists.logp(value_)
        except AttributeError:
            return tt.stack([comp_dist.logp(value) for comp_dist in comp_dists],
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
