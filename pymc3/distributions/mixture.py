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
        
        
    def logp(self, value):
        w = self.w
        
        return bound(logsumexp(tt.log(w) + self._comp_logp(value), axis=1).sum(),
                     w >= 0, w <= 1, tt.allclose(w.sum(axis=-1), 1))

    
class NormalMixture(Mixture):
    def __init__(self, w, mu, *args, **kwargs):
        _, sd = get_tau_sd(tau=kwargs.pop('tau', None),
                           sd=kwargs.pop('sd', None))
        
        super(NormalMixture, self).__init__(w, pm.Normal.dist(mu, sd=sd),
                                            *args, **kwargs)
