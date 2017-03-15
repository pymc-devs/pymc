import numpy as np
import theano.tensor as tt

from ..math import logsumexp
from .dist_math import bound
from .distribution import Discrete, Distribution, draw_values, generate_samples
from .continuous import get_tau_sd, Normal


def all_discrete(comp_dists):
    """
    Determine if all distributions in comp_dists are discrete
    """
    if isinstance(comp_dists, Distribution):
        return isinstance(comp_dists, Discrete)
    else:
        return all(isinstance(comp_dist, Discrete) for comp_dist in comp_dists)


class Mixture(Distribution):
    R"""
    Mixture log-likelihood

    Often used to model subpopulation heterogeneity

    .. math:: f(x \mid w, \theta) = \sum_{i = 1}^n w_i f_i(x \mid \theta_i)

    ========  ============================================
    Support   :math:`\cap_{i = 1}^n \textrm{support}(f_i)`
    Mean      :math:`\sum_{i = 1}^n w_i \mu_i`
    ========  ============================================

    Parameters
    ----------
    w : array of floats
        w >= 0 and w <= 1
        the mixutre weights
    comp_dists : multidimensional PyMC3 distribution or iterable of one-dimensional PyMC3 distributions
        the component distributions :math:`f_1, \ldots, f_n`
    """
    def __init__(self, w, comp_dists, *args, **kwargs):
        shape = kwargs.pop('shape', ())

        self.w = w
        self.comp_dists = comp_dists

        defaults = kwargs.pop('defaults', [])

        if all_discrete(comp_dists):
            dtype = kwargs.pop('dtype', 'int64')
        else:
            dtype = kwargs.pop('dtype', 'float64')

            try:
                self.mean = (w * self._comp_means()).sum(axis=-1)

                if 'mean' not in defaults:
                    defaults.append('mean')
            except AttributeError:
                pass

        try:
            comp_modes = self._comp_modes()
            comp_mode_logps = self.logp(comp_modes)
            self.mode = comp_modes[tt.argmax(w * comp_mode_logps, axis=-1)]

            if 'mode' not in defaults:
                defaults.append('mode')
        except AttributeError:
            pass

        super(Mixture, self).__init__(shape, dtype, defaults=defaults,
                                      *args, **kwargs)

    def _comp_logp(self, value):
        comp_dists = self.comp_dists

        try:
            value_ = value if value.ndim > 1 else tt.shape_padright(value)

            return comp_dists.logp(value_)
        except AttributeError:
            return tt.stack([comp_dist.logp(value) for comp_dist in comp_dists],
                            axis=1)

    def _comp_means(self):
        try:
            return tt.as_tensor_variable(self.comp_dists.mean)
        except AttributeError:
            return tt.stack([comp_dist.mean for comp_dist in self.comp_dists],
                            axis=1)

    def _comp_modes(self):
        try:
            return tt.as_tensor_variable(self.comp_dists.mode)
        except AttributeError:
            return tt.stack([comp_dist.mode for comp_dist in self.comp_dists],
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
        def random_choice(*args, **kwargs):
            w = kwargs.pop('w')
            w /= w.sum(axis=-1, keepdims=True)
            k = w.shape[-1]

            if w.ndim > 1:
                return np.row_stack([np.random.choice(k, p=w_) for w_ in w])
            else:
                return np.random.choice(k, p=w, *args, **kwargs)

        w = draw_values([self.w], point=point)

        w_samples = generate_samples(random_choice,
                                     w=w,
                                     broadcast_shape=w.shape[:-1] or (1,),
                                     dist_shape=self.shape,
                                     size=size).squeeze()
        comp_samples = self._comp_samples(point=point, size=size, repeat=repeat)

        if comp_samples.ndim > 1:
            return np.squeeze(comp_samples[np.arange(w_samples.size), w_samples])
        else:
            return np.squeeze(comp_samples[w_samples])


class NormalMixture(Mixture):
    R"""
    Normal mixture log-likelihood

    .. math:: f(x \mid w, \mu, \sigma^2) = \sum_{i = 1}^n w_i N(x \mid \mu_i, \sigma^2_i)

    ========  =======================================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\sum_{i = 1}^n w_i \mu_i`
    Variance  :math:`\sum_{i = 1}^n w_i^2 \sigma^2_i`
    ========  =======================================

    Parameters
    w : array of floats
        w >= 0 and w <= 1
        the mixutre weights
    mu : array of floats
        the component means
    sd : array of floats
        the component standard deviations
    tau : array of floats
        the component precisions
    """
    def __init__(self, w, mu, *args, **kwargs):
        _, sd = get_tau_sd(tau=kwargs.pop('tau', None),
                           sd=kwargs.pop('sd', None))

        super(NormalMixture, self).__init__(w, Normal.dist(mu, sd=sd),
                                            *args, **kwargs)
