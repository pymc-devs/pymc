import numpy as np
import theano
from theano import tensor as tt

import pymc3 as pm
from pymc3.distributions.dist_math import rho2sd, log_normal, log_normal_mv
from pymc3.variational.opvi import Approximation
from pymc3.theanof import memoize


__all__ = [
    'MeanField',
    'FullRank',
    'Empirical',
    'sample_approx'
]


class MeanField(Approximation):
    """
    Mean Field approximation to the posterior where spherical Gaussian family
    is fitted to minimize KL divergence from True posterior. It is assumed
    that latent space variables are uncorrelated that is the main drawback
    of the method

    Parameters
    ----------
    local_rv : dict
        mapping {model_variable -> local_variable}
        Local Vars are used for Autoencoding Variational Bayes
        See (AEVB; Kingma and Welling, 2014) for details

    model : PyMC3 model for inference

    start : Point
        initial mean

    cost_part_grad_scale : float or scalar tensor
        Scaling score part of gradient can be useful near optimum for
        archiving better convergence properties. Common schedule is
        1 at the start and 0 in the end. So slow decay will be ok.
        See (Sticking the Landing; Geoffrey Roeder,
        Yuhuai Wu, David Duvenaud, 2016) for details

    seed : None or int
        leave None to use package global RandomStream or other
        valid value to create instance specific one

    References
    ----------
    Geoffrey Roeder, Yuhuai Wu, David Duvenaud, 2016
        Sticking the Landing: A Simple Reduced-Variance Gradient for ADVI
        approximateinference.org/accepted/RoederEtAl2016.pdf
    """
    @property
    def mean(self):
        return self.shared_params['mu']

    @property
    def rho(self):
        return self.shared_params['rho']

    @property
    def cov(self):
        return tt.diag(rho2sd(self.rho)**2)

    def create_shared_params(self, **kwargs):
        start = self.gbij.map(kwargs.get('start', self.model.test_point))
        return {'mu': theano.shared(
                    pm.floatX(start),
                    'mu'),
                'rho': theano.shared(
                    np.zeros((self.global_size,), dtype=theano.config.floatX),
                    'rho')
                }

    def log_q_W_global(self, z):
        """
        log_q_W samples over q for global vars
        """
        mu = self.scale_grad(self.mean)
        rho = self.scale_grad(self.rho)
        z = z[self.global_slc]
        logq = tt.sum(log_normal(z, mu, rho=rho))
        return logq

    def random_global(self, size=None, no_rand=False):
        initial = self.initial(size, no_rand, l=self.global_size)
        sd = rho2sd(self.rho)
        mu = self.mean
        return sd * initial + mu


class FullRank(Approximation):
    """
    Full Rank approximation to the posterior where Multivariate Gaussian family
    is fitted to minimize KL divergence from True posterior. In contrast to
    MeanField approach correlations between variables are taken in account. The
    main drawback of the method is computational cost.

    Parameters
    ----------
    local_rv : dict
        mapping {model_variable -> local_variable}
        Local Vars are used for Autoencoding Variational Bayes
        See (AEVB; Kingma and Welling, 2014) for details

    model : PyMC3 model for inference

    start : Point
        initial mean

    cost_part_grad_scale : float or scalar tensor
        Scaling score part of gradient can be useful near optimum for
        archiving better convergence properties. Common schedule is
        1 at the start and 0 in the end. So slow decay will be ok.
        See (Sticking the Landing; Geoffrey Roeder,
        Yuhuai Wu, David Duvenaud, 2016) for details

    seed : None or int
        leave None to use package global RandomStream or other
        valid value to create instance specific one

    References
    ----------
    Geoffrey Roeder, Yuhuai Wu, David Duvenaud, 2016
        Sticking the Landing: A Simple Reduced-Variance Gradient for ADVI
        approximateinference.org/accepted/RoederEtAl2016.pdf
    """
    def __init__(self, local_rv=None, model=None, cost_part_grad_scale=1, gpu_compat=False, seed=None):
        super(FullRank, self).__init__(
            local_rv=local_rv, model=model,
            cost_part_grad_scale=cost_part_grad_scale,
            seed=seed
        )
        self.gpu_compat = gpu_compat

    @property
    def L(self):
        return self.shared_params['L_tril'][self.tril_index_matrix]

    @property
    def mean(self):
        return self.shared_params['mu']

    @property
    def cov(self):
        L = self.L
        return L.dot(L.T)

    @property
    def num_tril_entries(self):
        n = self.global_size
        return int(n * (n + 1) / 2)

    @property
    def tril_index_matrix(self):
        n = self.global_size
        num_tril_entries = self.num_tril_entries
        tril_index_matrix = np.zeros([n, n], dtype=int)
        tril_index_matrix[np.tril_indices(n)] = np.arange(num_tril_entries)
        tril_index_matrix[np.tril_indices(n)[::-1]] = np.arange(num_tril_entries)
        return tril_index_matrix

    def create_shared_params(self, **kwargs):
        start = self.gbij.map(kwargs.get('start', self.model.test_point))
        n = self.global_size
        L_tril = (
            np.eye(n)
            [np.tril_indices(n)]
            .astype(theano.config.floatX)
        )
        return {'mu': theano.shared(pm.floatX(start), 'mu'),
                'L_tril': theano.shared(L_tril, 'L_tril')
                }

    def log_q_W_global(self, z):
        """
        log_q_W samples over q for global vars
        """
        mu = self.scale_grad(self.mean)
        L = self.scale_grad(self.L)
        z = z[self.global_slc]
        return log_normal_mv(z, mu, chol=L, gpu_compat=self.gpu_compat)

    def random_global(self, size=None, no_rand=False):
        # (samples, dim) or (dim, )
        initial = self.initial(size, no_rand, l=self.global_size).T
        # (dim, dim)
        L = self.L
        # (dim, )
        mu = self.mean
        # x = Az + m, but it assumes z is vector
        # When we get z with shape (samples, dim)
        # we need to transpose Az
        return L.dot(initial).T + mu

    @classmethod
    def from_mean_field(cls, mean_field, gpu_compat=False):
        """
        Construct FullRank from MeanField approximation

        Parameters
        ----------
        mean_field : MeanField
            approximation to start with

        Flags
        -----
        gpu_compat : bool
            use GPU compatible version or not

        Returns
        -------
        FullRank
        """
        full_rank = object.__new__(cls)  # type: FullRank
        full_rank.gpu_compat = gpu_compat
        full_rank.__dict__.update(mean_field.__dict__)
        full_rank.shared_params = full_rank.create_shared_params()
        full_rank.shared_params['mu'].set_value(
            mean_field.shared_params['mu'].get_value()
        )
        rho = mean_field.shared_params['rho'].get_value()
        n = full_rank.global_size
        L_tril = (
            np.diag(np.log1p(np.exp(rho)))  # rho2sd
            [np.tril_indices(n)]
            .astype(theano.config.floatX)
        )
        full_rank.shared_params['L_tril'].set_value(L_tril)
        return full_rank


class Empirical(Approximation):
    """
    Builds Approximation instance from a given trace,
    it has the same interface as variational approximation

    Parameters
    ----------
    trace : MultiTrace
    local_rv : dict
        Experimental for Histogram approximation
        mapping {model_variable -> local_variable}
        Local Vars are used for Autoencoding Variational Bayes
        See (AEVB; Kingma and Welling, 2014) for details

    model : PyMC3 model

    seed : None or int
        leave None to use package global RandomStream or other
        valid value to create instance specific one

    Usage
    -----
    >>> with model:
    ...     step = NUTS()
    ...     trace = sample(1000, step=step)
    ...     histogram = Empirical(trace[100:])
    """
    def __init__(self, trace, local_rv=None, model=None, seed=None):
        super(Empirical, self).__init__(local_rv=local_rv, model=model, trace=trace, seed=seed)

    def check_model(self, model, **kwargs):
        trace = kwargs.get('trace')
        if (trace is not None
            and not all([var.name in trace.varnames
                         for var in model.free_RVs])):
            raise ValueError('trace has not all FreeRV')

    def create_shared_params(self, **kwargs):
        trace = kwargs.get('trace')
        if trace is None:
            histogram = np.atleast_2d(self.gbij.map(self.model.test_point))
        else:
            histogram = np.empty((len(trace), self.global_size))
            for i in range(len(trace)):
                histogram[i] = self.gbij.map(trace[i])
        return theano.shared(pm.floatX(histogram), 'histogram')

    def randidx(self, size=None):
        if size is None:
            size = ()
        elif isinstance(size, tt.TensorVariable):
            if size.ndim < 1:
                size = size[None]
            elif size.ndim > 1:
                raise ValueError('size ndim should be no more than 1d')
            else:
                pass
        else:
            size = tuple(np.atleast_1d(size))
        return (self._rng
                .uniform(size=size, low=0.0, high=self.histogram.shape[0] - 1e-16)
                .astype('int64'))

    def random_global(self, size=None, no_rand=False):
        theano_condition_is_here = isinstance(no_rand, tt.Variable)
        if theano_condition_is_here:
            return tt.switch(no_rand,
                             self.mean,
                             self.histogram[self.randidx(size)])
        else:
            if no_rand:
                return self.mean
            else:
                return self.histogram[self.randidx(size)]

    @property
    def histogram(self):
        """
        Shortcut to flattened Trace
        """
        return self.shared_params

    @property
    @memoize
    def histogram_logp(self):
        """
        Symbolic logp for every point in trace
        """
        node = self.to_flat_input(self.model.logpt)

        def mapping(z):
            return theano.clone(node, {self.input: z})
        x = self.histogram
        _histogram_logp, _ = theano.scan(
            mapping, x, n_steps=x.shape[0]
        )
        return _histogram_logp

    @property
    def mean(self):
        return self.histogram.mean(0)

    @property
    def cov(self):
        x = (self.histogram - self.mean)
        return x.T.dot(x) / self.histogram.shape[0]

    @classmethod
    def from_noise(cls, size, jitter=.01, local_rv=None, start=None, model=None, seed=None):
        """
        Initialize Histogram with random noise

        Parameters
        ----------
        size : number of initial particles
        jitter : initial sd
        local_rv : dict
            mapping {model_variable -> local_variable}
            Local Vars are used for Autoencoding Variational Bayes
            See (AEVB; Kingma and Welling, 2014) for details
        start : initial point
        model : pm.Model
            PyMC3 Model

        Returns
        -------
        Empirical
        """
        hist = cls(None, local_rv=local_rv, model=model, seed=seed)
        if start is None:
            start = hist.model.test_point
        start = hist.gbij.map(start)
        # Initialize particles
        x0 = np.tile(start, (size, 1))
        x0 += np.random.normal(0, jitter, x0.shape)
        hist.histogram.set_value(x0)
        return hist


def sample_approx(approx, draws=100, hide_transformed=False):
    """
    Draw samples from variational posterior.

    Parameters
    ----------
    approx : Approximation
    draws : int
        Number of random samples.
    hide_transformed : bool
        If False, transformed variables are also sampled. Default is True.

    Returns
    -------
    trace : pymc3.backends.base.MultiTrace
        Samples drawn from variational posterior.
    """
    if not isinstance(approx, Approximation):
        raise TypeError('Need Approximation instance, got %r' % approx)
    return approx.sample(draws=draws, hide_transformed=hide_transformed)
