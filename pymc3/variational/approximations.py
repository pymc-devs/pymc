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
    """Mean Field approximation to the posterior where spherical Gaussian family
    is fitted to minimize KL divergence from True posterior. It is assumed
    that latent space variables are uncorrelated that is the main drawback
    of the method

    Parameters
    ----------
    local_rv : dict[var->tuple]
        mapping {model_variable -> local_variable (:math:`\\mu`, :math:`\\rho`)}
        Local Vars are used for Autoencoding Variational Bayes
        See (AEVB; Kingma and Welling, 2014) for details
    model : :class:`pymc3.Model`
        PyMC3 model for inference
    start : `Point`
        initial mean
    cost_part_grad_scale : `scalar`
        Scaling score part of gradient can be useful near optimum for
        archiving better convergence properties. Common schedule is
        1 at the start and 0 in the end. So slow decay will be ok.
        See (Sticking the Landing; Geoffrey Roeder,
        Yuhuai Wu, David Duvenaud, 2016) for details
    scale_cost_to_minibatch : `bool`
        Scale cost to minibatch instead of full dataset, default False
    random_seed : None or int
        leave None to use package global RandomStream or other
        valid value to create instance specific one

    References
    ----------
    -   Geoffrey Roeder, Yuhuai Wu, David Duvenaud, 2016
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

    @property
    def std(self):
        return rho2sd(self.rho)

    def create_shared_params(self, **kwargs):
        start = kwargs.get('start')
        if start is None:
            start = self.model.test_point
        else:
            start_ = self.model.test_point.copy()
            pm.sampling._update_start_vals(start_, start, self.model)
            start = start_
        start = self.gbij.map(start)
        return {'mu': theano.shared(
                    pm.floatX(start), 'mu'),
                'rho': theano.shared(
                    pm.floatX(np.zeros((self.global_size,))), 'rho')}

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
    """Full Rank approximation to the posterior where Multivariate Gaussian family
    is fitted to minimize KL divergence from True posterior. In contrast to
    MeanField approach correlations between variables are taken in account. The
    main drawback of the method is computational cost.

    Parameters
    ----------
    local_rv : dict[var->tuple]
        mapping {model_variable -> local_variable (:math:`\\mu`, :math:`\\rho`)}
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
    scale_cost_to_minibatch : bool, default False
        Scale cost to minibatch instead of full dataset
    random_seed : None or int
        leave None to use package global RandomStream or other
        valid value to create instance specific one

    Other Parameters
    ----------------
    gpu_compat : bool
        use GPU compatible version or not

    References
    ----------
    -   Geoffrey Roeder, Yuhuai Wu, David Duvenaud, 2016
        Sticking the Landing: A Simple Reduced-Variance Gradient for ADVI
        approximateinference.org/accepted/RoederEtAl2016.pdf
    """

    def __init__(self, local_rv=None, model=None, cost_part_grad_scale=1,
                 scale_cost_to_minibatch=False,
                 gpu_compat=False, random_seed=None, **kwargs):
        super(FullRank, self).__init__(
            local_rv=local_rv, model=model,
            cost_part_grad_scale=cost_part_grad_scale,
            scale_cost_to_minibatch=scale_cost_to_minibatch,
            random_seed=random_seed, **kwargs
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
        tril_index_matrix[
            np.tril_indices(n)[::-1]
        ] = np.arange(num_tril_entries)
        return tril_index_matrix

    def create_shared_params(self, **kwargs):
        start = kwargs.get('start')
        if start is None:
            start = self.model.test_point
        else:
            start_ = self.model.test_point.copy()
            pm.sampling._update_start_vals(start_, start, self.model)
            start = start_
        start = pm.floatX(self.gbij.map(start))
        n = self.global_size
        L_tril = (
            np.eye(n)
            [np.tril_indices(n)]
            .astype(theano.config.floatX)
        )
        return {'mu': theano.shared(start, 'mu'),
                'L_tril': theano.shared(L_tril, 'L_tril')
                }

    def log_q_W_global(self, z):
        """log_q_W samples over q for global vars
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
        """Construct FullRank from MeanField approximation

        Parameters
        ----------
        mean_field : :class:`MeanField`
            approximation to start with

        Other Parameters
        ----------------
        gpu_compat : `bool`
            use GPU compatible version or not

        Returns
        -------
        :class:`FullRank`
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
    """Builds Approximation instance from a given trace,
    it has the same interface as variational approximation

    Parameters
    ----------
    trace : :class:`MultiTrace`
        Trace storing samples (e.g. from step methods)
    local_rv : dict[var->tuple]
        Experimental for Empirical Approximation
        mapping {model_variable -> local_variable (:math:`\\mu`, :math:`\\rho`)}
        Local Vars are used for Autoencoding Variational Bayes
        See (AEVB; Kingma and Welling, 2014) for details
    scale_cost_to_minibatch : `bool`
        Scale cost to minibatch instead of full dataset, default False
    model : :class:`pymc3.Model`
        PyMC3 model for inference
    random_seed : None or int
        leave None to use package global RandomStream or other
        valid value to create instance specific one

    Examples
    --------
    >>> with model:
    ...     step = NUTS()
    ...     trace = sample(1000, step=step)
    ...     histogram = Empirical(trace[100:])
    """

    def __init__(self, trace, local_rv=None,
                 scale_cost_to_minibatch=False,
                 model=None, random_seed=None, **kwargs):
        super(Empirical, self).__init__(
            local_rv=local_rv, scale_cost_to_minibatch=scale_cost_to_minibatch,
            model=model, trace=trace, random_seed=random_seed, **kwargs)

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
                .uniform(size=size,
                         low=pm.floatX(0),
                         high=pm.floatX(self.histogram.shape[0]) - pm.floatX(1e-16))
                .astype('int32'))

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
        """Shortcut to flattened Trace
        """
        return self.shared_params

    @property
    @memoize
    def histogram_logp(self):
        """Symbolic logp for every point in trace
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
    def from_noise(cls, size, jitter=.01, local_rv=None,
                   start=None, model=None, random_seed=None, **kwargs):
        """Initialize Histogram with random noise

        Parameters
        ----------
        size : `int`
            number of initial particles
        jitter : `float`
            initial sd
        local_rv : `dict`
            mapping {model_variable -> local_variable}
            Local Vars are used for Autoencoding Variational Bayes
            See (AEVB; Kingma and Welling, 2014) for details
        start : `Point`
            initial point
        model : :class:`pymc3.Model`
            PyMC3 model for inference
        random_seed : None or `int`
            leave None to use package global RandomStream or other
            valid value to create instance specific one
        kwargs : other kwargs passed to init

        Returns
        -------
        :class:`Empirical`
        """
        hist = cls(
            None,
            local_rv=local_rv,
            model=model,
            random_seed=random_seed,
            **kwargs)
        if start is None:
            start = hist.model.test_point
        else:
            start_ = hist.model.test_point.copy()
            pm.sampling._update_start_vals(start_, start, hist.model)
            start = start_
        start = pm.floatX(hist.gbij.map(start))
        # Initialize particles
        x0 = np.tile(start, (size, 1))
        x0 += pm.floatX(np.random.normal(0, jitter, x0.shape))
        hist.histogram.set_value(x0)
        return hist


def sample_approx(approx, draws=100, include_transformed=True):
    """Draw samples from variational posterior.

    Parameters
    ----------
    approx : :class:`Approximation`
        Approximation to sample from
    draws : `int`
        Number of random samples.
    include_transformed : `bool`
        If True, transformed variables are also sampled. Default is True.

    Returns
    -------
    trace : class:`pymc3.backends.base.MultiTrace`
        Samples drawn from variational posterior.
    """
    if not isinstance(approx, Approximation):
        raise TypeError('Need Approximation instance, got %r' % approx)
    return approx.sample(draws=draws, include_transformed=include_transformed)
