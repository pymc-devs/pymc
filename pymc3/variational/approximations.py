import numpy as np
import theano
from theano import tensor as tt

import pymc3 as pm
from pymc3.distributions.dist_math import rho2sd, log_normal
from pymc3.variational.opvi import Approximation, node_property
from pymc3.util import update_start_vals
from pymc3.variational import flows


__all__ = [
    'MeanField',
    'FullRank',
    'Empirical',
    'NormalizingFlow',
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

    def __init__(self, local_rv=None, model=None, cost_part_grad_scale=1,
                 scale_cost_to_minibatch=False,
                 random_seed=None, start=None, **kwargs):
        super(MeanField, self).__init__(
            local_rv=local_rv, model=model,
            cost_part_grad_scale=cost_part_grad_scale,
            scale_cost_to_minibatch=scale_cost_to_minibatch,
            random_seed=random_seed, **kwargs
        )
        self.shared_params = self.create_shared_params(start=start)

    @node_property
    def mean(self):
        return self.shared_params['mu']

    @node_property
    def rho(self):
        return self.shared_params['rho']

    @node_property
    def cov(self):
        return tt.diag(rho2sd(self.rho)**2)

    @node_property
    def std(self):
        return rho2sd(self.rho)

    def create_shared_params(self, start=None):
        if start is None:
            start = self.model.test_point
        else:
            start_ = self.model.test_point.copy()
            update_start_vals(start_, start, self.model)
            start = start_
        start = self.gbij.map(start)
        return {'mu': theano.shared(
                    pm.floatX(start), 'mu'),
                'rho': theano.shared(
                    pm.floatX(np.zeros((self.global_size,))), 'rho')}

    @node_property
    def symbolic_random_global_matrix(self):
        initial = self.symbolic_initial_global_matrix
        sd = rho2sd(self.rho)
        mu = self.mean
        return sd * initial + mu

    @node_property
    def symbolic_log_q_W_global(self):
        """
        log_q_W samples over q for global vars
        """
        mu = self.scale_grad(self.mean)
        rho = self.scale_grad(self.rho)
        z = self.symbolic_random_global_matrix
        logq = log_normal(z, mu, rho=rho)
        return logq.sum(1)


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
                 gpu_compat=False, random_seed=None, start=None, **kwargs):
        super(FullRank, self).__init__(
            local_rv=local_rv, model=model,
            cost_part_grad_scale=cost_part_grad_scale,
            scale_cost_to_minibatch=scale_cost_to_minibatch,
            random_seed=random_seed, **kwargs
        )
        self.gpu_compat = gpu_compat
        self.shared_params = self.create_shared_params(start=start)

    def create_shared_params(self, start=None):
        if start is None:
            start = self.model.test_point
        else:
            start_ = self.model.test_point.copy()
            update_start_vals(start_, start, self.model)
            start = start_
        start = pm.floatX(self.gbij.map(start))
        n = self.global_size
        L_tril = (
            np.eye(n)
            [np.tril_indices(n)]
            .astype(theano.config.floatX)
        )
        return {'mu': theano.shared(start, 'mu'),
                'L_tril': theano.shared(L_tril, 'L_tril')}

    @node_property
    def L(self):
        return self.shared_params['L_tril'][self.tril_index_matrix]

    @node_property
    def mean(self):
        return self.shared_params['mu']

    @node_property
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

    @node_property
    def symbolic_log_q_W_global(self):
        """log_q_W samples over q for global vars
        """
        mu = self.scale_grad(self.mean)
        L = self.scale_grad(self.L)
        z = self.symbolic_random_global_matrix
        return pm.MvNormal.dist(mu=mu, chol=L).logp(z)

    @node_property
    def symbolic_random_global_matrix(self):
        # (samples, dim) or (dim, )
        initial = self.symbolic_initial_global_matrix.T
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
        self.shared_params = self.create_shared_params(trace=trace)

    def create_shared_params(self, trace=None):
        if trace is None:
            histogram = np.atleast_2d(self.gbij.map(self.model.test_point))
        else:
            histogram = np.empty((len(trace) * len(trace.chains), self.global_size))
            i = 0
            for t in trace.chains:
                for j in range(len(trace)):
                    histogram[i] = self.gbij.map(trace.point(j, t))
                    i += 1
        return dict(histogram=theano.shared(pm.floatX(histogram), 'histogram'))

    def check_model(self, model, **kwargs):
        trace = kwargs.get('trace')
        if (trace is not None
            and not all([var.name in trace.varnames
                         for var in model.free_RVs])):
            raise ValueError('trace has not all FreeRV')

    def randidx(self, size=None):
        if size is None:
            size = (1,)
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

    def _initial_part_matrix(self, part, size, deterministic):
        if part == 'local':
            return super(Empirical, self)._initial_part_matrix(
                part, size, deterministic
            )
        elif part == 'global':
            theano_condition_is_here = isinstance(deterministic, tt.Variable)
            if theano_condition_is_here:
                return tt.switch(
                    deterministic,
                    tt.repeat(
                        self.mean.dimshuffle('x', 0),
                        size if size is not None else 1, -1),
                    self.histogram[self.randidx(size)])
            else:
                if deterministic:
                    return tt.repeat(
                        self.mean.dimshuffle('x', 0),
                        size if size is not None else 1, -1)
                else:
                    return self.histogram[self.randidx(size)]

    @property
    def symbolic_random_global_matrix(self):
        return self.symbolic_initial_global_matrix

    @property
    def histogram(self):
        """Shortcut to flattened Trace
        """
        return self.shared_params['histogram']

    @node_property
    def mean(self):
        return self.histogram.mean(0)

    @node_property
    def cov(self):
        x = (self.histogram - self.mean)
        return x.T.dot(x) / pm.floatX(self.histogram.shape[0])

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
            update_start_vals(start_, start, hist.model)
            start = start_
        start = pm.floatX(hist.gbij.map(start))
        # Initialize particles
        x0 = np.tile(start, (size, 1))
        x0 += pm.floatX(np.random.normal(0, jitter, x0.shape))
        hist.histogram.set_value(x0)
        return hist


class NormalizingFlow(Approximation):
    R"""
    Normalizing flow is a series of invertible transformations on initial distribution.

    .. math::

        z_K = f_K \circ \dots \circ f_2 \circ f_1(z_0)

    In that case we can compute tractable density for the flow.

    .. math::

        \ln q_K(z_K) = \ln q_0(z_0) - \sum_{k=1}^{K}\ln \left|\frac{\partial f_k}{\partial z_{k-1}}\right|


    Every :math:`f_k` here is a parametric function with defined determinant.
    We can choose every step here. For example the here is a simple flow
    is an affine transform:

    .. math::

        z = loc(scale(z_0)) = \mu + \sigma * z_0

    Here we get mean field approximation if :math:`z_0 \sim \mathcal{N}(0, 1)`

    **Flow Formulas**

    In PyMC3 there is a flexible way to define flows with formulas. We have 5 of them by the moment:

    -   Loc (:code:`loc`): :math:`z' = z + \mu`
    -   Scale (:code:`scale`): :math:`z' = \sigma * z`
    -   Planar (:code:`planar`): :math:`z' = z + u * \tanh(w^T z + b)`
    -   Radial (:code:`radial`): :math:`z' = z + \beta (\alpha + (z-z_r))^{-1}(z-z_r)`
    -   Householder (:code:`hh`): :math:`z' = H z`

    Formula can be written as a string, e.g. `'scale-loc'`, `'scale-hh*4-loc'`, `'panar*10'`.
    Every step is separated with `'-'`, repeated flow is marked with `'*'` produsing `'flow*repeats'`.

    Parameters
    ----------
    flow : str|AbstractFlow
        formula or initialized Flow, default is `'scale-loc'` that
        is identical to MeanField
    local_rv : dict[var->tuple]
        Experimental for Empirical Approximation
        mapping {model_variable -> local_variable (:math:`\mu`, :math:`\rho`)}
        Local Vars are used for Autoencoding Variational Bayes
        See (AEVB; Kingma and Welling, 2014) for details
    scale_cost_to_minibatch : `bool`
        Scale cost to minibatch instead of full dataset, default False
    model : :class:`pymc3.Model`
        PyMC3 model for inference
    random_seed : None or int
        leave None to use package global RandomStream or other
        valid value to create instance specific one
    jitter : float
        noise for flows' parameters initialization

    References
    ----------
    -   Danilo Jimenez Rezende, Shakir Mohamed, 2015
        Variational Inference with Normalizing Flows
        arXiv:1505.05770

    -   Jakub M. Tomczak, Max Welling, 2016
        Improving Variational Auto-Encoders using Householder Flow
        arXiv:1611.09630
    """

    def __init__(self, flow='scale-loc',
                 local_rv=None, model=None,
                 scale_cost_to_minibatch=False,
                 random_seed=None, jitter=.1, **kwargs):
        super(NormalizingFlow, self).__init__(
            local_rv=local_rv, scale_cost_to_minibatch=scale_cost_to_minibatch,
            model=model, random_seed=random_seed, **kwargs)
        if isinstance(flow, str):
            flow = flows.Formula(flow)(
                dim=self.global_size,
                z0=self.symbolic_initial_global_matrix,
                jitter=jitter
            )
        self.gflow = flow

    @property
    def shared_params(self):
        params = dict()
        current = self.gflow
        i = 0
        params[i] = current.shared_params
        while not current.isroot:
            i += 1
            current = current.parent
            params[i] = current.shared_params
        return params

    @shared_params.setter
    def shared_params(self, value):
        current = self.gflow
        i = 0
        current.shared_params = value[i]
        while not current.isroot:
            i += 1
            current = current.parent
            current.shared_params = value[i]

    @property
    def params(self):
        return self.gflow.all_params

    @node_property
    def symbolic_log_q_W_global(self):
        z0 = self.symbolic_initial_global_matrix
        q0 = pm.Normal.dist().logp(z0).sum(-1)
        return q0-self.gflow.sum_logdets

    @property
    def symbolic_random_global_matrix(self):
        return self.gflow.forward


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
