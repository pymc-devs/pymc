from __future__ import division

import logging

import numpy as np
import theano
from theano import tensor as tt
import tqdm

import pymc3 as pm
from pymc3.distributions.dist_math import log_normal, rho2sd, log_normal_mv
from pymc3.variational.opvi import Operator, Approximation, TestFunction

logger = logging.getLogger(__name__)

__all__ = [
    'TestFunction',
    'KL',
    'MeanField',
    'FullRank',
    'ADVI',
    'FullRankADVI',
    'Inference'
]
# OPERATORS


class KL(Operator):
    """
    Operator based on Kullback Leibler Divergence
    .. math::

        KL[q(v)||p(v)] = \int q(v)\log\\frac{q(v)}{p(v)}dv
    """
    def apply(self, f):
        z = self.input
        return self.logq(z) - self.logp(z)

# APPROXIMATIONS


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

    cost_part_grad_scale : float or scalar tensor
        Scaling score part of gradient can be useful near optimum for
        archiving better convergence properties. Common schedule is
        1 at the start and 0 in the end. So slow decay will be ok.
        See (Sticking the Landing; Geoffrey Roeder,
        Yuhuai Wu, David Duvenaud, 2016) for details

    References
    ----------
    Geoffrey Roeder, Yuhuai Wu, David Duvenaud, 2016
        Sticking the Landing: A Simple Reduced-Variance Gradient for ADVI
        approximateinference.org/accepted/RoederEtAl2016.pdf
    """
    @property
    def mu(self):
        return self.shared_params['mu']

    @property
    def rho(self):
        return self.shared_params['rho']

    @property
    def cov(self):
        return tt.diag(rho2sd(self.rho))

    def create_shared_params(self):
        return {'mu': theano.shared(
                    self.input.tag.test_value[self.global_slc]),
                'rho': theano.shared(
                    np.zeros((self.global_size,), dtype=theano.config.floatX))
                }

    def log_q_W_global(self, z):
        """
        log_q_W samples over q for global vars
        Gradient wrt mu, rho in density parametrization
        is set to zero to lower variance of ELBO
        """
        mu = self.scale_grad(self.mu)
        rho = self.scale_grad(self.rho)
        z = z[self.global_slc]
        logq = tt.sum(log_normal(z, mu, rho=rho))
        return logq

    def random_global(self, size=None, no_rand=False):
        initial = self.initial(size, no_rand, l=self.global_size)
        sd = rho2sd(self.rho)
        mu = self.mu
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

    cost_part_grad_scale : float or scalar tensor
        Scaling score part of gradient can be useful near optimum for
        archiving better convergence properties. Common schedule is
        1 at the start and 0 in the end. So slow decay will be ok.
        See (Sticking the Landing; Geoffrey Roeder,
        Yuhuai Wu, David Duvenaud, 2016) for details

    References
    ----------
    Geoffrey Roeder, Yuhuai Wu, David Duvenaud, 2016
        Sticking the Landing: A Simple Reduced-Variance Gradient for ADVI
        approximateinference.org/accepted/RoederEtAl2016.pdf
    """
    def __init__(self, local_rv=None, model=None, cost_part_grad_scale=1, gpu_compat=False):
        super(FullRank, self).__init__(
            local_rv=local_rv, model=model,
            cost_part_grad_scale=cost_part_grad_scale
        )
        self.gpu_compat = gpu_compat

    @property
    def L(self):
        return self.shared_params['L_tril'][self.tril_index_matrix]

    @property
    def mu(self):
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

    def create_shared_params(self):
        n = self.global_size
        L_tril = (
            np.eye(n)
            [np.tril_indices(n)]
            .astype(theano.config.floatX)
        )
        return {'mu': theano.shared(
                    self.input.tag.test_value[self.global_slc]),
                'L_tril': theano.shared(L_tril)
                }

    def log_q_W_global(self, z):
        """
        log_q_W samples over q for global vars
        Gradient wrt mu, rho in density parametrization
        is set to zero to lower variance of ELBO
        """
        mu = self.scale_grad(self.mu)
        L = self.scale_grad(self.L)
        z = z[self.global_slc]
        return log_normal_mv(z, mu, chol=L, gpu_compat=self.gpu_compat)

    def random_global(self, size=None, no_rand=False):
        # (samples, dim) or (dim, )
        initial = self.initial(size, no_rand, l=self.global_size).T
        # (dim, dim)
        L = self.L
        # (dim, )
        mu = self.mu
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


class Inference(object):
    """
    Base class for Variational Inference

    Communicates Operator, Approximation and Test Function to build Objective Function

    Parameters
    ----------
    op : Operator class
    approx : Approximation class or instance
    tf : TestFunction instance
    local_rv : list
    model : PyMC3 Model
    kwargs : kwargs for Approximation
    """
    def __init__(self, op, approx, tf, local_rv=None, model=None, **kwargs):
        self.hist = np.asarray(())
        if isinstance(approx, type) and issubclass(approx, Approximation):
            approx = approx(
                local_rv=local_rv,
                model=model, **kwargs)
        elif isinstance(approx, Approximation):
            pass
        else:
            raise TypeError('approx should be Approximation instance or Approximation subclass')
        self.objective = op(approx)(tf)

    approx = property(lambda self: self.objective.approx)

    def run_profiling(self, n=1000, score=True, **kwargs):
        fn_kwargs = kwargs.pop('fn_kwargs', dict())
        fn_kwargs.update(profile=True)
        step_func = self.objective.step_function(
            score=score, fn_kwargs=fn_kwargs,
            **kwargs
        )
        progress = tqdm.trange(n)
        try:
            for _ in progress:
                step_func()
        except KeyboardInterrupt:
            pass
        finally:
            progress.close()
        return step_func.profile

    def fit(self, n=10000, score=True, callbacks=None, callback_every=1,
            **kwargs):
        """
        Performs Operator Variational Inference

        Parameters
        ----------
        n : int
            number of iterations
        score : bool
            evaluate loss on each iteration or not
        callbacks : list[function : (Approximation, losses, i) -> any]
        callback_every : int
            call callback functions on `callback_every` step
        kwargs : kwargs for ObjectiveFunction.step_function

        Returns
        -------
        Approximation
        """
        if callbacks is None:
            callbacks = []
        step_func = self.objective.step_function(score=score, **kwargs)
        i = 0
        scores = np.empty(n)
        scores[:] = np.nan
        progress = tqdm.trange(n)
        if score:
            try:
                for i in progress:
                    e = step_func()
                    if np.isnan(e):
                        scores = scores[:i]
                        self.hist = np.concatenate([self.hist, scores])
                        raise FloatingPointError('NaN occurred in optimization.')
                    scores[i] = e
                    if i % 10 == 0:
                        avg_elbo = scores[max(0, i - 1000):i+1].mean()
                        progress.set_description('Average Loss = {:,.5g}'.format(avg_elbo))
                    if i % callback_every == 0:
                        for callback in callbacks:
                            callback(self.approx, scores[:i+1], i)
            except KeyboardInterrupt:
                scores = scores[:i]
                if n < 10:
                    logger.info('Interrupted at {:,d} [{:.0f}%]: Loss = {:,.5g}'.format(
                        i, 100 * i // n, scores[i]))
                else:
                    avg_elbo = scores[min(0, i - 1000):i].mean()
                    logger.info('Interrupted at {:,d} [{:.0f}%]: Average Loss = {:,.5g}'.format(
                        i, 100 * i // n, avg_elbo))
            else:
                if n < 10:
                    logger.info('Finished [100%]: Loss = {:,.5g}'.format(scores[-1]))
                else:
                    avg_elbo = scores[max(0, i - 1000):i].mean()
                    logger.info('Finished [100%]: Average Loss = {:,.5g}'.format(avg_elbo))
            finally:
                progress.close()
        else:
            try:
                for _ in progress:
                    step_func()
            except KeyboardInterrupt:
                pass
            finally:
                progress.close()
        self.hist = np.concatenate([self.hist, scores])
        return self.approx


class ADVI(Inference):
    """
    Automatic Differentiation Variational Inference (ADVI)

    Parameters
    ----------
    local_rv : dict
        mapping {model_variable -> local_variable}
        Local Vars are used for Autoencoding Variational Bayes
        See (AEVB; Kingma and Welling, 2014) for details

    model : PyMC3 model for inference

    cost_part_grad_scale : float or scalar tensor
        Scaling score part of gradient can be useful near optimum for
        archiving better convergence properties. Common schedule is
        1 at the start and 0 in the end. So slow decay will be ok.
        See (Sticking the Landing; Geoffrey Roeder,
        Yuhuai Wu, David Duvenaud, 2016) for details

    References
    ----------
    - Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A.,
        and Blei, D. M. (2016). Automatic Differentiation Variational
        Inference. arXiv preprint arXiv:1603.00788.

    - Geoffrey Roeder, Yuhuai Wu, David Duvenaud, 2016
        Sticking the Landing: A Simple Reduced-Variance Gradient for ADVI
        approximateinference.org/accepted/RoederEtAl2016.pdf

    - Kingma, D. P., & Welling, M. (2014).
      Auto-Encoding Variational Bayes. stat, 1050, 1.
    """
    def __init__(self, local_rv=None, model=None, cost_part_grad_scale=1):
        super(ADVI, self).__init__(
            KL, MeanField, None,
            local_rv=local_rv, model=model, cost_part_grad_scale=cost_part_grad_scale)


class FullRankADVI(Inference):
    """
    Full Rank Automatic Differentiation Variational Inference (ADVI)

    Parameters
    ----------
    local_rv : dict
        mapping {model_variable -> local_variable}
        Local Vars are used for Autoencoding Variational Bayes
        See (AEVB; Kingma and Welling, 2014) for details

    model : PyMC3 model for inference

    cost_part_grad_scale : float or scalar tensor
        Scaling score part of gradient can be useful near optimum for
        archiving better convergence properties. Common schedule is
        1 at the start and 0 in the end. So slow decay will be ok.
        See (Sticking the Landing; Geoffrey Roeder,
        Yuhuai Wu, David Duvenaud, 2016) for details

    References
    ----------
    - Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A.,
        and Blei, D. M. (2016). Automatic Differentiation Variational
        Inference. arXiv preprint arXiv:1603.00788.

    - Geoffrey Roeder, Yuhuai Wu, David Duvenaud, 2016
        Sticking the Landing: A Simple Reduced-Variance Gradient for ADVI
        approximateinference.org/accepted/RoederEtAl2016.pdf

    - Kingma, D. P., & Welling, M. (2014).
      Auto-Encoding Variational Bayes. stat, 1050, 1.
    """
    def __init__(self, local_rv=None, model=None, cost_part_grad_scale=1, gpu_compat=False):
        super(FullRankADVI, self).__init__(
            KL, FullRank, None,
            local_rv=local_rv, model=model, cost_part_grad_scale=cost_part_grad_scale, gpu_compat=gpu_compat)

    @classmethod
    def from_mean_field(cls, mean_field, gpu_compat=False):
        """
        Construct FullRankADVI from MeanField approximation

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
        FullRankADVI
        """
        full_rank = FullRank.from_mean_field(mean_field, gpu_compat)
        inference = object.__new__(cls)
        objective = KL(full_rank)(None)
        inference.objective = objective
        inference.hist = np.asarray(())
        return inference

    @classmethod
    def from_advi(cls, advi, gpu_compat=False):
        """
        Construct FullRankADVI from ADVI

        Parameters
        ----------
        advi : ADVI

        Flags
        -----
        gpu_compat : bool
            use GPU compatible version or not

        Returns
        -------
        FullRankADVI
        """
        inference = cls.from_mean_field(advi.approx, gpu_compat)
        inference.hist = advi.hist
        return inference


def fit(n=10000, local_rv=None, method='advi', model=None, **kwargs):
    """
    Handy shortcut for using inference methods in functional way

    Parameters
    ----------
    n : int
        number of iterations
    local_rv : dict
        mapping {model_variable -> local_variable}
        Local Vars are used for Autoencoding Variational Bayes
        See (AEVB; Kingma and Welling, 2014) for details
    method : str or Inference
        string name is case insensitive in {'advi', 'fullrank_advi', 'advi->fullrank_advi'}
    model : None or Model
    frac : float
        if method is 'advi->fullrank_advi' represents advi fraction when training
    kwargs : kwargs for Inference.fit

    Returns
    -------
    Approximation
    """
    if model is None:
        model = pm.modelcontext(model)
    _select = dict(
        advi=ADVI,
        fullrank_advi=FullRankADVI,
    )
    if isinstance(method, str) and method.lower() == 'advi->fullrank_advi':
        frac = kwargs.pop('frac', .5)
        if not 0. < frac < 1.:
            raise ValueError('frac should be in (0, 1)')
        n1 = int(n * frac)
        n2 = n-n1
        inference = ADVI(local_rv=local_rv, model=model)
        logger.info('fitting advi ...')
        inference.fit(n1, **kwargs)
        inference = FullRankADVI.from_advi(inference)
        logger.info('fitting fullrank advi ...')
        return inference.fit(n2, **kwargs)

    elif isinstance(method, str):
        try:
            inference = _select[method.lower()](
                local_rv=local_rv, model=model
            )
        except KeyError:
            raise KeyError('method should be one of %s '
                           'or Inference instance' %
                           set(_select.keys()))
    elif isinstance(method, Inference):
        inference = method
    else:
        raise TypeError('method should be one of %s '
                        'or Inference instance' %
                        set(_select.keys()))
    return inference.fit(n, **kwargs)
