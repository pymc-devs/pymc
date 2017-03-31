from __future__ import division

import logging

import numpy as np
import tqdm

import pymc3 as pm
from pymc3.variational.approximations import MeanField, FullRank
from pymc3.variational.operators import KL
from pymc3.variational.opvi import Approximation, TestFunction

logger = logging.getLogger(__name__)

__all__ = [
    'TestFunction',
    'ADVI',
    'FullRankADVI',
    'Inference'
]


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
        elif isinstance(approx, Approximation):    # pragma: no cover
            pass
        else:   # pragma: no cover
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
                    if np.isnan(e):     # pragma: no cover
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
            except KeyboardInterrupt:   # pragma: no cover
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
        else:   # pragma: no cover
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
