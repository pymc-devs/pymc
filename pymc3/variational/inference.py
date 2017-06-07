from __future__ import division

import logging
import warnings

import numpy as np
import tqdm

import pymc3 as pm
from pymc3.variational import test_functions
from pymc3.variational.approximations import MeanField, FullRank, Empirical
from pymc3.variational.operators import KL, KSD, AKSD
from pymc3.variational.opvi import Approximation

logger = logging.getLogger(__name__)

__all__ = [
    'ADVI',
    'FullRankADVI',
    'SVGD',
    'ASVGD',
    'Inference',
    'fit'
]


class Inference(object):
    R"""
    Base class for Variational Inference

    Communicates Operator, Approximation and Test Function to build Objective Function

    Parameters
    ----------
    op : Operator class
    approx : Approximation class or instance
    tf : TestFunction instance
    local_rv : dict
        mapping {model_variable -> local_variable}
        Local Vars are used for Autoencoding Variational Bayes
        See (AEVB; Kingma and Welling, 2014) for details
    model : Model
        PyMC3 Model
    op_kwargs : dict
        kwargs passed to :class:`Operator`
    kwargs : kwargs
        additional kwargs for :class:`Approximation`
    """

    def __init__(self, op, approx, tf, local_rv=None, model=None, op_kwargs=None, **kwargs):
        if op_kwargs is None:
            op_kwargs = dict()
        self.hist = np.asarray(())
        if isinstance(approx, type) and issubclass(approx, Approximation):
            approx = approx(
                local_rv=local_rv,
                model=model, **kwargs)
        elif isinstance(approx, Approximation):    # pragma: no cover
            pass
        else:   # pragma: no cover
            raise TypeError(
                'approx should be Approximation instance or Approximation subclass')
        self.objective = op(approx, **op_kwargs)(tf)

    approx = property(lambda self: self.objective.approx)

    def _maybe_score(self, score):
        returns_loss = self.objective.op.RETURNS_LOSS
        if score is None:
            score = returns_loss
        elif score and not returns_loss:
            warnings.warn('method `fit` got `score == True` but %s '
                          'does not return loss. Ignoring `score` argument'
                          % self.objective.op)
            score = False
        else:
            pass
        return score

    def run_profiling(self, n=1000, score=None, **kwargs):
        score = self._maybe_score(score)
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

    def fit(self, n=10000, score=None, callbacks=None, progressbar=True,
            **kwargs):
        """
        Performs Operator Variational Inference

        Parameters
        ----------
        n : int
            number of iterations
        score : bool
            evaluate loss on each iteration or not
        callbacks : list[function : (Approximation, losses, i) -> None]
            calls provided functions after each iteration step
        progressbar : bool
            whether to show progressbar or not
        kwargs : kwargs
            additional kwargs for :func:`ObjectiveFunction.step_function`

        Returns
        -------
        Approximation
        """
        if callbacks is None:
            callbacks = []
        score = self._maybe_score(score)
        step_func = self.objective.step_function(score=score, **kwargs)
        progress = tqdm.trange(n, disable=not progressbar)
        if score:
            self._iterate_with_loss(n, step_func, progress, callbacks)
        else:
            self._iterate_without_loss(n, step_func, progress, callbacks)

        # hack to allow pm.fit() access to loss hist
        self.approx.hist = self.hist

        return self.approx

    def _iterate_without_loss(self, _, step_func, progress, callbacks):
        try:
            for i in progress:
                step_func()
                if np.isnan(self.approx.params[0].get_value()).any():
                    raise FloatingPointError('NaN occurred in optimization.')
                for callback in callbacks:
                    callback(self.approx, None, i)
        except (KeyboardInterrupt, StopIteration) as e:
            progress.close()
            if isinstance(e, StopIteration):
                logger.info(str(e))
        finally:
            progress.close()

    def _iterate_with_loss(self, n, step_func, progress, callbacks):
        def _infmean(input_array):
            """Return the mean of the finite values of the array"""
            input_array = input_array[np.isfinite(input_array)].astype('float64')
            if len(input_array) == 0:
                return np.nan
            else:
                return np.mean(input_array)
        scores = np.empty(n)
        scores[:] = np.nan
        i = 0
        try:
            for i in progress:
                e = step_func()
                if np.isnan(e):  # pragma: no cover
                    scores = scores[:i]
                    self.hist = np.concatenate([self.hist, scores])
                    raise FloatingPointError('NaN occurred in optimization.')
                scores[i] = e
                if i % 10 == 0:
                    avg_loss = _infmean(scores[max(0, i - 1000):i + 1])
                    progress.set_description('Average Loss = {:,.5g}'.format(avg_loss))
                    avg_loss = scores[max(0, i - 1000):i + 1].mean()
                    progress.set_description(
                        'Average Loss = {:,.5g}'.format(avg_loss))
                for callback in callbacks:
                    callback(self.approx, scores[:i + 1], i)
        except (KeyboardInterrupt, StopIteration) as e:
            # do not print log on the same line
            progress.close()
            scores = scores[:i]
            if isinstance(e, StopIteration):
                logger.info(str(e))
            if n < 10:
                logger.info('Interrupted at {:,d} [{:.0f}%]: Loss = {:,.5g}'.format(
                    i, 100 * i // n, scores[i]))
            else:
                avg_loss = _infmean(scores[min(0, i - 1000):i + 1])
                logger.info('Interrupted at {:,d} [{:.0f}%]: Average Loss = {:,.5g}'.format(
                    i, 100 * i // n, avg_loss))
        else:
            if n < 10:
                logger.info(
                    'Finished [100%]: Loss = {:,.5g}'.format(scores[-1]))
            else:
                avg_loss = _infmean(scores[max(0, i - 1000):i + 1])
                logger.info(
                    'Finished [100%]: Average Loss = {:,.5g}'.format(avg_loss))
        finally:
            progress.close()
        self.hist = np.concatenate([self.hist, scores])


class ADVI(Inference):
    R"""
    Automatic Differentiation Variational Inference (ADVI)

    This class implements the meanfield ADVI, where the variational
    posterior distribution is assumed to be spherical Gaussian without
    correlation of parameters and fit to the true posterior distribution.
    The means and standard deviations of the variational posterior are referred
    to as variational parameters.

    For explanation, we classify random variables in probabilistic models into
    three types. Observed random variables
    :math:`{\cal Y}=\{\mathbf{y}_{i}\}_{i=1}^{N}` are :math:`N` observations.
    Each :math:`\mathbf{y}_{i}` can be a set of observed random variables,
    i.e., :math:`\mathbf{y}_{i}=\{\mathbf{y}_{i}^{k}\}_{k=1}^{V_{o}}`, where
    :math:`V_{k}` is the number of the types of observed random variables
    in the model.

    The next ones are global random variables
    :math:`\Theta=\{\theta^{k}\}_{k=1}^{V_{g}}`, which are used to calculate
    the probabilities for all observed samples.

    The last ones are local random variables
    :math:`{\cal Z}=\{\mathbf{z}_{i}\}_{i=1}^{N}`, where
    :math:`\mathbf{z}_{i}=\{\mathbf{z}_{i}^{k}\}_{k=1}^{V_{l}}`.
    These RVs are used only in AEVB.

    The goal of ADVI is to approximate the posterior distribution
    :math:`p(\Theta,{\cal Z}|{\cal Y})` by variational posterior
    :math:`q(\Theta)\prod_{i=1}^{N}q(\mathbf{z}_{i})`. All of these terms
    are normal distributions (mean-field approximation).

    :math:`q(\Theta)` is parametrized with its means and standard deviations.
    These parameters are denoted as :math:`\gamma`. While :math:`\gamma` is
    a constant, the parameters of :math:`q(\mathbf{z}_{i})` are dependent on
    each observation. Therefore these parameters are denoted as
    :math:`\xi(\mathbf{y}_{i}; \nu)`, where :math:`\nu` is the parameters
    of :math:`\xi(\cdot)`. For example, :math:`\xi(\cdot)` can be a
    multilayer perceptron or convolutional neural network.

    In addition to :math:`\xi(\cdot)`, we can also include deterministic
    mappings for the likelihood of observations. We denote the parameters of
    the deterministic mappings as :math:`\eta`. An example of such mappings is
    the deconvolutional neural network used in the convolutional VAE example
    in the PyMC3 notebook directory.

    This function maximizes the evidence lower bound (ELBO)
    :math:`{\cal L}(\gamma, \nu, \eta)` defined as follows:

    .. math::

        {\cal L}(\gamma,\nu,\eta) & =
        \mathbf{c}_{o}\mathbb{E}_{q(\Theta)}\left[
        \sum_{i=1}^{N}\mathbb{E}_{q(\mathbf{z}_{i})}\left[
        \log p(\mathbf{y}_{i}|\mathbf{z}_{i},\Theta,\eta)
        \right]\right] \\ &
        - \mathbf{c}_{g}KL\left[q(\Theta)||p(\Theta)\right]
        - \mathbf{c}_{l}\sum_{i=1}^{N}
            KL\left[q(\mathbf{z}_{i})||p(\mathbf{z}_{i})\right],

    where :math:`KL[q(v)||p(v)]` is the Kullback-Leibler divergence

    .. math::

        KL[q(v)||p(v)] = \int q(v)\log\frac{q(v)}{p(v)}dv,

    :math:`\mathbf{c}_{o/g/l}` are vectors for weighting each term of ELBO.
    More precisely, we can write each of the terms in ELBO as follows:

    .. math::

        \mathbf{c}_{o}\log p(\mathbf{y}_{i}|\mathbf{z}_{i},\Theta,\eta) & = &
        \sum_{k=1}^{V_{o}}c_{o}^{k}
            \log p(\mathbf{y}_{i}^{k}|
                   {\rm pa}(\mathbf{y}_{i}^{k},\Theta,\eta)) \\
        \mathbf{c}_{g}KL\left[q(\Theta)||p(\Theta)\right] & = &
        \sum_{k=1}^{V_{g}}c_{g}^{k}KL\left[
            q(\theta^{k})||p(\theta^{k}|{\rm pa(\theta^{k})})\right] \\
        \mathbf{c}_{l}KL\left[q(\mathbf{z}_{i}||p(\mathbf{z}_{i})\right] & = &
        \sum_{k=1}^{V_{l}}c_{l}^{k}KL\left[
            q(\mathbf{z}_{i}^{k})||
            p(\mathbf{z}_{i}^{k}|{\rm pa}(\mathbf{z}_{i}^{k}))\right],

    where :math:`{\rm pa}(v)` denotes the set of parent variables of :math:`v`
    in the directed acyclic graph of the model.

    When using mini-batches, :math:`c_{o}^{k}` and :math:`c_{l}^{k}` should be
    set to :math:`N/M`, where :math:`M` is the number of observations in each
    mini-batch. This is done with supplying `total_size` parameter to
    observed nodes (e.g. :code:`Normal('x', 0, 1, observed=data, total_size=10000)`).
    In this case it is possible to automatically determine appropriate scaling for :math:`logp`
    of observed nodes. Interesting to note that it is possible to have two independent
    observed variables with different `total_size` and iterate them independently
    during inference.

    For working with ADVI, we need to give

    -   The probabilistic model

        `model` with three types of RVs (`observed_RVs`,
        `global_RVs` and `local_RVs`).

    -   (optional) Minibatches

        The tensors to which mini-bathced samples are supplied are
        handled separately by using callbacks in :func:`Inference.fit` method
        that change storage of shared theano variable or by :func:`pymc3.generator`
        that automatically iterates over minibatches and defined beforehand.

    -   (optional) Parameters of deterministic mappings

        They have to be passed along with other params to :func:`Inference.fit` method
        as `more_obj_params` argument.

    For more information concerning training stage please reference
    :func:`pymc3.variational.opvi.ObjectiveFunction.step_function`

    Parameters
    ----------
    local_rv : dict[var->tuple]
        mapping {model_variable -> local_variable (:math:`\mu`, :math:`\rho`)}
        Local Vars are used for Autoencoding Variational Bayes
        See (AEVB; Kingma and Welling, 2014) for details
    model : :class:`pymc3.Model`
        PyMC3 model for inference
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
    start : `Point`
        starting point for inference

    References
    ----------
    -   Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A.,
        and Blei, D. M. (2016). Automatic Differentiation Variational
        Inference. arXiv preprint arXiv:1603.00788.

    -   Geoffrey Roeder, Yuhuai Wu, David Duvenaud, 2016
        Sticking the Landing: A Simple Reduced-Variance Gradient for ADVI
        approximateinference.org/accepted/RoederEtAl2016.pdf

    -   Kingma, D. P., & Welling, M. (2014).
        Auto-Encoding Variational Bayes. stat, 1050, 1.
    """

    def __init__(self, local_rv=None, model=None,
                 cost_part_grad_scale=1,
                 scale_cost_to_minibatch=False,
                 random_seed=None, start=None):
        super(ADVI, self).__init__(
            KL, MeanField, None,
            local_rv=local_rv, model=model,
            cost_part_grad_scale=cost_part_grad_scale,
            scale_cost_to_minibatch=scale_cost_to_minibatch,
            random_seed=random_seed, start=start)

    @classmethod
    def from_mean_field(cls, mean_field):
        """
        Construct ADVI from MeanField approximation

        Parameters
        ----------
        mean_field : :class:`MeanField`
            approximation to start with

        Returns
        -------
        :class:`ADVI`
        """
        if not isinstance(mean_field, MeanField):
            raise TypeError('Expected MeanField, got %r' % mean_field)
        inference = object.__new__(cls)
        objective = KL(mean_field)(None)
        inference.hist = np.asarray(())
        inference.objective = objective
        return inference


class FullRankADVI(Inference):
    R"""
    Full Rank Automatic Differentiation Variational Inference (ADVI)

    Parameters
    ----------
    local_rv : dict[var->tuple]
        mapping {model_variable -> local_variable (:math:`\mu`, :math:`\rho`)}
        Local Vars are used for Autoencoding Variational Bayes
        See (AEVB; Kingma and Welling, 2014) for details
    model : :class:`pymc3.Model`
        PyMC3 model for inference
    cost_part_grad_scale : `scalar`
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
    start : `Point`
        starting point for inference

    References
    ----------
    -   Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A.,
        and Blei, D. M. (2016). Automatic Differentiation Variational
        Inference. arXiv preprint arXiv:1603.00788.

    -   Geoffrey Roeder, Yuhuai Wu, David Duvenaud, 2016
        Sticking the Landing: A Simple Reduced-Variance Gradient for ADVI
        approximateinference.org/accepted/RoederEtAl2016.pdf

    -   Kingma, D. P., & Welling, M. (2014).
        Auto-Encoding Variational Bayes. stat, 1050, 1.
    """

    def __init__(self, local_rv=None, model=None,
                 cost_part_grad_scale=1,
                 scale_cost_to_minibatch=False,
                 gpu_compat=False, random_seed=None, start=None):
        super(FullRankADVI, self).__init__(
            KL, FullRank, None,
            local_rv=local_rv, model=model,
            cost_part_grad_scale=cost_part_grad_scale,
            scale_cost_to_minibatch=scale_cost_to_minibatch,
            gpu_compat=gpu_compat, random_seed=random_seed, start=start)

    @classmethod
    def from_full_rank(cls, full_rank):
        """
        Construct FullRankADVI from FullRank approximation

        Parameters
        ----------
        full_rank : :class:`FullRank`
            approximation to start with

        Returns
        -------
        :class:`FullRankADVI`
        """
        if not isinstance(full_rank, FullRank):
            raise TypeError('Expected MeanField, got %r' % full_rank)
        inference = object.__new__(cls)
        objective = KL(full_rank)(None)
        inference.hist = np.asarray(())
        inference.objective = objective
        return inference

    @classmethod
    def from_mean_field(cls, mean_field, gpu_compat=False):
        """
        Construct FullRankADVI from MeanField approximation

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
        :class:`FullRankADVI`
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
        advi : :class:`ADVI`

        Other Parameters
        ----------------
        gpu_compat : bool
            use GPU compatible version or not

        Returns
        -------
        :class:`FullRankADVI`
        """
        inference = cls.from_mean_field(advi.approx, gpu_compat)
        inference.hist = advi.hist
        return inference


class SVGD(Inference):
    R"""
    Stein Variational Gradient Descent

    This inference is based on Kernelized Stein Discrepancy
    it's main idea is to move initial noisy particles so that
    they fit target distribution best.

    Algorithm is outlined below

    *Input:* A target distribution with density function :math:`p(x)`
            and a set of initial particles :math:`{x^0_i}^n_{i=1}`

    *Output:* A set of particles :math:`{x_i}^n_{i=1}` that approximates the target distribution.

    .. math::

        x_i^{l+1} &\leftarrow x_i^{l} + \epsilon_l \hat{\phi}^{*}(x_i^l) \\
        \hat{\phi}^{*}(x) &= \frac{1}{n}\sum^{n}_{j=1}[k(x^l_j,x) \nabla_{x^l_j} logp(x^l_j)+ \nabla_{x^l_j} k(x^l_j,x)]

    Parameters
    ----------
    n_particles : `int`
        number of particles to use for approximation
    jitter : `float`
        noise sd for initial point
    model : :class:`pymc3.Model`
        PyMC3 model for inference
    kernel : `callable`
        kernel function for KSD :math:`f(histogram) -> (k(x,.), \nabla_x k(x,.))`
    temperature : float
        parameter responsible for exploration, higher temperature gives more broad posterior estimate
    scale_cost_to_minibatch : bool, default False
        Scale cost to minibatch instead of full dataset
    start : `dict`
        initial point for inference
    histogram : :class:`Empirical`
        initialize SVGD with given Empirical approximation instead of default initial particles
    random_seed : None or int
        leave None to use package global RandomStream or other
        valid value to create instance specific one
    start : `Point`
        starting point for inference

    References
    ----------
    -   Qiang Liu, Dilin Wang (2016)
        Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm
        arXiv:1608.04471

    -   Yang Liu, Prajit Ramachandran, Qiang Liu, Jian Peng (2017)
        Stein Variational Policy Gradient
        arXiv:1704.02399
    """

    def __init__(self, n_particles=100, jitter=.01, model=None, kernel=test_functions.rbf,
                 temperature=1, scale_cost_to_minibatch=False, start=None, histogram=None,
                 random_seed=None, local_rv=None):
        if histogram is None:
            histogram = Empirical.from_noise(
                n_particles, jitter=jitter,
                scale_cost_to_minibatch=scale_cost_to_minibatch,
                start=start, model=model, local_rv=local_rv, random_seed=random_seed)
        super(SVGD, self).__init__(
            KSD, histogram,
            kernel, op_kwargs=dict(temperature=temperature),
            model=model, random_seed=random_seed)


class ASVGD(Inference):
    R"""
    Amortized Stein Variational Gradient Descent

    This inference is based on Kernelized Stein Discrepancy
    it's main idea is to move initial noisy particles so that
    they fit target distribution best.

    Algorithm is outlined below

    *Input:* Parametrized random generator :math:`R_{\theta}`

    *Output:* :math:`R_{\theta^{*}}` that approximates the target distribution.

    .. math::

        \Delta x_i &= \hat{\phi}^{*}(x_i) \\
        \hat{\phi}^{*}(x) &= \frac{1}{n}\sum^{n}_{j=1}[k(x_j,x) \nabla_{x_j} logp(x_j)+ \nabla_{x_j} k(x_j,x)] \\
        \Delta_{\theta} &= \frac{1}{n}\sum^{n}_{i=1}\Delta x_i\frac{\partial x_i}{\partial \theta}

    Parameters
    ----------
    approx : :class:`Approximation`
    local_rv : dict[var->tuple]
        mapping {model_variable -> local_variable (:math:`\mu`, :math:`\rho`)}
        Local Vars are used for Autoencoding Variational Bayes
        See (AEVB; Kingma and Welling, 2014) for details
    kernel : `callable`
        kernel function for KSD :math:`f(histogram) -> (k(x,.), \nabla_x k(x,.))`
    temperature : float
        parameter responsible for exploration, higher temperature gives more broad posterior estimate
    model : :class:`Model`
    kwargs : kwargs for :class:`Approximation`

    References
    ----------
    -   Dilin Wang, Yihao Feng, Qiang Liu (2016)
        Learning to Sample Using Stein Discrepancy
        http://bayesiandeeplearning.org/papers/BDL_21.pdf

    -   Dilin Wang, Qiang Liu (2016)
        Learning to Draw Samples: With Application to Amortized MLE for Generative Adversarial Learning
        arXiv:1611.01722
    
    -   Yang Liu, Prajit Ramachandran, Qiang Liu, Jian Peng (2017)
        Stein Variational Policy Gradient
        arXiv:1704.02399
    """

    def __init__(self, approx=FullRank, local_rv=None,
                 kernel=test_functions.rbf, temperature=1, model=None, **kwargs):
        super(ASVGD, self).__init__(
            op=AKSD,
            approx=approx,
            local_rv=local_rv,
            tf=kernel,
            model=model,
            op_kwargs=dict(temperature=temperature),
            **kwargs
        )

    def fit(self, n=10000, score=None, callbacks=None, progressbar=True,
            obj_n_mc=30, **kwargs):
        """
        Performs Amortized Stein Variational Gradient Descent

        Parameters
        ----------
        n : int
            number of iterations
        score : bool
            evaluate loss on each iteration or not
        callbacks : list[function : (Approximation, losses, i) -> None]
            calls provided functions after each iteration step
        progressbar : bool
            whether to show progressbar or not
        obj_n_mc : int
            sample `n` particles for Stein gradient
        kwargs : kwargs
            additional kwargs for :func:`ObjectiveFunction.step_function`

        Returns
        -------
        Approximation
        """
        return super(ASVGD, self).fit(
            n=n, score=score, callbacks=callbacks,
            progressbar=progressbar, obj_n_mc=obj_n_mc, **kwargs)

    def run_profiling(self, n=1000, score=None, obj_n_mc=30, **kwargs):
        return super(ASVGD, self).run_profiling(
            n=n, score=score, obj_n_mc=obj_n_mc, **kwargs)


def fit(n=10000, local_rv=None, method='advi', model=None,
        random_seed=None, start=None, inf_kwargs=None, **kwargs):
    R"""
    Handy shortcut for using inference methods in functional way

    Parameters
    ----------
    n : `int`
        number of iterations
    local_rv : dict[var->tuple]
        mapping {model_variable -> local_variable (:math:`\mu`, :math:`\rho`)}
        Local Vars are used for Autoencoding Variational Bayes
        See (AEVB; Kingma and Welling, 2014) for details
    method : str or :class:`Inference`
        string name is case insensitive in {'advi', 'fullrank_advi', 'advi->fullrank_advi', 'svgd', 'asvgd'}
    model : :class:`pymc3.Model`
        PyMC3 model for inference
    random_seed : None or int
        leave None to use package global RandomStream or other
        valid value to create instance specific one
    inf_kwargs : dict
        additional kwargs passed to :class:`Inference`
    start : `Point`
        starting point for inference

    Other Parameters
    ----------------
    frac : `float`
        if method is 'advi->fullrank_advi' represents advi fraction when training
    kwargs : kwargs
        additional kwargs for :func:`Inference.fit`

    Returns
    -------
    :class:`Approximation`
    """
    if inf_kwargs is None:
        inf_kwargs = dict()
    if model is None:
        model = pm.modelcontext(model)
    _select = dict(
        advi=ADVI,
        fullrank_advi=FullRankADVI,
        svgd=SVGD,
        asvgd=ASVGD
    )
    if isinstance(method, str) and method.lower() == 'advi->fullrank_advi':
        frac = kwargs.pop('frac', .5)
        if not 0. < frac < 1.:
            raise ValueError('frac should be in (0, 1)')
        n1 = int(n * frac)
        n2 = n - n1
        inference = ADVI(
            local_rv=local_rv,
            model=model,
            random_seed=random_seed,
            start=start)
        logger.info('fitting advi ...')
        inference.fit(n1, **kwargs)
        inference = FullRankADVI.from_advi(inference)
        logger.info('fitting fullrank advi ...')
        return inference.fit(n2, **kwargs)

    elif isinstance(method, str):
        try:
            inference = _select[method.lower()](
                local_rv=local_rv, model=model, random_seed=random_seed,
                start=start,
                **inf_kwargs
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
