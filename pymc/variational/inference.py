#   Copyright 2023 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import collections
import logging
import warnings

import numpy as np

from fastprogress.fastprogress import progress_bar

import pymc as pm

from pymc.variational import test_functions
from pymc.variational.approximations import Empirical, FullRank, MeanField
from pymc.variational.operators import KL, KSD

logger = logging.getLogger(__name__)

__all__ = [
    "ADVI",
    "FullRankADVI",
    "SVGD",
    "ASVGD",
    "Inference",
    "ImplicitGradient",
    "KLqp",
    "fit",
]

State = collections.namedtuple("State", "i,step,callbacks,score")


class Inference:
    r"""**Base class for Variational Inference**

    Communicates Operator, Approximation and Test Function to build Objective Function

    Parameters
    ----------
    op : Operator class    #:class:`~pymc.variational.operators`
    approx : Approximation class or instance    #:class:`~pymc.variational.approximations`
    tf : TestFunction instance  #?
    model : Model
        PyMC Model
    kwargs : kwargs passed to :class:`Operator` #:class:`~pymc.variational.operators`, optional
    """

    def __init__(self, op, approx, tf, **kwargs):
        self.hist = np.asarray(())
        self.objective = op(approx, **kwargs)(tf)
        self.state = None

    approx = property(lambda self: self.objective.approx)

    def _maybe_score(self, score):
        returns_loss = self.objective.op.returns_loss
        if score is None:
            score = returns_loss
        elif score and not returns_loss:
            warnings.warn(
                "method `fit` got `score == True` but %s "
                "does not return loss. Ignoring `score` argument" % self.objective.op
            )
            score = False
        else:
            pass
        return score

    def run_profiling(self, n=1000, score=None, **kwargs):
        score = self._maybe_score(score)
        fn_kwargs = kwargs.pop("fn_kwargs", dict())
        fn_kwargs["profile"] = True
        step_func = self.objective.step_function(score=score, fn_kwargs=fn_kwargs, **kwargs)
        progress = progress_bar(range(n))
        try:
            for _ in progress:
                step_func()
        except KeyboardInterrupt:
            pass
        return step_func.profile

    def fit(self, n=10000, score=None, callbacks=None, progressbar=True, **kwargs):
        """Perform Operator Variational Inference

        Parameters
        ----------
        n : int
            number of iterations
        score : bool
            evaluate loss on each iteration or not
        callbacks : list[function: (Approximation, losses, i) -> None]
            calls provided functions after each iteration step
        progressbar : bool
            whether to show progressbar or not

        Other Parameters
        ----------------
        obj_n_mc: int
            Number of monte carlo samples used for approximation of objective gradients
        tf_n_mc: `int`
            Number of monte carlo samples used for approximation of test function gradients
        obj_optimizer: function (grads, params) -> updates
            Optimizer that is used for objective params
        test_optimizer: function (grads, params) -> updates
            Optimizer that is used for test function params
        more_obj_params: `list`
            Add custom params for objective optimizer
        more_tf_params: `list`
            Add custom params for test function optimizer
        more_updates: `dict`
            Add custom updates to resulting updates
        total_grad_norm_constraint: `float`
            Bounds gradient norm, prevents exploding gradient problem
        fn_kwargs: `dict`
            Add kwargs to pytensor.function (e.g. `{'profile': True}`)
        more_replacements: `dict`
            Apply custom replacements before calculating gradients

        Returns
        -------
        :class:`Approximation`
        """
        if callbacks is None:
            callbacks = []
        score = self._maybe_score(score)
        step_func = self.objective.step_function(score=score, **kwargs)
        if progressbar:
            progress = progress_bar(range(n), display=progressbar)
        else:
            progress = range(n)
        if score:
            state = self._iterate_with_loss(0, n, step_func, progress, callbacks)
        else:
            state = self._iterate_without_loss(0, n, step_func, progress, callbacks)

        # hack to allow pm.fit() access to loss hist
        self.approx.hist = self.hist
        self.state = state

        return self.approx

    def _iterate_without_loss(self, s, _, step_func, progress, callbacks):
        i = 0
        try:
            for i in progress:
                step_func()
                current_param = self.approx.params[0].get_value()
                if np.isnan(current_param).any():
                    name_slc = []
                    tmp_hold = list(range(current_param.size))
                    for varname, slice_info in self.approx.groups[0].ordering.items():
                        slclen = len(tmp_hold[slice_info[1]])
                        for j in range(slclen):
                            name_slc.append((varname, j))
                    index = np.where(np.isnan(current_param))[0]
                    errmsg = ["NaN occurred in optimization. "]
                    suggest_solution = (
                        "Try tracking this parameter: "
                        "http://docs.pymc.io/notebooks/variational_api_quickstart.html#Tracking-parameters"
                    )
                    try:
                        for ii in index:
                            errmsg.append(
                                "The current approximation of RV `{}`.ravel()[{}]"
                                " is NaN.".format(*name_slc[ii])
                            )
                        errmsg.append(suggest_solution)
                    except IndexError:
                        pass
                    raise FloatingPointError("\n".join(errmsg))
                for callback in callbacks:
                    callback(self.approx, None, i + s + 1)
        except (KeyboardInterrupt, StopIteration) as e:
            if isinstance(e, StopIteration):
                logger.info(str(e))
        return State(i + s, step=step_func, callbacks=callbacks, score=False)

    def _iterate_with_loss(self, s, n, step_func, progress, callbacks):
        def _infmean(input_array):
            """Return the mean of the finite values of the array"""
            input_array = input_array[np.isfinite(input_array)].astype("float64")
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
                if np.isnan(e):
                    scores = scores[:i]
                    self.hist = np.concatenate([self.hist, scores])
                    current_param = self.approx.params[0].get_value()
                    name_slc = []
                    tmp_hold = list(range(current_param.size))
                    for varname, slice_info in self.approx.groups[0].ordering.items():
                        slclen = len(tmp_hold[slice_info[1]])
                        for j in range(slclen):
                            name_slc.append((varname, j))
                    index = np.where(np.isnan(current_param))[0]
                    errmsg = ["NaN occurred in optimization. "]
                    suggest_solution = (
                        "Try tracking this parameter: "
                        "http://docs.pymc.io/notebooks/variational_api_quickstart.html#Tracking-parameters"
                    )
                    try:
                        for ii in index:
                            errmsg.append(
                                "The current approximation of RV `{}`.ravel()[{}]"
                                " is NaN.".format(*name_slc[ii])
                            )
                        errmsg.append(suggest_solution)
                    except IndexError:
                        pass
                    raise FloatingPointError("\n".join(errmsg))
                scores[i] = e
                if i % 10 == 0:
                    avg_loss = _infmean(scores[max(0, i - 1000) : i + 1])
                    if hasattr(progress, "comment"):
                        progress.comment = f"Average Loss = {avg_loss:,.5g}"
                    avg_loss = scores[max(0, i - 1000) : i + 1].mean()
                    if hasattr(progress, "comment"):
                        progress.comment = f"Average Loss = {avg_loss:,.5g}"
                for callback in callbacks:
                    callback(self.approx, scores[: i + 1], i + s + 1)
        except (KeyboardInterrupt, StopIteration) as e:  # pragma: no cover
            # do not print log on the same line
            scores = scores[:i]
            if isinstance(e, StopIteration):
                logger.info(str(e))
            if n < 10:
                logger.info(
                    "Interrupted at {:,d} [{:.0f}%]: Loss = {:,.5g}".format(
                        i, 100 * i // n, scores[i]
                    )
                )
            else:
                avg_loss = _infmean(scores[min(0, i - 1000) : i + 1])
                logger.info(
                    "Interrupted at {:,d} [{:.0f}%]: Average Loss = {:,.5g}".format(
                        i, 100 * i // n, avg_loss
                    )
                )
        else:
            if n == 0:
                logger.info("Initialization only")
            elif n < 10:
                logger.info(f"Finished [100%]: Loss = {scores[-1]:,.5g}")
            else:
                avg_loss = _infmean(scores[max(0, i - 1000) : i + 1])
                logger.info(f"Finished [100%]: Average Loss = {avg_loss:,.5g}")
        self.hist = np.concatenate([self.hist, scores])
        return State(i + s, step=step_func, callbacks=callbacks, score=True)

    def refine(self, n, progressbar=True):
        """Refine the solution using the last compiled step function"""
        if self.state is None:
            raise TypeError("Need to call `.fit` first")
        i, step, callbacks, score = self.state
        if progressbar:
            progress = progress_bar(range(n), display=progressbar)
        else:
            progress = range(n)  # This is a guess at what progress_bar(n) does.
        if score:
            state = self._iterate_with_loss(i, n, step, progress, callbacks)
        else:
            state = self._iterate_without_loss(i, n, step, progress, callbacks)
        self.state = state


class KLqp(Inference):
    r"""**Kullback Leibler Divergence Inference**

    General approach to fit Approximations that define :math:`logq`
    by maximizing ELBO (Evidence Lower Bound). In some cases
    rescaling the regularization term KL may be beneficial

    .. math::

        ELBO_\beta = \log p(D|\theta) - \beta KL(q||p)

    Parameters
    ----------
    approx: :class:`Approximation`
        Approximation to fit, it is required to have `logQ`
    beta: float
        Scales the regularization term in ELBO (see Christopher P. Burgess et al., 2017)

    References
    ----------
    -   Christopher P. Burgess et al. (NIPS, 2017)
        Understanding disentangling in :math:`\beta`-VAE
        arXiv preprint 1804.03599
    """

    def __init__(self, approx, beta=1.0):
        super().__init__(KL, approx, None, beta=beta)


class ADVI(KLqp):
    r"""**Automatic Differentiation Variational Inference (ADVI)**

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
    These RVs are used only in AEVB (which is not implemented in PyMC).

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
    in the PyMC notebook directory.

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

        `model` with two types of RVs (`observed_RVs`,
        `global_RVs`).

    -   (optional) Minibatches

        The tensors to which mini-bathced samples are supplied are
        handled separately by using callbacks in :func:`Inference.fit` method
        that change storage of shared PyTensor variable or by :func:`pymc.generator`
        that automatically iterates over minibatches and defined beforehand.

    -   (optional) Parameters of deterministic mappings

        They have to be passed along with other params to :func:`Inference.fit` method
        as `more_obj_params` argument.

    For more information concerning training stage please reference
    :func:`pymc.variational.opvi.ObjectiveFunction.step_function`

    Parameters
    ----------
    model: :class:`pymc.Model`
        PyMC model for inference
    random_seed: None or int
    start: `dict[str, np.ndarray]` or `StartDict`
        starting point for inference
    start_sigma: `dict[str, np.ndarray]`
        starting standard deviation for inference, only available for method 'advi'

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

    def __init__(self, *args, **kwargs):
        super().__init__(MeanField(*args, **kwargs))


class FullRankADVI(KLqp):
    r"""**Full Rank Automatic Differentiation Variational Inference (ADVI)**

    Parameters
    ----------
    model: :class:`pymc.Model`
        PyMC model for inference
    random_seed: None or int
    start: `dict[str, np.ndarray]` or `StartDict`
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

    def __init__(self, *args, **kwargs):
        super().__init__(FullRank(*args, **kwargs))


class ImplicitGradient(Inference):
    """**Implicit Gradient for Variational Inference**

    **not suggested to use**

    An approach to fit arbitrary approximation by computing kernel based gradient
    By default RBF kernel is used for gradient estimation. Default estimator is
    Kernelized Stein Discrepancy with temperature equal to 1. This temperature works
    only for large number of samples. Larger temperature is needed for small number of
    samples but there is no theoretical approach to choose the best one in such case.
    """

    def __init__(self, approx, estimator=KSD, kernel=test_functions.rbf, **kwargs):
        super().__init__(op=estimator, approx=approx, tf=kernel, **kwargs)


class SVGD(ImplicitGradient):
    r"""**Stein Variational Gradient Descent**

    This inference is based on Kernelized Stein Discrepancy
    it's main idea is to move initial noisy particles so that
    they fit target distribution best.

    Algorithm is outlined below

    *Input:* A target distribution with density function :math:`p(x)`
            and a set of initial particles :math:`\{x^0_i\}^n_{i=1}`

    *Output:* A set of particles :math:`\{x^{*}_i\}^n_{i=1}` that approximates the target distribution.

    .. math::

        x_i^{l+1} &\leftarrow x_i^{l} + \epsilon_l \hat{\phi}^{*}(x_i^l) \\
        \hat{\phi}^{*}(x) &= \frac{1}{n}\sum^{n}_{j=1}[k(x^l_j,x) \nabla_{x^l_j} logp(x^l_j)+ \nabla_{x^l_j} k(x^l_j,x)]

    Parameters
    ----------
    n_particles: `int`
        number of particles to use for approximation
    jitter: `float`
        noise sd for initial point
    model: :class:`pymc.Model`
        PyMC model for inference
    kernel: `callable`
        kernel function for KSD :math:`f(histogram) -> (k(x,.), \nabla_x k(x,.))`
    temperature: float
        parameter responsible for exploration, higher temperature gives more broad posterior estimate
    start: `dict[str, np.ndarray]` or `StartDict`
        initial point for inference
    random_seed: None or int
    kwargs: other keyword arguments passed to estimator

    References
    ----------
    -   Qiang Liu, Dilin Wang (2016)
        Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm
        arXiv:1608.04471

    -   Yang Liu, Prajit Ramachandran, Qiang Liu, Jian Peng (2017)
        Stein Variational Policy Gradient
        arXiv:1704.02399
    """

    def __init__(
        self,
        n_particles=100,
        jitter=1,
        model=None,
        start=None,
        random_seed=None,
        estimator=KSD,
        kernel=test_functions.rbf,
        **kwargs,
    ):
        empirical = Empirical(
            size=n_particles,
            jitter=jitter,
            start=start,
            model=model,
            random_seed=random_seed,
        )
        super().__init__(approx=empirical, estimator=estimator, kernel=kernel, **kwargs)


class ASVGD(ImplicitGradient):
    r"""**Amortized Stein Variational Gradient Descent**

    **not suggested to use**

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
    approx: :class:`Approximation`
        default is :class:`FullRank` but can be any
    kernel: `callable`
        kernel function for KSD :math:`f(histogram) -> (k(x,.), \nabla_x k(x,.))`
    model: :class:`Model`
    kwargs: kwargs for gradient estimator

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

    def __init__(self, approx=None, estimator=KSD, kernel=test_functions.rbf, **kwargs):
        warnings.warn(
            "You are using experimental inference Operator. "
            "It requires careful choice of temperature, default is 1. "
            "Default temperature works well for low dimensional problems and "
            "for significant `n_obj_mc`. Temperature > 1 gives more exploration "
            "power to algorithm, < 1 leads to undesirable results. Please take "
            "it in account when looking at inference result. Posterior variance "
            "is often **underestimated** when using temperature = 1."
        )
        if approx is None:
            approx = FullRank(
                model=kwargs.pop("model", None), random_seed=kwargs.pop("random_seed", None)
            )
        super().__init__(estimator=estimator, approx=approx, kernel=kernel, **kwargs)

    def fit(
        self,
        n=10000,
        score=None,
        callbacks=None,
        progressbar=True,
        obj_n_mc=500,
        **kwargs,
    ):
        return super().fit(
            n=n,
            score=score,
            callbacks=callbacks,
            progressbar=progressbar,
            obj_n_mc=obj_n_mc,
            **kwargs,
        )

    def run_profiling(self, n=1000, score=None, obj_n_mc=500, **kwargs):
        return super().run_profiling(n=n, score=score, obj_n_mc=obj_n_mc, **kwargs)


def fit(
    n=10000,
    method="advi",
    model=None,
    random_seed=None,
    start=None,
    start_sigma=None,
    inf_kwargs=None,
    **kwargs,
):
    r"""Handy shortcut for using inference methods in functional way

    Parameters
    ----------
    n: `int`
        number of iterations
    method: str or :class:`Inference`
        string name is case insensitive in:

        -   'advi'  for ADVI
        -   'fullrank_advi'  for FullRankADVI
        -   'svgd'  for Stein Variational Gradient Descent
        -   'asvgd'  for Amortized Stein Variational Gradient Descent

    model: :class:`Model`
        PyMC model for inference
    random_seed: None or int
    inf_kwargs: dict
        additional kwargs passed to :class:`Inference`
    start: `dict[str, np.ndarray]` or `StartDict`
        starting point for inference
    start_sigma: `dict[str, np.ndarray]`
        starting standard deviation for inference, only available for method 'advi'

    Other Parameters
    ----------------
    score: bool
            evaluate loss on each iteration or not
    callbacks: list[function: (Approximation, losses, i) -> None]
        calls provided functions after each iteration step
    progressbar: bool
        whether to show progressbar or not
    obj_n_mc: `int`
        Number of monte carlo samples used for approximation of objective gradients
    tf_n_mc: `int`
        Number of monte carlo samples used for approximation of test function gradients
    obj_optimizer: function (grads, params) -> updates
        Optimizer that is used for objective params
    test_optimizer: function (grads, params) -> updates
        Optimizer that is used for test function params
    more_obj_params: `list`
        Add custom params for objective optimizer
    more_tf_params: `list`
        Add custom params for test function optimizer
    more_updates: `dict`
        Add custom updates to resulting updates
    total_grad_norm_constraint: `float`
        Bounds gradient norm, prevents exploding gradient problem
    fn_kwargs: `dict`
        Add kwargs to pytensor.function (e.g. `{'profile': True}`)
    more_replacements: `dict`
        Apply custom replacements before calculating gradients

    Returns
    -------
    :class:`Approximation`
    """
    if inf_kwargs is None:
        inf_kwargs = dict()
    else:
        inf_kwargs = inf_kwargs.copy()
    if random_seed is not None:
        inf_kwargs["random_seed"] = random_seed
    if start is not None:
        inf_kwargs["start"] = start
    if start_sigma is not None:
        if method != "advi":
            raise NotImplementedError("start_sigma is only available for method advi")
        inf_kwargs["start_sigma"] = start_sigma
    if model is None:
        model = pm.modelcontext(model)
    _select = dict(advi=ADVI, fullrank_advi=FullRankADVI, svgd=SVGD, asvgd=ASVGD)
    if isinstance(method, str):
        method = method.lower()
        if method in _select:
            inference = _select[method](model=model, **inf_kwargs)
        else:
            raise KeyError(f"method should be one of {set(_select.keys())} or Inference instance")
    elif isinstance(method, Inference):
        inference = method
    else:
        raise TypeError(f"method should be one of {set(_select.keys())} or Inference instance")
    return inference.fit(n, **kwargs)
