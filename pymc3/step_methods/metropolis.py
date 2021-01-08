#   Copyright 2020 The PyMC Developers
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

import numpy as np
import numpy.random as nr
import scipy.linalg
import theano

import pymc3 as pm

from pymc3.distributions import draw_values
from pymc3.step_methods.arraystep import (
    ArrayStep,
    ArrayStepShared,
    Competence,
    PopulationArrayStepShared,
    metrop_select,
)
from pymc3.theanof import floatX

__all__ = [
    "Metropolis",
    "DEMetropolis",
    "DEMetropolisZ",
    "BinaryMetropolis",
    "BinaryGibbsMetropolis",
    "CategoricalGibbsMetropolis",
    "NormalProposal",
    "CauchyProposal",
    "LaplaceProposal",
    "PoissonProposal",
    "MultivariateNormalProposal",
]

# Available proposal distributions for Metropolis


class Proposal:
    def __init__(self, s):
        self.s = s


class NormalProposal(Proposal):
    def __call__(self):
        return nr.normal(scale=self.s)


class UniformProposal(Proposal):
    def __call__(self):
        return nr.uniform(low=-self.s, high=self.s, size=len(self.s))


class CauchyProposal(Proposal):
    def __call__(self):
        return nr.standard_cauchy(size=np.size(self.s)) * self.s


class LaplaceProposal(Proposal):
    def __call__(self):
        size = np.size(self.s)
        return (nr.standard_exponential(size=size) - nr.standard_exponential(size=size)) * self.s


class PoissonProposal(Proposal):
    def __call__(self):
        return nr.poisson(lam=self.s, size=np.size(self.s)) - self.s


class MultivariateNormalProposal(Proposal):
    def __init__(self, s):
        n, m = s.shape
        if n != m:
            raise ValueError("Covariance matrix is not symmetric.")
        self.n = n
        self.chol = scipy.linalg.cholesky(s, lower=True)

    def __call__(self, num_draws=None):
        if num_draws is not None:
            b = np.random.randn(self.n, num_draws)
            return np.dot(self.chol, b).T
        else:
            b = np.random.randn(self.n)
            return np.dot(self.chol, b)


class Metropolis(ArrayStepShared):
    """
    Metropolis-Hastings sampling step

    Parameters
    ----------
    vars: list
        List of variables for sampler
    S: standard deviation or covariance matrix
        Some measure of variance to parameterize proposal distribution
    proposal_dist: function
        Function that returns zero-mean deviates when parameterized with
        S (and n). Defaults to normal.
    scaling: scalar or array
        Initial scale factor for proposal. Defaults to 1.
    tune: bool
        Flag for tuning. Defaults to True.
    tune_interval: int
        The frequency of tuning. Defaults to 100 iterations.
    model: PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).
    mode:  string or `Mode` instance.
        compilation mode passed to Theano functions
    """

    name = "metropolis"

    default_blocked = False
    generates_stats = True
    stats_dtypes = [
        {
            "accept": np.float64,
            "accepted": np.bool,
            "tune": np.bool,
            "scaling": np.float64,
        }
    ]

    def __init__(
        self,
        vars=None,
        S=None,
        proposal_dist=None,
        scaling=1.0,
        tune=True,
        tune_interval=100,
        model=None,
        mode=None,
        **kwargs
    ):

        model = pm.modelcontext(model)

        if vars is None:
            vars = model.vars
        vars = pm.inputvars(vars)

        if S is None:
            S = np.ones(sum(v.dsize for v in vars))

        if proposal_dist is not None:
            self.proposal_dist = proposal_dist(S)
        elif S.ndim == 1:
            self.proposal_dist = NormalProposal(S)
        elif S.ndim == 2:
            self.proposal_dist = MultivariateNormalProposal(S)
        else:
            raise ValueError("Invalid rank for variance: %s" % S.ndim)

        self.scaling = np.atleast_1d(scaling).astype("d")
        self.tune = tune
        self.tune_interval = tune_interval
        self.steps_until_tune = tune_interval
        self.accepted = 0

        # Determine type of variables
        self.discrete = np.concatenate(
            [[v.dtype in pm.discrete_types] * (v.dsize or 1) for v in vars]
        )
        self.any_discrete = self.discrete.any()
        self.all_discrete = self.discrete.all()

        # remember initial settings before tuning so they can be reset
        self._untuned_settings = dict(
            scaling=self.scaling, steps_until_tune=tune_interval, accepted=self.accepted
        )

        self.mode = mode

        shared = pm.make_shared_replacements(vars, model)
        self.delta_logp = delta_logp(model.logpt, vars, shared)
        super().__init__(vars, shared)

    def reset_tuning(self):
        """Resets the tuned sampler parameters to their initial values."""
        for attr, initial_value in self._untuned_settings.items():
            setattr(self, attr, initial_value)
        return

    def astep(self, q0):
        if not self.steps_until_tune and self.tune:
            # Tune scaling parameter
            self.scaling = tune(self.scaling, self.accepted / float(self.tune_interval))
            # Reset counter
            self.steps_until_tune = self.tune_interval
            self.accepted = 0

        delta = self.proposal_dist() * self.scaling

        if self.any_discrete:
            if self.all_discrete:
                delta = np.round(delta, 0).astype("int64")
                q0 = q0.astype("int64")
                q = (q0 + delta).astype("int64")
            else:
                delta[self.discrete] = np.round(delta[self.discrete], 0)
                q = q0 + delta
        else:
            q = floatX(q0 + delta)

        accept = self.delta_logp(q, q0)
        q_new, accepted = metrop_select(accept, q, q0)
        self.accepted += accepted

        self.steps_until_tune -= 1

        stats = {
            "tune": self.tune,
            "scaling": self.scaling,
            "accept": np.exp(accept),
            "accepted": accepted,
        }

        return q_new, [stats]

    @staticmethod
    def competence(var, has_grad):
        return Competence.COMPATIBLE


def tune(scale, acc_rate):
    """
    Tunes the scaling parameter for the proposal distribution
    according to the acceptance rate over the last tune_interval:

    Rate    Variance adaptation
    ----    -------------------
    <0.001        x 0.1
    <0.05         x 0.5
    <0.2          x 0.9
    >0.5          x 1.1
    >0.75         x 2
    >0.95         x 10

    """
    if acc_rate < 0.001:
        # reduce by 90 percent
        return scale * 0.1
    elif acc_rate < 0.05:
        # reduce by 50 percent
        return scale * 0.5
    elif acc_rate < 0.2:
        # reduce by ten percent
        return scale * 0.9
    elif acc_rate > 0.95:
        # increase by factor of ten
        return scale * 10.0
    elif acc_rate > 0.75:
        # increase by double
        return scale * 2.0
    elif acc_rate > 0.5:
        # increase by ten percent
        return scale * 1.1

    return scale


class BinaryMetropolis(ArrayStep):
    """Metropolis-Hastings optimized for binary variables

    Parameters
    ----------
    vars: list
        List of variables for sampler
    scaling: scalar or array
        Initial scale factor for proposal. Defaults to 1.
    tune: bool
        Flag for tuning. Defaults to True.
    tune_interval: int
        The frequency of tuning. Defaults to 100 iterations.
    model: PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).

    """

    name = "binary_metropolis"

    generates_stats = True
    stats_dtypes = [
        {
            "accept": np.float64,
            "tune": np.bool,
            "p_jump": np.float64,
        }
    ]

    def __init__(self, vars, scaling=1.0, tune=True, tune_interval=100, model=None):

        model = pm.modelcontext(model)

        self.scaling = scaling
        self.tune = tune
        self.tune_interval = tune_interval
        self.steps_until_tune = tune_interval
        self.accepted = 0

        if not all([v.dtype in pm.discrete_types for v in vars]):
            raise ValueError("All variables must be Bernoulli for BinaryMetropolis")

        super().__init__(vars, [model.fastlogp])

    def astep(self, q0, logp):

        # Convert adaptive_scale_factor to a jump probability
        p_jump = 1.0 - 0.5 ** self.scaling

        rand_array = nr.random(q0.shape)
        q = np.copy(q0)
        # Locations where switches occur, according to p_jump
        switch_locs = rand_array < p_jump
        q[switch_locs] = True - q[switch_locs]

        accept = logp(q) - logp(q0)
        q_new, accepted = metrop_select(accept, q, q0)
        self.accepted += accepted

        stats = {
            "tune": self.tune,
            "accept": np.exp(accept),
            "p_jump": p_jump,
        }

        return q_new, [stats]

    @staticmethod
    def competence(var):
        """
        BinaryMetropolis is only suitable for binary (bool)
        and Categorical variables with k=1.
        """
        distribution = getattr(var.distribution, "parent_dist", var.distribution)
        if isinstance(distribution, pm.Bernoulli) or (var.dtype in pm.bool_types):
            return Competence.COMPATIBLE
        elif isinstance(distribution, pm.Categorical) and (distribution.k == 2):
            return Competence.COMPATIBLE
        return Competence.INCOMPATIBLE


class BinaryGibbsMetropolis(ArrayStep):
    """A Metropolis-within-Gibbs step method optimized for binary variables

    Parameters
    ----------
    vars: list
        List of variables for sampler
    order: list or 'random'
        List of integers indicating the Gibbs update order
        e.g., [0, 2, 1, ...]. Default is random
    transit_p: float
        The diagonal of the transition kernel. A value > .5 gives anticorrelated proposals,
        which resulting in more efficient antithetical sampling. Default is 0.8
    model: PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).

    """

    name = "binary_gibbs_metropolis"

    def __init__(self, vars, order="random", transit_p=0.8, model=None):

        model = pm.modelcontext(model)

        # transition probabilities
        self.transit_p = transit_p

        self.dim = sum(v.dsize for v in vars)

        if order == "random":
            self.shuffle_dims = True
            self.order = list(range(self.dim))
        else:
            if sorted(order) != list(range(self.dim)):
                raise ValueError("Argument 'order' has to be a permutation")
            self.shuffle_dims = False
            self.order = order

        if not all([v.dtype in pm.discrete_types for v in vars]):
            raise ValueError("All variables must be binary for BinaryGibbsMetropolis")

        super().__init__(vars, [model.fastlogp])

    def astep(self, q0, logp):
        order = self.order
        if self.shuffle_dims:
            nr.shuffle(order)

        q = np.copy(q0)
        logp_curr = logp(q)

        for idx in order:
            # No need to do metropolis update if the same value is proposed,
            # as you will get the same value regardless of accepted or reject
            if nr.rand() < self.transit_p:
                curr_val, q[idx] = q[idx], True - q[idx]
                logp_prop = logp(q)
                q[idx], accepted = metrop_select(logp_prop - logp_curr, q[idx], curr_val)
                if accepted:
                    logp_curr = logp_prop

        return q

    @staticmethod
    def competence(var):
        """
        BinaryMetropolis is only suitable for Bernoulli
        and Categorical variables with k=2.
        """
        distribution = getattr(var.distribution, "parent_dist", var.distribution)
        if isinstance(distribution, pm.Bernoulli) or (var.dtype in pm.bool_types):
            return Competence.IDEAL
        elif isinstance(distribution, pm.Categorical) and (distribution.k == 2):
            return Competence.IDEAL
        return Competence.INCOMPATIBLE


class CategoricalGibbsMetropolis(ArrayStep):
    """A Metropolis-within-Gibbs step method optimized for categorical variables.
    This step method works for Bernoulli variables as well, but it is not
    optimized for them, like BinaryGibbsMetropolis is. Step method supports
    two types of proposals: A uniform proposal and a proportional proposal,
    which was introduced by Liu in his 1996 technical report
    "Metropolized Gibbs Sampler: An Improvement".
    """

    name = "categorical_gibbs_metropolis"

    def __init__(self, vars, proposal="uniform", order="random", model=None):

        model = pm.modelcontext(model)
        vars = pm.inputvars(vars)

        dimcats = []
        # The above variable is a list of pairs (aggregate dimension, number
        # of categories). For example, if vars = [x, y] with x being a 2-D
        # variable with M categories and y being a 3-D variable with N
        # categories, we will have dimcats = [(0, M), (1, M), (2, N), (3, N), (4, N)].
        for v in vars:
            distr = getattr(v.distribution, "parent_dist", v.distribution)
            if isinstance(distr, pm.Categorical):
                k = draw_values([distr.k])[0]
            elif isinstance(distr, pm.Bernoulli) or (v.dtype in pm.bool_types):
                k = 2
            else:
                raise ValueError(
                    "All variables must be categorical or binary" + "for CategoricalGibbsMetropolis"
                )
            start = len(dimcats)
            dimcats += [(dim, k) for dim in range(start, start + v.dsize)]

        if order == "random":
            self.shuffle_dims = True
            self.dimcats = dimcats
        else:
            if sorted(order) != list(range(len(dimcats))):
                raise ValueError("Argument 'order' has to be a permutation")
            self.shuffle_dims = False
            self.dimcats = [dimcats[j] for j in order]

        if proposal == "uniform":
            self.astep = self.astep_unif
        elif proposal == "proportional":
            # Use the optimized "Metropolized Gibbs Sampler" described in Liu96.
            self.astep = self.astep_prop
        else:
            raise ValueError("Argument 'proposal' should either be 'uniform' or 'proportional'")

        super().__init__(vars, [model.fastlogp])

    def astep_unif(self, q0, logp):
        dimcats = self.dimcats
        if self.shuffle_dims:
            nr.shuffle(dimcats)

        q = np.copy(q0)
        logp_curr = logp(q)

        for dim, k in dimcats:
            curr_val, q[dim] = q[dim], sample_except(k, q[dim])
            logp_prop = logp(q)
            q[dim], accepted = metrop_select(logp_prop - logp_curr, q[dim], curr_val)
            if accepted:
                logp_curr = logp_prop
        return q

    def astep_prop(self, q0, logp):
        dimcats = self.dimcats
        if self.shuffle_dims:
            nr.shuffle(dimcats)

        q = np.copy(q0)
        logp_curr = logp(q)

        for dim, k in dimcats:
            logp_curr = self.metropolis_proportional(q, logp, logp_curr, dim, k)

        return q

    def metropolis_proportional(self, q, logp, logp_curr, dim, k):
        given_cat = int(q[dim])
        log_probs = np.zeros(k)
        log_probs[given_cat] = logp_curr
        candidates = list(range(k))
        for candidate_cat in candidates:
            if candidate_cat != given_cat:
                q[dim] = candidate_cat
                log_probs[candidate_cat] = logp(q)
        probs = softmax(log_probs)
        prob_curr, probs[given_cat] = probs[given_cat], 0.0
        probs /= 1.0 - prob_curr
        proposed_cat = nr.choice(candidates, p=probs)
        accept_ratio = (1.0 - prob_curr) / (1.0 - probs[proposed_cat])
        if not np.isfinite(accept_ratio) or nr.uniform() >= accept_ratio:
            q[dim] = given_cat
            return logp_curr
        q[dim] = proposed_cat
        return log_probs[proposed_cat]

    @staticmethod
    def competence(var):
        """
        CategoricalGibbsMetropolis is only suitable for Bernoulli and
        Categorical variables.
        """
        distribution = getattr(var.distribution, "parent_dist", var.distribution)
        if isinstance(distribution, pm.Categorical):
            if distribution.k > 2:
                return Competence.IDEAL
            return Competence.COMPATIBLE
        elif isinstance(distribution, pm.Bernoulli) or (var.dtype in pm.bool_types):
            return Competence.COMPATIBLE
        return Competence.INCOMPATIBLE


class DEMetropolis(PopulationArrayStepShared):
    """
    Differential Evolution Metropolis sampling step.

    Parameters
    ----------
    lamb: float
        Lambda parameter of the DE proposal mechanism. Defaults to 2.38 / sqrt(2 * ndim)
    vars: list
        List of variables for sampler
    S: standard deviation or covariance matrix
        Some measure of variance to parameterize proposal distribution
    proposal_dist: function
        Function that returns zero-mean deviates when parameterized with
        S (and n). Defaults to Uniform(-S,+S).
    scaling: scalar or array
        Initial scale factor for epsilon. Defaults to 0.001
    tune: str
        Which hyperparameter to tune. Defaults to None, but can also be 'scaling' or 'lambda'.
    tune_interval: int
        The frequency of tuning. Defaults to 100 iterations.
    model: PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).
    mode:  string or `Mode` instance.
        compilation mode passed to Theano functions

    References
    ----------
    .. [Braak2006] Cajo C.F. ter Braak (2006).
        A Markov Chain Monte Carlo version of the genetic algorithm
        Differential Evolution: easy Bayesian computing for real parameter spaces.
        Statistics and Computing
        `link <https://doi.org/10.1007/s11222-006-8769-1>`__
    """

    name = "DEMetropolis"

    default_blocked = True
    generates_stats = True
    stats_dtypes = [
        {
            "accept": np.float64,
            "accepted": np.bool,
            "tune": np.bool,
            "scaling": np.float64,
            "lambda": np.float64,
        }
    ]

    def __init__(
        self,
        vars=None,
        S=None,
        proposal_dist=None,
        lamb=None,
        scaling=0.001,
        tune=None,
        tune_interval=100,
        model=None,
        mode=None,
        **kwargs
    ):

        model = pm.modelcontext(model)

        if vars is None:
            vars = model.cont_vars
        vars = pm.inputvars(vars)

        if S is None:
            S = np.ones(model.ndim)

        if proposal_dist is not None:
            self.proposal_dist = proposal_dist(S)
        else:
            self.proposal_dist = UniformProposal(S)

        self.scaling = np.atleast_1d(scaling).astype("d")
        if lamb is None:
            # default to the optimal lambda for normally distributed targets
            lamb = 2.38 / np.sqrt(2 * model.ndim)
        self.lamb = float(lamb)
        if tune not in {None, "scaling", "lambda"}:
            raise ValueError('The parameter "tune" must be one of {None, scaling, lambda}')
        self.tune = tune
        self.tune_interval = tune_interval
        self.steps_until_tune = tune_interval
        self.accepted = 0

        self.mode = mode

        shared = pm.make_shared_replacements(vars, model)
        self.delta_logp = delta_logp(model.logpt, vars, shared)
        super().__init__(vars, shared)

    def astep(self, q0):
        if not self.steps_until_tune and self.tune:
            if self.tune == "scaling":
                self.scaling = tune(self.scaling, self.accepted / float(self.tune_interval))
            elif self.tune == "lambda":
                self.lamb = tune(self.lamb, self.accepted / float(self.tune_interval))
            # Reset counter
            self.steps_until_tune = self.tune_interval
            self.accepted = 0

        epsilon = self.proposal_dist() * self.scaling

        # differential evolution proposal
        # select two other chains
        ir1, ir2 = np.random.choice(self.other_chains, 2, replace=False)
        r1 = self.bij.map(self.population[ir1])
        r2 = self.bij.map(self.population[ir2])
        # propose a jump
        q = floatX(q0 + self.lamb * (r1 - r2) + epsilon)

        accept = self.delta_logp(q, q0)
        q_new, accepted = metrop_select(accept, q, q0)
        self.accepted += accepted

        self.steps_until_tune -= 1

        stats = {
            "tune": self.tune,
            "scaling": self.scaling,
            "lambda": self.lamb,
            "accept": np.exp(accept),
            "accepted": accepted,
        }

        return q_new, [stats]

    @staticmethod
    def competence(var, has_grad):
        if var.dtype in pm.discrete_types:
            return Competence.INCOMPATIBLE
        return Competence.COMPATIBLE


class DEMetropolisZ(ArrayStepShared):
    """
    Adaptive Differential Evolution Metropolis sampling step that uses the past to inform jumps.

    Parameters
    ----------
    lamb: float
        Lambda parameter of the DE proposal mechanism. Defaults to 2.38 / sqrt(2 * ndim)
    vars: list
        List of variables for sampler
    S: standard deviation or covariance matrix
        Some measure of variance to parameterize proposal distribution
    proposal_dist: function
        Function that returns zero-mean deviates when parameterized with
        S (and n). Defaults to Uniform(-S,+S).
    scaling: scalar or array
        Initial scale factor for epsilon. Defaults to 0.001
    tune: str
        Which hyperparameter to tune. Defaults to 'lambda', but can also be 'scaling' or None.
    tune_interval: int
        The frequency of tuning. Defaults to 100 iterations.
    tune_drop_fraction: float
        Fraction of tuning steps that will be removed from the samplers history when the tuning ends.
        Defaults to 0.9 - keeping the last 10% of tuning steps for good mixing while removing 90% of
        potentially unconverged tuning positions.
    model: PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).
    mode:  string or `Mode` instance.
        compilation mode passed to Theano functions

    References
    ----------
    .. [Braak2006] Cajo C.F. ter Braak (2006).
        Differential Evolution Markov Chain with snooker updater and fewer chains.
        Statistics and Computing
        `link <https://doi.org/10.1007/s11222-008-9104-9>`__
    """

    name = "DEMetropolisZ"

    default_blocked = True
    generates_stats = True
    stats_dtypes = [
        {
            "accept": np.float64,
            "accepted": np.bool,
            "tune": np.bool,
            "scaling": np.float64,
            "lambda": np.float64,
        }
    ]

    def __init__(
        self,
        vars=None,
        S=None,
        proposal_dist=None,
        lamb=None,
        scaling=0.001,
        tune="lambda",
        tune_interval=100,
        tune_drop_fraction: float = 0.9,
        model=None,
        mode=None,
        **kwargs
    ):
        model = pm.modelcontext(model)

        if vars is None:
            vars = model.cont_vars
        vars = pm.inputvars(vars)

        if S is None:
            S = np.ones(model.ndim)

        if proposal_dist is not None:
            self.proposal_dist = proposal_dist(S)
        else:
            self.proposal_dist = UniformProposal(S)

        self.scaling = np.atleast_1d(scaling).astype("d")
        if lamb is None:
            # default to the optimal lambda for normally distributed targets
            lamb = 2.38 / np.sqrt(2 * model.ndim)
        self.lamb = float(lamb)
        if tune not in {None, "scaling", "lambda"}:
            raise ValueError('The parameter "tune" must be one of {None, scaling, lambda}')
        self.tune = True
        self.tune_target = tune
        self.tune_interval = tune_interval
        self.tune_drop_fraction = tune_drop_fraction
        self.steps_until_tune = tune_interval
        self.accepted = 0

        # cache local history for the Z-proposals
        self._history = []
        # remember initial settings before tuning so they can be reset
        self._untuned_settings = dict(
            scaling=self.scaling,
            lamb=self.lamb,
            steps_until_tune=tune_interval,
            accepted=self.accepted,
        )

        self.mode = mode

        shared = pm.make_shared_replacements(vars, model)
        self.delta_logp = delta_logp(model.logpt, vars, shared)
        super().__init__(vars, shared)

    def reset_tuning(self):
        """Resets the tuned sampler parameters and history to their initial values."""
        # history can't be reset via the _untuned_settings dict because it's a list
        self._history = []
        for attr, initial_value in self._untuned_settings.items():
            setattr(self, attr, initial_value)
        return

    def astep(self, q0):
        # same tuning scheme as DEMetropolis
        if not self.steps_until_tune and self.tune:
            if self.tune_target == "scaling":
                self.scaling = tune(self.scaling, self.accepted / float(self.tune_interval))
            elif self.tune_target == "lambda":
                self.lamb = tune(self.lamb, self.accepted / float(self.tune_interval))
            # Reset counter
            self.steps_until_tune = self.tune_interval
            self.accepted = 0

        epsilon = self.proposal_dist() * self.scaling

        it = len(self._history)
        # use the DE-MCMC-Z proposal scheme as soon as the history has 2 entries
        if it > 1:
            # differential evolution proposal
            # select two other chains
            iz1 = np.random.randint(it)
            iz2 = np.random.randint(it)
            while iz2 == iz1:
                iz2 = np.random.randint(it)

            z1 = self._history[iz1]
            z2 = self._history[iz2]
            # propose a jump
            q = floatX(q0 + self.lamb * (z1 - z2) + epsilon)
        else:
            # propose just with noise in the first 2 iterations
            q = floatX(q0 + epsilon)

        accept = self.delta_logp(q, q0)
        q_new, accepted = metrop_select(accept, q, q0)
        self.accepted += accepted
        self._history.append(q_new)

        self.steps_until_tune -= 1

        stats = {
            "tune": self.tune,
            "scaling": self.scaling,
            "lambda": self.lamb,
            "accept": np.exp(accept),
            "accepted": accepted,
        }

        return q_new, [stats]

    def stop_tuning(self):
        """At the end of the tuning phase, this method removes the first x% of the history
        so future proposals are not informed by unconverged tuning iterations.
        """
        it = len(self._history)
        n_drop = int(self.tune_drop_fraction * it)
        self._history = self._history[n_drop:]
        return super().stop_tuning()

    @staticmethod
    def competence(var, has_grad):
        if var.dtype in pm.discrete_types:
            return Competence.INCOMPATIBLE
        return Competence.COMPATIBLE


def sample_except(limit, excluded):
    candidate = nr.choice(limit - 1)
    if candidate >= excluded:
        candidate += 1
    return candidate


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=0)


def delta_logp(logp, vars, shared):
    [logp0], inarray0 = pm.join_nonshared_inputs([logp], vars, shared)

    tensor_type = inarray0.type
    inarray1 = tensor_type("inarray1")

    logp1 = pm.CallableTensor(logp0)(inarray1)

    f = theano.function([inarray1, inarray0], logp1 - logp0)
    f.trust_input = True
    return f
