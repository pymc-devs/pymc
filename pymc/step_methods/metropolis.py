#   Copyright 2024 - present The PyMC Developers
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
from collections.abc import Callable
from dataclasses import field
from typing import Any, cast

import numpy as np
import numpy.random as nr
import pytensor
import scipy.linalg
import scipy.special

from pytensor import tensor as pt
from pytensor.graph.basic import Variable
from pytensor.graph.fg import MissingInputError
from pytensor.tensor.random.basic import BernoulliRV, CategoricalRV
from rich.progress import TextColumn
from rich.table import Column

import pymc as pm

from pymc.blocking import DictToArrayBijection, RaveledVars
from pymc.initial_point import PointType
from pymc.pytensorf import (
    CallableTensor,
    compile,
    floatX,
    join_nonshared_inputs,
    replace_rng_nodes,
)
from pymc.step_methods.arraystep import (
    ArrayStep,
    ArrayStepShared,
    PopulationArrayStepShared,
    StatsType,
    metrop_select,
)
from pymc.step_methods.compound import Competence, StepMethodState
from pymc.step_methods.state import dataclass_state

__all__ = [
    "BinaryGibbsMetropolis",
    "BinaryMetropolis",
    "CategoricalGibbsMetropolis",
    "CauchyProposal",
    "DEMetropolis",
    "DEMetropolisZ",
    "LaplaceProposal",
    "Metropolis",
    "MultivariateNormalProposal",
    "NormalProposal",
    "PoissonProposal",
]

from pymc.util import RandomGenerator, get_value_vars_from_user_vars

# Available proposal distributions for Metropolis


class Proposal:
    def __init__(self, s):
        self.s = s


class NormalProposal(Proposal):
    def __call__(self, rng: np.random.Generator | None = None):
        return (rng or nr).normal(scale=self.s)


class UniformProposal(Proposal):
    def __call__(self, rng: np.random.Generator | None = None):
        return (rng or nr).uniform(low=-self.s, high=self.s, size=len(self.s))


class CauchyProposal(Proposal):
    def __call__(self, rng: np.random.Generator | None = None):
        return (rng or nr).standard_cauchy(size=np.size(self.s)) * self.s


class LaplaceProposal(Proposal):
    def __call__(self, rng: np.random.Generator | None = None):
        size = np.size(self.s)
        r = rng or nr
        return (r.standard_exponential(size=size) - r.standard_exponential(size=size)) * self.s


class PoissonProposal(Proposal):
    def __call__(self, rng: np.random.Generator | None = None):
        return (rng or nr).poisson(lam=self.s, size=np.size(self.s)) - self.s


class MultivariateNormalProposal(Proposal):
    def __init__(self, s):
        n, m = s.shape
        if n != m:
            raise ValueError("Covariance matrix is not symmetric.")
        self.n = n
        self.chol = scipy.linalg.cholesky(s, lower=True)

    def __call__(self, num_draws=None, rng: np.random.Generator | None = None):
        rng_ = rng or nr
        if num_draws is not None:
            b = rng_.normal(size=(self.n, num_draws))
            return np.dot(self.chol, b).T
        else:
            b = rng_.normal(size=self.n)
            return np.dot(self.chol, b)


@dataclass_state
class MetropolisState(StepMethodState):
    scaling: np.ndarray
    tune: bool
    steps_until_tune: float
    tune_interval: float
    accepted_sum: np.ndarray
    accept_rate_iter: np.ndarray
    accepted_iter: np.ndarray
    enum_dims: np.ndarray

    discrete: np.ndarray = field(metadata={"frozen": True})
    any_discrete: bool = field(metadata={"frozen": True})
    all_discrete: bool = field(metadata={"frozen": True})
    elemwise_update: bool = field(metadata={"frozen": True})
    _untuned_settings: dict[str, np.ndarray | float] = field(metadata={"frozen": True})
    mode: Any = field(metadata={"frozen": True})


class Metropolis(ArrayStepShared):
    """Metropolis-Hastings sampling step."""

    name = "metropolis"

    default_blocked = False
    stats_dtypes_shapes = {
        "accept": (np.float64, []),
        "accepted": (np.float64, []),
        "tune": (bool, []),
        "scaling": (np.float64, []),
    }

    _state_class = MetropolisState

    def __init__(
        self,
        vars=None,
        *,
        S=None,
        proposal_dist=None,
        scaling=1.0,
        tune=True,
        tune_interval=100,
        model=None,
        mode=None,
        rng=None,
        initial_point: PointType | None = None,
        compile_kwargs: dict | None = None,
        blocked: bool = False,
    ):
        """Create an instance of a Metropolis stepper.

        Parameters
        ----------
        vars: list
            List of value variables for sampler
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
        mode: string or `Mode` instance.
            compilation mode passed to PyTensor functions
        rng: RandomGenerator
            An object that can produce be used to produce the step method's
            :py:class:`~numpy.random.Generator` object. Refer to
            :py:func:`pymc.util.get_random_generator` for more information.
        """
        model = pm.modelcontext(model)
        if initial_point is None:
            initial_point = model.initial_point()

        if vars is None:
            vars = model.value_vars
        else:
            vars = get_value_vars_from_user_vars(vars, model)

        initial_values_shape = [initial_point[v.name].shape for v in vars]
        if S is None:
            S = np.ones(int(sum(np.prod(ivs) for ivs in initial_values_shape)))

        if proposal_dist is not None:
            self.proposal_dist = proposal_dist(S)
        elif S.ndim == 1:
            self.proposal_dist = NormalProposal(S)
        elif S.ndim == 2:
            self.proposal_dist = MultivariateNormalProposal(S)
        else:
            raise ValueError(f"Invalid rank for variance: {S.ndim}")

        self.scaling = np.atleast_1d(scaling).astype("d")
        self.tune = tune
        self.tune_interval = tune_interval
        self.steps_until_tune = tune_interval

        # Determine type of variables
        self.discrete = np.concatenate(
            [[v.dtype in pm.discrete_types] * (initial_point[v.name].size or 1) for v in vars]
        )
        self.any_discrete = self.discrete.any()
        self.all_discrete = self.discrete.all()

        # Metropolis will try to handle one batched dimension at a time This, however,
        # is not safe for discrete multivariate distributions (looking at you Multinomial),
        # due to high dependency among the support dimensions. For continuous multivariate
        # distributions we assume they are being transformed in a way that makes each
        # dimension semi-independent.
        is_scalar = len(initial_values_shape) == 1 and initial_values_shape[0] == ()
        self.elemwise_update = not (
            is_scalar
            or (
                self.any_discrete
                and max(getattr(model.values_to_rvs[var].owner.op, "ndim_supp", 1) for var in vars)
                > 0
            )
        )
        if self.elemwise_update:
            dims = int(sum(np.prod(ivs) for ivs in initial_values_shape))
        else:
            dims = 1
        self.enum_dims = np.arange(dims, dtype=int)
        self.accept_rate_iter = np.zeros(dims, dtype=float)
        self.accepted_iter = np.zeros(dims, dtype=bool)
        self.accepted_sum = np.zeros(dims, dtype=int)

        # remember initial settings before tuning so they can be reset
        self._untuned_settings = {"scaling": self.scaling, "steps_until_tune": tune_interval}

        # TODO: This is not being used when compiling the logp function!
        self.mode = mode

        shared = pm.make_shared_replacements(initial_point, vars, model)
        self.delta_logp = delta_logp(initial_point, model.logp(), vars, shared, compile_kwargs)
        super().__init__(vars, shared, blocked=blocked, rng=rng)

    def reset_tuning(self):
        """Reset the tuned sampler parameters to their initial values."""
        for attr, initial_value in self._untuned_settings.items():
            setattr(self, attr, initial_value)
        self.accepted_sum[:] = 0
        return

    def astep(self, q0: RaveledVars) -> tuple[RaveledVars, StatsType]:
        point_map_info = q0.point_map_info
        q0d = q0.data

        if not self.steps_until_tune and self.tune:
            # Tune scaling parameter
            self.scaling = tune(self.scaling, self.accepted_sum / float(self.tune_interval))
            # Reset counter
            self.steps_until_tune = self.tune_interval
            self.accepted_sum[:] = 0

        delta = self.proposal_dist(rng=self.rng) * self.scaling

        if self.any_discrete:
            if self.all_discrete:
                delta = np.round(delta, 0).astype("int64")
                q0d = q0d.astype("int64")
                q = (q0d + delta).astype("int64")
            else:
                delta[self.discrete] = np.round(delta[self.discrete], 0)
                q = q0d + delta
        else:
            q = floatX(q0d + delta)

        if self.elemwise_update:
            q0d = q0d.copy()
            q_temp = q0d.copy()
            # Shuffle order of updates (probably we don't need to do this in every step)
            self.rng.shuffle(self.enum_dims)
            for i in self.enum_dims:
                q_temp[i] = q[i]
                accept_rate_i = self.delta_logp(q_temp, q0d)
                q_temp_, accepted_i = metrop_select(accept_rate_i, q_temp, q0d, rng=self.rng)
                q_temp[i] = q0d[i] = q_temp_[i]
                self.accept_rate_iter[i] = accept_rate_i
                self.accepted_iter[i] = accepted_i
                self.accepted_sum[i] += accepted_i
            q = q_temp
        else:
            accept_rate = self.delta_logp(q, q0d)
            q, accepted = metrop_select(accept_rate, q, q0d, rng=self.rng)
            self.accept_rate_iter = accept_rate
            self.accepted_iter[0] = accepted
            self.accepted_sum += accepted

        self.steps_until_tune -= 1

        stats = {
            "tune": self.tune,
            "scaling": np.mean(self.scaling),
            "accept": np.mean(np.exp(self.accept_rate_iter)),
            "accepted": np.mean(self.accepted_iter),
        }

        return RaveledVars(q, point_map_info), [stats]

    @staticmethod
    def competence(var, has_grad):
        return Competence.COMPATIBLE

    @staticmethod
    def _progressbar_config(n_chains=1):
        columns = [
            TextColumn("{task.fields[tune]}", table_column=Column("Tuning", ratio=1)),
            TextColumn("{task.fields[scaling]:0.2f}", table_column=Column("Scaling", ratio=1)),
            TextColumn(
                "{task.fields[accept_rate]:0.2f}", table_column=Column("Accept Rate", ratio=1)
            ),
        ]

        stats = {
            "tune": [True] * n_chains,
            "scaling": [0] * n_chains,
            "accept_rate": [0.0] * n_chains,
        }

        return columns, stats

    @staticmethod
    def _make_progressbar_update_functions():
        def update_stats(step_stats):
            return {
                "accept_rate" if key == "accept" else key: step_stats[key]
                for key in ("tune", "accept", "scaling")
            }

        return (update_stats,)


def tune(scale, acc_rate):
    """
    Tune the scaling parameter for the proposal distribution.

    Uses the acceptance rate over the last tune_interval.

    Rate    Variance adaptation
    ----    -------------------
    <0.001        x 0.1
    <0.05         x 0.5
    <0.2          x 0.9
    >0.5          x 1.1
    >0.75         x 2
    >0.95         x 10

    """
    return scale * np.where(
        acc_rate < 0.001,
        # reduce by 90 percent
        0.1,
        np.where(
            acc_rate < 0.05,
            # reduce by 50 percent
            0.5,
            np.where(
                acc_rate < 0.2,
                # reduce by ten percent
                0.9,
                np.where(
                    acc_rate > 0.95,
                    # increase by factor of ten
                    10.0,
                    np.where(
                        acc_rate > 0.75,
                        # increase by double
                        2.0,
                        np.where(
                            acc_rate > 0.5,
                            # increase by ten percent
                            1.1,
                            # Do not change
                            1.0,
                        ),
                    ),
                ),
            ),
        ),
    )


@dataclass_state
class BinaryMetropolisState(StepMethodState):
    tune: bool
    accepted: int
    scaling: float
    tune_interval: int
    steps_until_tune: int


class BinaryMetropolis(ArrayStep):
    """Metropolis-Hastings optimized for binary variables.

    Unlike BinaryGibbsMetropolis, this step sampler proposes an update for all variable dimensions at once.

    This will perform a single logp evaluation per step, at the expense of a lower acceptance rate when
    the posteriors of the binary variables are highly correlated.

    The BinaryGibbsMetropolis (not this one) is the default step sampler for binary variables

    Parameters
    ----------
    vars: list
        List of value variables for sampler
    scaling: scalar or array
        Initial scale factor for proposal. Defaults to 1.
    tune: bool
        Flag for tuning. Defaults to True.
    tune_interval: int
        The frequency of tuning. Defaults to 100 iterations.
    model: PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).
    rng: RandomGenerator
        An object that can produce be used to produce the step method's
        :py:class:`~numpy.random.Generator` object. Refer to
        :py:func:`pymc.util.get_random_generator` for more information.
    """

    name = "binary_metropolis"

    stats_dtypes_shapes = {
        "accept": (np.float64, []),
        "tune": (bool, []),
        "p_jump": (np.float64, []),
    }

    _state_class = BinaryMetropolisState

    def __init__(
        self,
        vars,
        *,
        scaling=1.0,
        tune=True,
        tune_interval=100,
        model=None,
        rng=None,
        initial_point: PointType | None = None,
        compile_kwargs: dict | None = None,
        blocked: bool = True,
    ):
        model = pm.modelcontext(model)

        self.scaling = scaling
        self.tune = tune
        self.tune_interval = tune_interval
        self.steps_until_tune = tune_interval
        self.accepted = 0

        vars = get_value_vars_from_user_vars(vars, model)

        if not all(v.dtype in pm.discrete_types for v in vars):
            raise ValueError("All variables must be Bernoulli for BinaryMetropolis")

        if compile_kwargs is None:
            compile_kwargs = {}
        super().__init__(vars, [model.compile_logp(**compile_kwargs)], blocked=blocked, rng=rng)

    def astep(self, apoint: RaveledVars, *args) -> tuple[RaveledVars, StatsType]:
        logp = args[0]
        logp_q0 = logp(apoint)
        point_map_info = apoint.point_map_info
        q0 = apoint.data

        # Convert adaptive_scale_factor to a jump probability
        p_jump = 1.0 - 0.5**self.scaling

        rand_array = self.rng.random(q0.shape)
        q = np.copy(q0)
        # Locations where switches occur, according to p_jump
        switch_locs = rand_array < p_jump
        q[switch_locs] = True - q[switch_locs]
        logp_q = logp(RaveledVars(q, point_map_info))

        accept = logp_q - logp_q0
        q_new, accepted = metrop_select(accept, q, q0, rng=self.rng)
        self.accepted += accepted

        stats = {
            "tune": self.tune,
            "accept": np.exp(accept),
            "p_jump": p_jump,
        }

        return RaveledVars(q_new, point_map_info), [stats]

    @staticmethod
    def competence(var):
        """BinaryMetropolis is only suitable for binary (bool) and Categorical variables with k=1."""
        distribution = getattr(var.owner, "op", None)

        if isinstance(distribution, BernoulliRV):
            return Competence.COMPATIBLE

        if isinstance(distribution, CategoricalRV):
            # TODO: We could compute the initial value of `k`
            # if we had a model object.
            # k_graph = var.owner.inputs[-1].shape[-1]
            # (k_graph,), _ = rvs_to_value_vars((k_graph,), apply_transforms=True)
            # k = model.fn(k_graph)(initial_point)
            try:
                k = var.owner.inputs[-1].shape[-1].eval()
                if k == 2:
                    return Competence.COMPATIBLE
            except MissingInputError:
                pass
        return Competence.INCOMPATIBLE


@dataclass_state
class BinaryGibbsMetropolisState(StepMethodState):
    tune: bool
    transit_p: int
    shuffle_dims: bool
    order: list


class BinaryGibbsMetropolis(ArrayStep):
    """A Metropolis-within-Gibbs step method optimized for binary variables.

    Unlike BinaryMetropolis, this step sampler proposes a variable dimension update at a time.

    This will increase acceptance rate when the posteriors of the binary variables are highly correlated,
    at the expense of doing more logp evaluations per step.

    This is the default step sampler for binary variables.


    Parameters
    ----------
    vars: list
        List of value variables for sampler
    order: list or 'random'
        List of integers indicating the Gibbs update order
        e.g., [0, 2, 1, ...]. Default is random
    transit_p: float
        The diagonal of the transition kernel. A value > .5 gives anticorrelated proposals,
        which resulting in more efficient antithetical sampling. Default is 0.8
    model: PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).
    rng: RandomGenerator
        An object that can produce be used to produce the step method's
        :py:class:`~numpy.random.Generator` object. Refer to
        :py:func:`pymc.util.get_random_generator` for more information.
    """

    name = "binary_gibbs_metropolis"

    stats_dtypes_shapes = {
        "tune": (bool, []),
    }

    _state_class = BinaryGibbsMetropolisState

    def __init__(
        self,
        vars,
        *,
        order="random",
        transit_p=0.8,
        model=None,
        rng=None,
        initial_point: PointType | None = None,
        compile_kwargs: dict | None = None,
        blocked: bool = True,
    ):
        model = pm.modelcontext(model)

        # Doesn't actually tune, but it's required to emit a sampler stat
        # that indicates whether a draw was done in a tuning phase.
        self.tune = True
        # transition probabilities
        self.transit_p = transit_p

        vars = get_value_vars_from_user_vars(vars, model)

        if initial_point is None:
            initial_point = model.initial_point()
        self.dim = sum(initial_point[v.name].size for v in vars)

        if order == "random":
            self.shuffle_dims = True
            self.order = list(range(self.dim))
        else:
            if sorted(order) != list(range(self.dim)):
                raise ValueError("Argument 'order' has to be a permutation")
            self.shuffle_dims = False
            self.order = order

        if not all(v.dtype in pm.discrete_types for v in vars):
            raise ValueError("All variables must be binary for BinaryGibbsMetropolis")

        if compile_kwargs is None:
            compile_kwargs = {}

        super().__init__(vars, [model.compile_logp(**compile_kwargs)], blocked=blocked, rng=rng)

    def reset_tuning(self):
        # There are no tuning parameters in this step method.
        return

    def astep(self, apoint: RaveledVars, *args) -> tuple[RaveledVars, StatsType]:
        logp: Callable[[RaveledVars], np.ndarray] = args[0]
        order = self.order
        if self.shuffle_dims:
            self.rng.shuffle(order)

        q = RaveledVars(np.copy(apoint.data), apoint.point_map_info)

        logp_curr = logp(q)

        for idx in order:
            # No need to do metropolis update if the same value is proposed,
            # as you will get the same value regardless of accepted or reject
            if self.rng.random() < self.transit_p:
                curr_val, q.data[idx] = q.data[idx], True - q.data[idx]
                logp_prop = logp(q)
                q.data[idx], accepted = metrop_select(
                    logp_prop - logp_curr, q.data[idx], curr_val, rng=self.rng
                )
                if accepted:
                    logp_curr = logp_prop

        stats = {
            "tune": self.tune,
        }
        return q, [stats]

    @staticmethod
    def competence(var):
        """BinaryMetropolis is only suitable for Bernoulli and Categorical variables with k=2."""
        distribution = getattr(var.owner, "op", None)

        if isinstance(distribution, BernoulliRV):
            return Competence.IDEAL

        if isinstance(distribution, CategoricalRV):
            # TODO: We could compute the initial value of `k`
            # if we had a model object.
            # k_graph = var.owner.inputs[-1].shape[-1]
            # (k_graph,), _ = rvs_to_value_vars((k_graph,), apply_transforms=True)
            # k = model.fn(k_graph)(initial_point)
            try:
                k = var.owner.inputs[-1].shape[-1].eval()
                if k == 2:
                    return Competence.IDEAL
            except MissingInputError:
                pass
        return Competence.INCOMPATIBLE


@dataclass_state
class CategoricalGibbsMetropolisState(StepMethodState):
    shuffle_dims: bool
    dimcats: list[tuple]


class CategoricalGibbsMetropolis(ArrayStep):
    """A Metropolis-within-Gibbs step method optimized for categorical variables.

    This step method works for Bernoulli variables as well, but it is not
    optimized for them, like BinaryGibbsMetropolis is. Step method supports
    two types of proposals: A uniform proposal and a proportional proposal,
    which was introduced by Liu in his 1996 technical report
    "Metropolized Gibbs Sampler: An Improvement".
    """

    name = "categorical_gibbs_metropolis"

    stats_dtypes_shapes = {
        "tune": (bool, []),
    }

    _state_class = CategoricalGibbsMetropolisState

    def __init__(
        self,
        vars,
        *,
        proposal="uniform",
        order="random",
        model=None,
        rng: RandomGenerator = None,
        initial_point: PointType | None = None,
        compile_kwargs: dict | None = None,
        blocked: bool = True,
    ):
        model = pm.modelcontext(model)

        vars = get_value_vars_from_user_vars(vars, model)

        if initial_point is None:
            initial_point = model.initial_point()

        dimcats: list[tuple[int, int]] = []
        # The above variable is a list of pairs (aggregate dimension, number
        # of categories). For example, if vars = [x, y] with x being a 2-D
        # variable with M categories and y being a 3-D variable with N
        # categories, we will have dimcats = [(0, M), (1, M), (2, N), (3, N), (4, N)].
        for v in vars:
            v_init_val = initial_point[v.name]

            rv_var = model.values_to_rvs[v]
            distr = getattr(rv_var.owner, "op", None)

            if isinstance(distr, CategoricalRV):
                k_graph = rv_var.owner.inputs[-1].shape[-1]
                (k_graph,) = model.replace_rvs_by_values((k_graph,))
                k = model.compile_fn(k_graph, inputs=model.value_vars, on_unused_input="ignore")(
                    initial_point
                )
            elif isinstance(distr, BernoulliRV):
                k = 2
            else:
                raise ValueError(
                    "All variables must be categorical or binary" + "for CategoricalGibbsMetropolis"
                )
            start = len(dimcats)
            dimcats += [(dim, k) for dim in range(start, start + v_init_val.size)]

        if order == "random":
            self.shuffle_dims = True
            self.dimcats = dimcats
        else:
            if sorted(order) != list(range(len(dimcats))):
                raise ValueError("Argument 'order' has to be a permutation")
            self.shuffle_dims = False
            self.dimcats = [dimcats[j] for j in order]

        if proposal == "uniform":
            self.astep = self.astep_unif  # type: ignore[assignment]
        elif proposal == "proportional":
            # Use the optimized "Metropolized Gibbs Sampler" described in Liu96.
            self.astep = self.astep_prop  # type: ignore[assignment]
        else:
            raise ValueError("Argument 'proposal' should either be 'uniform' or 'proportional'")

        if compile_kwargs is None:
            compile_kwargs = {}
        super().__init__(vars, [model.compile_logp(**compile_kwargs)], blocked=blocked, rng=rng)

    def reset_tuning(self):
        # There are no tuning parameters in this step method.
        return

    def astep_unif(self, apoint: RaveledVars, *args) -> tuple[RaveledVars, StatsType]:
        logp = args[0]
        point_map_info = apoint.point_map_info
        q0 = apoint.data

        dimcats = self.dimcats
        if self.shuffle_dims:
            self.rng.shuffle(dimcats)

        q = RaveledVars(np.copy(q0), point_map_info)
        logp_curr = logp(q)

        for dim, k in dimcats:
            curr_val, q.data[dim] = q.data[dim], sample_except(k, q.data[dim], rng=self.rng)
            logp_prop = logp(q)
            q.data[dim], accepted = metrop_select(
                logp_prop - logp_curr, q.data[dim], curr_val, rng=self.rng
            )
            if accepted:
                logp_curr = logp_prop

        # This step doesn't have any tunable parameters
        return q, [{"tune": False}]

    def astep_prop(self, apoint: RaveledVars, *args) -> tuple[RaveledVars, StatsType]:
        logp = args[0]
        point_map_info = apoint.point_map_info
        q0 = apoint.data

        dimcats = self.dimcats
        if self.shuffle_dims:
            self.rng.shuffle(dimcats)

        q = RaveledVars(np.copy(q0), point_map_info)
        logp_curr = logp(q)

        for dim, k in dimcats:
            logp_curr = self.metropolis_proportional(q, logp, logp_curr, dim, k)

        # This step doesn't have any tunable parameters
        return q, [{"tune": False}]

    def astep(self, apoint: RaveledVars, *args) -> tuple[RaveledVars, StatsType]:
        raise NotImplementedError()

    def metropolis_proportional(self, q, logp, logp_curr, dim, k):
        given_cat = int(q.data[dim])
        log_probs = np.zeros(k)
        log_probs[given_cat] = logp_curr
        candidates = list(range(k))
        for candidate_cat in candidates:
            if candidate_cat != given_cat:
                q.data[dim] = candidate_cat
                log_probs[candidate_cat] = logp(q)
        probs = scipy.special.softmax(log_probs, axis=0)
        prob_curr, probs[given_cat] = probs[given_cat], 0.0
        probs /= 1.0 - prob_curr
        proposed_cat = self.rng.choice(candidates, p=probs)
        accept_ratio = (1.0 - prob_curr) / (1.0 - probs[proposed_cat])
        if not np.isfinite(accept_ratio) or self.rng.uniform() >= accept_ratio:
            q.data[dim] = given_cat
            return logp_curr
        q.data[dim] = proposed_cat
        return log_probs[proposed_cat]

    @staticmethod
    def competence(var):
        """CategoricalGibbsMetropolis is only suitable for Bernoulli and Categorical variables."""
        distribution = getattr(var.owner, "op", None)

        if isinstance(distribution, CategoricalRV):
            # TODO: We could compute the initial value of `k`
            # if we had a model object.
            # k_graph = var.owner.inputs[-1].shape[-1]
            # (k_graph,), _ = rvs_to_value_vars((k_graph,), apply_transforms=True)
            # k = model.fn(k_graph)(initial_point)
            try:
                k = var.owner.inputs[-1].shape[-1].eval()
                if k > 2:
                    return Competence.IDEAL
            except MissingInputError:
                pass

            return Competence.COMPATIBLE

        if isinstance(distribution, BernoulliRV):
            return Competence.COMPATIBLE

        return Competence.INCOMPATIBLE


@dataclass_state
class DEMetropolisState(StepMethodState):
    scaling: np.ndarray
    lamb: float
    tune: str | None
    tune_interval: int
    steps_until_tune: int
    accepted: int

    mode: Any = field(metadata={"frozen": True})


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
        S (and n). Defaults to NormalProposal(S).
    scaling: scalar or array
        Initial scale factor for epsilon. Defaults to 0.001
    tune: str
        Which hyperparameter to tune. Defaults to 'scaling', but can also be 'lambda' or None.
    tune_interval: int
        The frequency of tuning. Defaults to 100 iterations.
    model: PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).
    mode:  string or `Mode` instance.
        compilation mode passed to PyTensor functions
    rng: RandomGenerator
        An object that can produce be used to produce the step method's
        :py:class:`~numpy.random.Generator` object. Refer to
        :py:func:`pymc.util.get_random_generator` for more information.

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
    stats_dtypes_shapes = {
        "accept": (np.float64, []),
        "accepted": (bool, []),
        "tune": (bool, []),
        "scaling": (np.float64, []),
        "lambda": (np.float64, []),
    }

    _state_class = DEMetropolisState

    def __init__(
        self,
        vars=None,
        *,
        S=None,
        proposal_dist=None,
        lamb=None,
        scaling=0.001,
        tune: str | None = "scaling",
        tune_interval=100,
        model=None,
        mode=None,
        rng=None,
        initial_point: PointType | None = None,
        compile_kwargs: dict | None = None,
        blocked: bool = True,
    ):
        model = pm.modelcontext(model)
        if initial_point is None:
            initial_point = model.initial_point()
        initial_values_size = sum(initial_point[n.name].size for n in model.value_vars)

        if vars is None:
            vars = model.continuous_value_vars
        else:
            vars = get_value_vars_from_user_vars(vars, model)

        if S is None:
            S = np.ones(initial_values_size)

        if proposal_dist is not None:
            self.proposal_dist = proposal_dist(S)
        else:
            self.proposal_dist = NormalProposal(S)

        self.scaling = np.atleast_1d(scaling).astype("d")
        if lamb is None:
            # default to the optimal lambda for normally distributed targets
            lamb = 2.38 / np.sqrt(2 * initial_values_size)
        self.lamb = float(lamb)
        if tune not in {None, "scaling", "lambda"}:
            raise ValueError('The parameter "tune" must be one of {None, scaling, lambda}')
        self.tune = tune
        self.tune_interval = tune_interval
        self.steps_until_tune = tune_interval
        self.accepted = 0

        self.mode = mode

        shared = pm.make_shared_replacements(initial_point, vars, model)
        self.delta_logp = delta_logp(initial_point, model.logp(), vars, shared, compile_kwargs)
        super().__init__(vars, shared, blocked=blocked, rng=rng)

    def astep(self, q0: RaveledVars) -> tuple[RaveledVars, StatsType]:
        point_map_info = q0.point_map_info
        q0d = q0.data

        if not self.steps_until_tune and self.tune:
            if self.tune == "scaling":
                self.scaling = tune(self.scaling, self.accepted / float(self.tune_interval))
            elif self.tune == "lambda":
                self.lamb = tune(self.lamb, self.accepted / float(self.tune_interval))
            # Reset counter
            self.steps_until_tune = self.tune_interval
            self.accepted = 0

        epsilon = self.proposal_dist(rng=self.rng) * self.scaling

        # differential evolution proposal
        # select two other chains
        if self.other_chains is None:  # pragma: no cover
            raise RuntimeError("Population sampler has not been linked to the other chains")
        ir1, ir2 = self.rng.choice(self.other_chains, 2, replace=False)
        r1 = DictToArrayBijection.map(self.population[ir1])  # type: ignore[index]
        r2 = DictToArrayBijection.map(self.population[ir2])  # type: ignore[index]
        # propose a jump
        q = floatX(q0d + self.lamb * (r1.data - r2.data) + epsilon)

        accept = self.delta_logp(q, q0d)
        q_new, accepted = metrop_select(accept, q, q0d, rng=self.rng)
        self.accepted += accepted

        self.steps_until_tune -= 1

        stats = {
            "tune": self.tune,
            "scaling": self.scaling,
            "lambda": self.lamb,
            "accept": np.exp(accept),
            "accepted": accepted,
        }

        return RaveledVars(q_new, point_map_info), [stats]

    @staticmethod
    def competence(var, has_grad):
        if var.dtype in pm.discrete_types:
            return Competence.INCOMPATIBLE
        return Competence.COMPATIBLE


@dataclass_state
class DEMetropolisZState(StepMethodState):
    scaling: np.ndarray
    lamb: float
    tune: bool
    tune_target: str | None
    tune_interval: int
    steps_until_tune: int
    accepted: int
    _history: list

    _untuned_settings: dict[str, np.ndarray | float] = field(metadata={"frozen": True})
    mode: Any = field(metadata={"frozen": True})


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
        S (and n). Defaults to NormalProposal(S).
    scaling: scalar or array
        Initial scale factor for epsilon. Defaults to 0.001
    tune: str
        Which hyperparameter to tune. Defaults to 'scaling', but can also be 'lambda' or None.
    tune_interval: int
        The frequency of tuning. Defaults to 100 iterations.
    tune_drop_fraction: float
        Fraction of tuning steps that will be removed from the samplers history when the tuning ends.
        Defaults to 0.9 - keeping the last 10% of tuning steps for good mixing while removing 90% of
        potentially unconverged tuning positions.
    model: PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).
    mode:  string or `Mode` instance.
        compilation mode passed to PyTensor functions
    rng: RandomGenerator
        An object that can produce be used to produce the step method's
        :py:class:`~numpy.random.Generator` object. Refer to
        :py:func:`pymc.util.get_random_generator` for more information.

    References
    ----------
    .. [Braak2008] Cajo C.F. ter Braak (2008).
        Differential Evolution Markov Chain with snooker updater and fewer chains.
        Statistics and Computing
        `link <https://doi.org/10.1007/s11222-008-9104-9>`__
    """

    name = "DEMetropolisZ"

    default_blocked = True
    stats_dtypes_shapes = {
        "accept": (np.float64, []),
        "accepted": (bool, []),
        "tune": (bool, []),
        "scaling": (np.float64, []),
        "lambda": (np.float64, []),
    }

    _state_class = DEMetropolisZState

    def __init__(
        self,
        vars=None,
        *,
        S=None,
        proposal_dist=None,
        lamb=None,
        scaling=0.001,
        tune: str | None = "scaling",
        tune_interval=100,
        tune_drop_fraction: float = 0.9,
        model=None,
        initial_point: PointType | None = None,
        compile_kwargs: dict | None = None,
        mode=None,
        rng=None,
        blocked: bool = True,
    ):
        model = pm.modelcontext(model)
        if initial_point is None:
            initial_point = model.initial_point()
        initial_values_size = sum(initial_point[n.name].size for n in model.value_vars)

        if vars is None:
            vars = model.continuous_value_vars
        else:
            vars = get_value_vars_from_user_vars(vars, model)

        if S is None:
            S = np.ones(initial_values_size)

        if proposal_dist is not None:
            self.proposal_dist = proposal_dist(S)
        else:
            self.proposal_dist = NormalProposal(S)

        self.scaling = np.atleast_1d(scaling).astype("d")
        if lamb is None:
            # default to the optimal lambda for normally distributed targets
            lamb = 2.38 / np.sqrt(2 * initial_values_size)
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
        self._history: list[np.ndarray] = []
        # remember initial settings before tuning so they can be reset
        self._untuned_settings = {
            "scaling": self.scaling,
            "lamb": self.lamb,
            "steps_until_tune": tune_interval,
            "accepted": self.accepted,
        }

        self.mode = mode

        shared = pm.make_shared_replacements(initial_point, vars, model)
        self.delta_logp = delta_logp(initial_point, model.logp(), vars, shared, compile_kwargs)
        super().__init__(vars, shared, blocked=blocked, rng=rng)

    def reset_tuning(self):
        """Reset the tuned sampler parameters and history to their initial values."""
        # history can't be reset via the _untuned_settings dict because it's a list
        self._history = []
        for attr, initial_value in self._untuned_settings.items():
            setattr(self, attr, initial_value)
        return

    def astep(self, q0: RaveledVars) -> tuple[RaveledVars, StatsType]:
        point_map_info = q0.point_map_info
        q0d = q0.data

        # same tuning scheme as DEMetropolis
        if not self.steps_until_tune and self.tune:
            if self.tune_target == "scaling":
                self.scaling = tune(self.scaling, self.accepted / float(self.tune_interval))
            elif self.tune_target == "lambda":
                self.lamb = tune(self.lamb, self.accepted / float(self.tune_interval))
            # Reset counter
            self.steps_until_tune = self.tune_interval
            self.accepted = 0

        epsilon = self.proposal_dist(rng=self.rng) * self.scaling

        it = len(self._history)
        # use the DE-MCMC-Z proposal scheme as soon as the history has 2 entries
        if it > 1:
            # differential evolution proposal
            # select two other chains
            iz1 = self.rng.integers(it)
            iz2 = self.rng.integers(it)
            while iz2 == iz1:
                iz2 = self.rng.integers(it)

            z1 = self._history[iz1]
            z2 = self._history[iz2]
            # propose a jump
            q = floatX(q0d + self.lamb * (z1 - z2) + epsilon)
        else:
            # propose just with noise in the first 2 iterations
            q = floatX(q0d + epsilon)

        accept = self.delta_logp(q, q0d)
        q_new, accepted = metrop_select(accept, q, q0d, rng=self.rng)
        self.accepted += accepted
        self._history.append(q_new)

        self.steps_until_tune -= 1

        stats = {
            "tune": self.tune,
            "scaling": np.mean(self.scaling),
            "lambda": self.lamb,
            "accept": np.exp(accept),
            "accepted": accepted,
        }

        return RaveledVars(q_new, point_map_info), [stats]

    def stop_tuning(self):
        """Remove the first x% of the history at the end of the tuning phase.

        This is so future proposals are not informed by unconverged tuning iterations.
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


def sample_except(limit, excluded, rng: np.random.Generator):
    candidate = rng.choice(limit - 1)
    if candidate >= excluded:
        candidate += 1
    return candidate


def delta_logp(
    point: dict[str, np.ndarray],
    logp: pt.TensorVariable,
    vars: list[pt.TensorVariable],
    shared: dict[pt.TensorVariable, pt.sharedvar.TensorSharedVariable],
    compile_kwargs: dict | None,
) -> pytensor.compile.Function:
    [logp0], inarray0 = join_nonshared_inputs(
        point=point,
        outputs=[logp],
        inputs=vars,
        shared_inputs=cast(dict[Variable, Variable], shared),
    )

    tensor_type = inarray0.type
    inarray1 = tensor_type("inarray1")

    logp1 = CallableTensor(logp0)(inarray1)
    # Replace any potential duplicated RNG nodes
    (logp1,) = replace_rng_nodes((logp1,))

    if compile_kwargs is None:
        compile_kwargs = {}
    f = compile([inarray1, inarray0], logp1 - logp0, **compile_kwargs)
    f.trust_input = True
    return f
