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

# Modified from original implementation by Dominik Wabersich (2013)


import numpy as np

from rich.progress import TextColumn
from rich.table import Column

from pymc.blocking import RaveledVars, StatsType
from pymc.initial_point import PointType
from pymc.model import modelcontext
from pymc.pytensorf import compile, join_nonshared_inputs, make_shared_replacements
from pymc.step_methods.arraystep import ArrayStepShared
from pymc.step_methods.compound import Competence, StepMethodState
from pymc.step_methods.state import dataclass_state
from pymc.util import get_value_vars_from_user_vars
from pymc.vartypes import continuous_types

__all__ = ["Slice"]


@dataclass_state
class SliceState(StepMethodState):
    w: np.ndarray
    tune: bool
    n_tunes: float
    max_steps: int


class Slice(ArrayStepShared):
    """
    Univariate slice sampler step method.

    Parameters
    ----------
    vars : list, optional
        List of value variables for sampler.
    w : float, default 1.0
        Initial width of slice.
    tune : bool, default True
        Flag for tuning.
    model : Model, optional
        Optional model for sampling step. It will be taken from the context if not provided.
    max_steps : int, default 100
        Maximum interval width as a multiple of ``w``. Must be a positive integer.
        At most ``max_steps - 1`` stepping-out expansions are performed per coordinate,
        randomly divided between the left and right endpoints. Sampling and shrinkage
        proceed when this budget is exhausted, even if an endpoint is still inside
        the slice. The budget applies during both tuning and sampling. Set to 1 to
        disable stepping out.
    rng: RandomGenerator
        An object that can produce be used to produce the step method's
        :py:class:`~numpy.random.Generator` object. Refer to
        :py:func:`pymc.util.get_random_generator` for more information.

    References
    ----------
    .. [1] Neal, R. M. (2003). Slice sampling. The Annals of Statistics, 31(3),
       705-767. Stepping-out and shrinkage procedures in Figures 3 and 5.
       https://doi.org/10.1214/aos/1056562461

    """

    name = "slice"
    default_blocked = False
    stats_dtypes_shapes = {
        "nstep_out": (int, []),
        "nstep_in": (int, []),
    }

    _state_class = SliceState

    def __init__(
        self,
        vars=None,
        *,
        w=1.0,
        tune=True,
        model=None,
        max_steps: int = 100,
        rng=None,
        initial_point: PointType | None = None,
        compile_kwargs: dict | None = None,
        blocked: bool = False,  # Could be true since tuning is independent across dims?
    ):
        model = modelcontext(model)
        self.w = np.asarray(w).copy()
        self.tune = tune
        self.n_tunes = 0.0
        if not isinstance(max_steps, int | np.integer) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        self.max_steps = int(max_steps)

        if vars is None:
            vars = model.continuous_value_vars
        else:
            vars = get_value_vars_from_user_vars(vars, model)

        if initial_point is None:
            initial_point = model.initial_point()

        shared = make_shared_replacements(initial_point, vars, model)
        [logp], raveled_inp = join_nonshared_inputs(
            point=initial_point, outputs=[model.logp()], inputs=vars, shared_inputs=shared
        )
        if compile_kwargs is None:
            compile_kwargs = {}
        self.logp = compile([raveled_inp], logp, **compile_kwargs)
        self.logp.trust_input = True

        super().__init__(vars, shared, blocked=blocked, rng=rng)

    def astep(self, apoint: RaveledVars) -> tuple[RaveledVars, StatsType]:
        # The arguments are determined by the list passed via `super().__init__(..., fs, ...)`
        q0_val = apoint.data

        if q0_val.shape != self.w.shape:
            self.w = np.resize(self.w, len(q0_val))  # this is a repmat

        nstep_out = nstep_in = 0

        q = np.copy(q0_val)
        ql = np.copy(q0_val)  # l for left boundary
        qr = np.copy(q0_val)  # r for right boundary

        logp = self.logp
        for i, wi in enumerate(self.w):
            # uniformly sample from 0 to p(q), but in log space
            y = logp(q) - self.rng.standard_exponential()

            # Create initial interval
            ql[i] = q[i] - self.rng.uniform() * wi  # q[i] + r * w
            qr[i] = ql[i] + wi  # Equivalent to q[i] + (1-r) * w

            # Randomly split the expansion budget (Neal, 2003, Figure 3).
            # This random allocation is required for a reversible transition.
            max_steps_left = self.rng.integers(self.max_steps)
            max_steps_right = self.max_steps - 1 - max_steps_left

            # Stepping out procedure
            cnt = 0
            while cnt < max_steps_left and y <= logp(ql):
                ql[i] -= wi
                cnt += 1
            nstep_out += cnt

            cnt = 0
            while cnt < max_steps_right and y <= logp(qr):
                qr[i] += wi
                cnt += 1
            nstep_out += cnt

            cnt = 0
            q[i] = self.rng.uniform(ql[i], qr[i])
            while y > logp(q):  # Changed leq to lt, to accommodate for locally flat posteriors
                # Sample uniformly from slice
                if q[i] > q0_val[i]:
                    qr[i] = q[i]
                elif q[i] < q0_val[i]:
                    ql[i] = q[i]
                q[i] = self.rng.uniform(ql[i], qr[i])
                cnt += 1
            nstep_in += cnt

            if self.tune:
                # Update the mean interval width during tuning only.
                self.w[i] = wi * (self.n_tunes / (self.n_tunes + 1)) + (qr[i] - ql[i]) / (
                    self.n_tunes + 1
                )

            # Set qr and ql to the accepted points (they matter for subsequent iterations)
            qr[i] = ql[i] = q[i]

        if self.tune:
            self.n_tunes += 1

        stats = {
            "nstep_out": nstep_out,
            "nstep_in": nstep_in,
        }

        return RaveledVars(q, apoint.point_map_info), [stats]

    @staticmethod
    def competence(var, has_grad):
        if var.dtype in continuous_types:
            if not has_grad and var.ndim == 0:
                return Competence.PREFERRED
            return Competence.COMPATIBLE
        return Competence.INCOMPATIBLE

    @staticmethod
    def _progressbar_config(n_chains=1):
        columns = [
            TextColumn("{task.fields[nstep_out]}", table_column=Column("Steps out", ratio=1)),
            TextColumn("{task.fields[nstep_in]}", table_column=Column("Steps in", ratio=1)),
        ]

        stats = {"nstep_out": [0] * n_chains, "nstep_in": [0] * n_chains}

        return columns, stats

    @staticmethod
    def _make_progressbar_update_functions():
        def update_stats(step_stats):
            return {key: step_stats[key] for key in {"nstep_out", "nstep_in"}}

        return (update_stats,)
