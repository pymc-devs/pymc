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


import numpy as np

from scipy import stats

from pymc.stats.convergence import SamplerWarning, WarningType
from pymc.step_methods.state import DataClassState, WithSamplingState, dataclass_state


@dataclass_state
class StepSizeState(DataClassState):
    _log_step: np.ndarray
    _log_bar: np.ndarray
    _hbar: float
    _count: int
    _mu: np.ndarray
    _tuned_stats: list
    _initial_step: np.ndarray
    _target: float
    _k: float
    _t0: float
    _gamma: float


class DualAverageAdaptation(WithSamplingState):
    _state_class = StepSizeState

    def __init__(self, initial_step, target, gamma, k, t0):
        self._initial_step = initial_step
        self._target = target
        self._k = k
        self._t0 = t0
        self._gamma = gamma
        self.reset()

    def reset(self):
        self._log_step = np.log(self._initial_step)
        self._log_bar = self._log_step
        self._hbar = 0.0
        self._count = 1
        self._mu = np.log(10 * self._initial_step)
        self._tuned_stats = []

    def current(self, tune):
        if tune:
            return np.exp(self._log_step)
        else:
            return np.exp(self._log_bar)

    def update(self, accept_stat, tune):
        if not tune:
            self._tuned_stats.append(accept_stat)
            return

        count, k, t0 = self._count, self._k, self._t0
        w = 1.0 / (count + t0)
        self._hbar = (1 - w) * self._hbar + w * (self._target - accept_stat)

        self._log_step = self._mu - self._hbar * np.sqrt(count) / self._gamma
        mk = count**-k
        self._log_bar = mk * self._log_step + (1 - mk) * self._log_bar
        self._count += 1

    def stats(self):
        return {
            "step_size": np.exp(self._log_step),
            "step_size_bar": np.exp(self._log_bar),
        }

    def warnings(self):
        accept = np.array(self._tuned_stats)
        mean_accept = np.mean(accept)
        target_accept = self._target
        # Try to find a reasonable interval for acceptable acceptance
        # probabilities. Finding this was mostly trial and error.
        n_bound = min(100, len(accept))
        n_good, n_bad = mean_accept * n_bound, (1 - mean_accept) * n_bound
        lower, upper = stats.beta(n_good + 1, n_bad + 1).interval(0.95)
        if target_accept < lower or target_accept > upper:
            msg = (
                f"The acceptance probability does not match the target. "
                f"It is {mean_accept:0.4g}, but should be close to {target_accept:0.4g}. "
                f"Try to increase the number of tuning steps."
            )
            info = {"target": target_accept, "actual": mean_accept}
            warning = SamplerWarning(WarningType.BAD_ACCEPTANCE, msg, "warn", extra=info)
            return [warning]
        else:
            return []
