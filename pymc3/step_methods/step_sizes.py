import numpy as np
from scipy import stats

from pymc3.backends.report import SamplerWarning, WarningType


class DualAverageAdaptation:
    def __init__(self, initial_step, target, gamma, k, t0):
        self._log_step = np.log(initial_step)
        self._log_bar = self._log_step
        self._target = target
        self._hbar = 0.
        self._k = k
        self._t0 = t0
        self._count = 1
        self._mu = np.log(10 * initial_step)
        self._gamma = gamma
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
        w = 1. / (count + t0)
        self._hbar = ((1 - w) * self._hbar + w * (self._target - accept_stat))

        self._log_step = self._mu - self._hbar * np.sqrt(count) / self._gamma
        mk = count ** -k
        self._log_bar = mk * self._log_step + (1 - mk) * self._log_bar
        self._count += 1

    def stats(self):
        return {
            'step_size': np.exp(self._log_step),
            'step_size_bar': np.exp(self._log_bar),
        }

    def warnings(self):
        accept = np.array(self._tuned_stats)
        mean_accept = np.mean(accept)
        target_accept = self._target
        # Try to find a reasonable interval for acceptable acceptance
        # probabilities. Finding this was mostry trial and error.
        n_bound = min(100, len(accept))
        n_good, n_bad = mean_accept * n_bound, (1 - mean_accept) * n_bound
        lower, upper = stats.beta(n_good + 1, n_bad + 1).interval(0.95)
        if target_accept < lower or target_accept > upper:
            msg = ('The acceptance probability does not match the target. It '
                   'is %s, but should be close to %s. Try to increase the '
                   'number of tuning steps.'
                   % (mean_accept, target_accept))
            info = {'target': target_accept, 'actual': mean_accept}
            warning = SamplerWarning(
                WarningType.BAD_ACCEPTANCE, msg, 'warn', None, None, info)
            return [warning]
        else:
            return []
