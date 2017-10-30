import math


class DualAverageAdaptation(object):
    def __init__(self, initial_step, target, gamma, k, t0):
        self._log_step = math.log(initial_step)
        self._log_bar = 0.
        self._target = target
        self._hbar = 0.
        self._k = k
        self._t0 = t0
        self._count = 1
        self._mu = math.log(10 * initial_step)
        self._gamma = gamma

    def current(self, tune):
        if tune:
            return math.exp(self._log_step)
        else:
            return math.exp(self._log_bar)

    def update(self, accept_stat):
        count, k, t0 = self._count, self._k, self._t0
        w = 1. / (count + t0)
        self._hbar = ((1 - w) * self._hbar + w * (self._target - accept_stat))

        self._log_step = self._mu - self._hbar * math.sqrt(count) / self._gamma
        mk = count ** -k
        self._log_bar = mk * self._log_step + (1 - mk) * self._log_bar
        self._count += 1

    def stats(self):
        return {
            'step_size': math.exp(self._log_step),
            'step_size_bar': math.exp(self._log_bar),
        }
