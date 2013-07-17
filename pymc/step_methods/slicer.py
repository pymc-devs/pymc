# Modified from original implementation by Dominik Wabersich (2013)

from ..core import *
from arraystep import *
from numpy import floor, abs, atleast_1d, empty, isfinite, sum
from numpy.random import standard_exponential, random, uniform

__all__ = ['Slice']


def sub(x, i, val):
    y = x.copy()
    y[i] = val
    return y


class Slice(ArrayStep):

    """Slice sampler"""
    def __init__(self, vars, w=1, tune=True, model=None):

        model = modelcontext(model)
        self.vars = vars
        self.w = w
        self.tune = tune
        self.w_tune = []
        self.model = model

        super(Slice, self).__init__(vars, [model.logpc])

    def astep(self, q0, logp):

        q = q0.copy()
        k = q0.size
        self.w = np.resize(self.w, k)

        for i in range(k):
            y = logp(q0) - standard_exponential()

            # Stepping out procedure
            ql = q0.copy()
            ql[i] -= uniform(0, self.w[i])
            qr = q0.copy()
            qr[i] = ql[i] + self.w[i]

            yl = logp(ql)
            yr = logp(qr)

            while(y < yl):
                ql[i] -= self.w[i]
                yl = logp(ql)

            while(y < yr):
                qr[i] += self.w[i]
                yr = logp(qr)

            q_next = q0.copy()
            while True:

                # Sample uniformly from slice
                qi = uniform(ql[i], qr[i])
                q_next[i] = qi

                yi = logp(q_next)

                if yi > y:
                    q[i] = qi
                    break
                elif qi > q[i]:
                    qr[i] = qi
                elif qi < q[i]:
                    ql[i] = qi

        if self.tune:
            # Tune sampler parameters
            self.w_tune.append(abs(q0 - q))
            self.w = 2 * sum(self.w_tune, 0) / len(self.w_tune)

        return q