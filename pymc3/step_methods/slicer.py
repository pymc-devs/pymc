# Modified from original implementation by Dominik Wabersich (2013)

from ..core import *
from .arraystep import *
from ..distributions import *
from numpy import floor, abs, atleast_1d, empty, isfinite, sum
from numpy.random import standard_exponential, random, uniform

__all__ = ['Slice']


class Slice(ArrayStep):
    """Slice sampler"""
    default_blocked = False
    def __init__(self, vars=None, w=1, tune=True, model=None, **kwargs):

        model = modelcontext(model)

        if vars is None:
            vars = model.cont_vars
        vars = inputvars(vars)

        self.w = w
        self.tune = tune
        self.w_tune = []
        self.model = model

        super(Slice, self).__init__(vars, [model.fastlogp], **kwargs)

    def astep(self, q0, logp):

        q = q0.copy()
        self.w = np.resize(self.w, len(q))

        y = logp(q0) - standard_exponential()

        # Stepping out procedure
        ql = q0.copy()
        ql -= uniform(0, self.w)
        qr = q0.copy()
        qr = ql + self.w

        yl = logp(ql)
        yr = logp(qr)

        while((y < yl).all()):
            ql -= self.w
            yl = logp(ql)

        while((y < yr).all()):
            qr += self.w
            yr = logp(qr)

        q_next = q0.copy()
        while True:

            # Sample uniformly from slice
            qi = uniform(ql, qr, size=ql.size)

            yi = logp(qi)

            if yi > y:
                q = qi
                break
            elif (qi > q).all():
                qr = qi
            elif (qi < q).all():
                ql = qi

        if self.tune:
            # Tune sampler parameters
            self.w_tune.append(abs(q0 - q))
            self.w = 2 * sum(self.w_tune, 0) / len(self.w_tune)

        return q

    @staticmethod
    def competence(var):
        if var.dtype in continuous_types:
            if not var.shape:
                return 2
            return 1
        return 0