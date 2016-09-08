# Modified from original implementation by Dominik Wabersich (2013)

import numpy as np
import numpy.random as nr

from .arraystep import ArrayStep, Competence
from ..model import modelcontext
from ..theanof import inputvars
from ..vartypes import continuous_types

__all__ = ['Slice']


class Slice(ArrayStep):
    """
    Univariate slice sampler step method

    Parameters
    ----------
    vars : list
        List of variables for sampler.
    w : float
        Initial width of slice (Defaults to 1).
    tune : bool
        Flag for tuning (Defaults to True).
    model : PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).

    """
    default_blocked = False

    def __init__(self, vars=None, w=1., tune=True, model=None, **kwargs):
        self.model = modelcontext(model)
        self.w = w
        self.tune = tune
        self.w_sum = 0
        self.n_tunes = 0

        if vars is None:
            vars = self.model.cont_vars
        vars = inputvars(vars)

        super(Slice, self).__init__(vars, [self.model.fastlogp], **kwargs)

    def astep(self, q0, logp):
        self.w = np.resize(self.w, len(q0))
        y = logp(q0) - nr.standard_exponential()

        # Stepping out procedure
        q_left = q0 - nr.uniform(0, self.w)
        q_right = q_left + self.w

        while (y < logp(q_left)).all():
            q_left -= self.w

        while (y < logp(q_right)).all():
            q_right += self.w

        q = nr.uniform(q_left, q_right, size=q_left.size)  # new variable to avoid copies
        while logp(q) <= y:
            # Sample uniformly from slice
            if (q > q0).all():
                q_right = q
            elif (q < q0).all():
                q_left = q
            q = nr.uniform(q_left, q_right, size=q_left.size)

        if self.tune:
            # Tune sampler parameters
            self.w_sum += np.abs(q0 - q)
            self.n_tunes += 1.
            self.w = 2. * self.w_sum / self.n_tunes
        return q

    @staticmethod
    def competence(var):
        if var.dtype in continuous_types:
            if not var.shape:
                return Competence.PREFERRED
            return Competence.COMPATIBLE
        return Competence.INCOMPATIBLE
