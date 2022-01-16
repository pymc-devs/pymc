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

# Modified from original implementation by Dominik Wabersich (2013)

import numpy as np
import numpy.random as nr

from pymc.aesaraf import inputvars
from pymc.blocking import RaveledVars
from pymc.model import modelcontext
from pymc.step_methods.arraystep import ArrayStep, Competence
from pymc.vartypes import continuous_types

__all__ = ["Slice"]

LOOP_ERR_MSG = "max slicer iters %d exceeded"


class Slice(ArrayStep):
    """
    Univariate slice sampler step method

    Parameters
    ----------
    vars: list
        List of value variables for sampler.
    w: float
        Initial width of slice (Defaults to 1).
    tune: bool
        Flag for tuning (Defaults to True).
    model: PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).

    """

    name = "slice"
    default_blocked = False

    def __init__(self, vars=None, w=1.0, tune=True, model=None, iter_limit=np.inf, **kwargs):
        self.model = modelcontext(model)
        self.w = w
        self.tune = tune
        self.n_tunes = 0.0
        self.iter_limit = iter_limit

        if vars is None:
            vars = self.model.cont_vars
        else:
            vars = [self.model.rvs_to_values.get(var, var) for var in vars]
        vars = inputvars(vars)

        super().__init__(vars, [self.model.compile_logp()], **kwargs)

    def astep(self, q0, logp):
        q0_val = q0.data
        self.w = np.resize(self.w, len(q0_val))  # this is a repmat
        q = np.copy(q0_val)  # TODO: find out if we need this
        ql = np.copy(q0_val)  # l for left boundary
        qr = np.copy(q0_val)  # r for right boudary
        for i in range(len(q0_val)):
            # uniformly sample from 0 to p(q), but in log space
            q_ra = RaveledVars(q, q0.point_map_info)
            y = logp(q_ra) - nr.standard_exponential()
            ql[i] = q[i] - nr.uniform(0, self.w[i])
            qr[i] = q[i] + self.w[i]
            # Stepping out procedure
            cnt = 0
            while y <= logp(
                RaveledVars(ql, q0.point_map_info)
            ):  # changed lt to leq  for locally uniform posteriors
                ql[i] -= self.w[i]
                cnt += 1
                if cnt > self.iter_limit:
                    raise RuntimeError(LOOP_ERR_MSG % self.iter_limit)
            cnt = 0
            while y <= logp(RaveledVars(qr, q0.point_map_info)):
                qr[i] += self.w[i]
                cnt += 1
                if cnt > self.iter_limit:
                    raise RuntimeError(LOOP_ERR_MSG % self.iter_limit)

            cnt = 0
            q[i] = nr.uniform(ql[i], qr[i])
            while logp(q_ra) < y:  # Changed leq to lt, to accomodate for locally flat posteriors
                # Sample uniformly from slice
                if q[i] > q0_val[i]:
                    qr[i] = q[i]
                elif q[i] < q0_val[i]:
                    ql[i] = q[i]
                q[i] = nr.uniform(ql[i], qr[i])
                cnt += 1
                if cnt > self.iter_limit:
                    raise RuntimeError(LOOP_ERR_MSG % self.iter_limit)

            if (
                self.tune
            ):  # I was under impression from MacKays lectures that slice width can be tuned without
                # breaking markovianness. Can we do it regardless of self.tune?(@madanh)
                self.w[i] = self.w[i] * (self.n_tunes / (self.n_tunes + 1)) + (qr[i] - ql[i]) / (
                    self.n_tunes + 1
                )  # same as before
                # unobvious and important: return qr and ql to the same point
                qr[i] = q[i]
                ql[i] = q[i]
        if self.tune:
            self.n_tunes += 1
        return q

    @staticmethod
    def competence(var, has_grad):
        if var.dtype in continuous_types:
            if not has_grad and var.ndim == 0:
                return Competence.PREFERRED
            return Competence.COMPATIBLE
        return Competence.INCOMPATIBLE
