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

from numpy import exp, log, sqrt

from pymc3.blocking import ArrayOrdering, DictToArrayBijection
from pymc3.model import Point, modelcontext
from pymc3.theanof import hessian_diag, inputvars
from pymc3.util import get_var_name

__all__ = ["find_hessian", "trace_cov", "guess_scaling"]


def fixed_hessian(point, vars=None, model=None):
    """
    Returns a fixed Hessian for any chain location.

    Parameters
    ----------
    model: Model (optional if in `with` context)
    point: dict
    vars: list
        Variables for which Hessian is to be calculated.
    """

    model = modelcontext(model)
    if vars is None:
        vars = model.cont_vars
    vars = inputvars(vars)

    point = Point(point, model=model)

    bij = DictToArrayBijection(ArrayOrdering(vars), point)
    rval = np.ones(bij.map(point).size) / 10
    return rval


def find_hessian(point, vars=None, model=None):
    """
    Returns Hessian of logp at the point passed.

    Parameters
    ----------
    model: Model (optional if in `with` context)
    point: dict
    vars: list
        Variables for which Hessian is to be calculated.
    """
    model = modelcontext(model)
    H = model.fastd2logp(vars)
    return H(Point(point, model=model))


def find_hessian_diag(point, vars=None, model=None):
    """
    Returns Hessian of logp at the point passed.

    Parameters
    ----------
    model: Model (optional if in `with` context)
    point: dict
    vars: list
        Variables for which Hessian is to be calculated.
    """
    model = modelcontext(model)
    H = model.fastfn(hessian_diag(model.logpt, vars))
    return H(Point(point, model=model))


def guess_scaling(point, vars=None, model=None, scaling_bound=1e-8):
    model = modelcontext(model)
    try:
        h = find_hessian_diag(point, vars, model=model)
    except NotImplementedError:
        h = fixed_hessian(point, vars, model=model)
    return adjust_scaling(h, scaling_bound)


def adjust_scaling(s, scaling_bound):
    if s.ndim < 2:
        return adjust_precision(s, scaling_bound)
    else:
        val, vec = np.linalg.eigh(s)
        val = adjust_precision(val, scaling_bound)
        return eig_recompose(val, vec)


def adjust_precision(tau, scaling_bound=1e-8):
    mag = sqrt(abs(tau))

    bounded = bound(log(mag), log(scaling_bound), log(1.0 / scaling_bound))
    return exp(bounded) ** 2


def bound(a, l, u):
    return np.maximum(np.minimum(a, u), l)


def eig_recompose(val, vec):
    return vec.dot(np.diag(val)).dot(vec.T)


def trace_cov(trace, vars=None, model=None):
    """
    Calculate the flattened covariance matrix using a sample trace

    Useful if you want to base your covariance matrix for further sampling on some initial samples.

    Parameters
    ----------
    trace: Trace
    vars: list
        variables for which to calculate covariance matrix

    Returns
    -------
    r: array (n,n)
        covariance matrix
    """
    model = modelcontext(model)

    if model is not None:
        vars = model.free_RVs
    elif vars is None:
        vars = trace.varnames

    def flat_t(var):
        x = trace[get_var_name(var)]
        return x.reshape((x.shape[0], np.prod(x.shape[1:], dtype=int)))

    return np.cov(np.concatenate(list(map(flat_t, vars)), 1).T)
