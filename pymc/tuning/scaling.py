#   Copyright 2023 The PyMC Developers
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

from pymc.blocking import DictToArrayBijection
from pymc.model import Point, modelcontext
from pymc.pytensorf import hessian_diag
from pymc.util import get_var_name

__all__ = ["find_hessian", "trace_cov", "guess_scaling"]


def fixed_hessian(point, model=None):
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
    point = Point(point, model=model)

    rval = np.ones(DictToArrayBijection.map(point).size) / 10
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
    H = model.compile_d2logp(vars)
    return H(Point(point, filter_model_vars=True, model=model))


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
    H = model.compile_fn(hessian_diag(model.logp(), vars))
    return H(Point(point, model=model))


def guess_scaling(point, vars=None, model=None, scaling_bound=1e-8):
    """
    Applies scaling on Hessian matrix using the `adjust_scaling` function. If the
    `find_hessian_diag` function is not implemented for the model, it instead uses the `fixed_hessian` function to
    compute the Hessian.

    Parameters
    ----------
    point: dict
    vars: list
        Variables for which Hessian is to be calculated.
    model= Model (optional if in `with` context)
    scaling_bound = float
        specifies a lower bound on the scaling factor.

    Returns
    -------
    numpy.ndarray, shape (m, n)
        scaled Hessian matrix
    """
    model = modelcontext(model)
    try:
        h = find_hessian_diag(point, vars, model=model)
    except NotImplementedError:
        h = fixed_hessian(point, model=model)
    return adjust_scaling(h, scaling_bound)


def adjust_scaling(s, scaling_bound):
    """
    If the matrix has dimension less than 2,
    it applies scaling to a matrix `s` using the `adjust_precision`
    function to the matrix.
    Or, it applies precision adjustment to the eigenvalues of the matrix and then reconstructs the matrix
    using the eigenvectors and adjusted eigenvalues.
    """
    if s.ndim < 2:
        return adjust_precision(s, scaling_bound)
    else:
        val, vec = np.linalg.eigh(s)
        val = adjust_precision(val, scaling_bound)
        return eig_recompose(val, vec)


def adjust_precision(tau, scaling_bound=1e-8):
    """
    Adjusts a precision parameter `tau` by taking the
    logarithm of its magnitude, applying a lower and upper bound to this logarithm, and then exponentiating the
    result raised to the power of 2.
    """
    mag = sqrt(abs(tau))

    bounded = bound(log(mag), log(scaling_bound), log(1.0 / scaling_bound))
    return exp(bounded) ** 2


def bound(a, l, u):
    """
    Returns a lower and upper bound to a given value `a`, with bounds specified
    by the values `l` and `u`.
    """
    return np.maximum(np.minimum(a, u), l)


def eig_recompose(val, vec):
    """
    Returns reconstructed_matrix
    Parameters
    ----------
    val:constant(eigenvalue)
    vec:constant(eigenvector)

    Returns
    -------
    numpy.ndarray, shape (m, n)
        reconstructed_matrix = eigenvectors * diagonal_matrix_of_eigenvalues * eigenvectors^T
    """
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
