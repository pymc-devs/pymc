#   Copyright 2021 The PyMC Developers
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

from typing import List

import aesara
import aesara.tensor as aet
import numpy as np

from scipy import optimize

import pymc as pm

__all__ = ["find_optim_prior"]


def find_optim_prior(
    pm_dist: pm.Distribution,
    lower: float,
    upper: float,
    init_guess: List[float],
    mass: float = 0.95,
) -> np.ndarray:
    """
    Find optimal parameters to get `mass` % of probability
    of `pm_dist` between `lower` and `upper`.
    Note: only works for two-parameter distributions, as there
    are exactly two constraints.

    Parameters
    ----------
    pm_dist : pm.Distribution
        PyMC distribution you want to set a prior on.
        Needs to have a ``logcdf`` method implemented in PyMC.
    lower : float
        Lower bound to get `mass` % of probability of `pm_dist`.
    upper : float
        Upper bound to get `mass` % of probability of `pm_dist`.
    init_guess: List[float]
        Initial guess for ``scipy.optimize.least_squares`` to find the
        optimal parameters of `pm_dist` fitting the interval constraint.
    mass: float, default to 0.95
        Share of the probability mass we want between ``lower`` and ``upper``.
        Defaults to 95%.

    Returns
    -------
    The optimized distribution parameters as a numpy array.
    """
    if len(pm_dist.rv_op.ndims_params) != 2:
        raise NotImplementedError(
            "This function only works for two-parameter distributions, as there are exactly two "
            "constraints. "
        )

    dist_params = aet.vector("dist_params")
    lower_, upper_ = aet.scalars("lower", "upper")

    dist_ = pm_dist.dist(*[dist_params[i] for i in range(len(init_guess))])
    try:
        logcdf_lower = pm.logcdf(dist_, lower_)
        logcdf_upper = pm.logcdf(dist_, upper_)
    except AttributeError:
        raise AttributeError(
            f"You cannot use `find_optim_params` with {pm_dist} -- it doesn't have a logcdf "
            f"method yet. Open an issue or, even better, a PR on PyMC repo if you really need it."
        )

    alpha = 1 - mass
    out = [logcdf_lower - np.log(alpha / 2), logcdf_upper - np.log(1 - alpha / 2)]
    logcdf = aesara.function([dist_params, lower_, upper_], out)

    try:
        symb_grad = aet.as_tensor_variable([pm.gradient(o, [dist_params]) for o in out])
        jac = aesara.function([dist_params, lower_, upper_], symb_grad)

    # when PyMC cannot compute the gradient
    except Exception:
        jac = "2-point"

    opt = optimize.least_squares(logcdf, init_guess, jac=jac, args=(lower, upper))
    if not opt.success:
        raise ValueError("Optimization of parameters failed.")

    return opt.x
