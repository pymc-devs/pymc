#   Copyright 2024 The PyMC Developers
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
import warnings

from collections.abc import Callable

import numpy as np
import pytensor.tensor as pt

from pytensor.gradient import NullTypeGradError
from scipy import optimize

import pymc as pm

__all__ = ["find_constrained_prior"]


def find_constrained_prior(
    distribution: pm.Distribution,
    lower: float,
    upper: float,
    init_guess: dict[str, float],
    mass: float = 0.95,
    fixed_params: dict[str, float] | None = None,
    mass_below_lower: float | None = None,
    **kwargs,
) -> dict[str, float]:
    """
    Find optimal parameters to get `mass` % of probability of a distribution between `lower` and `upper`.

    Note: only works for one- and two-parameter distributions, as there
    are exactly two constraints. Fix some combination of parameters
    if you want to use it on >=3-parameter distributions.

    Parameters
    ----------
    distribution : Distribution
        PyMC distribution you want to set a prior on.
        Needs to have a ``logcdf`` method implemented in PyMC.
    lower : float
        Lower bound to get `mass` % of probability of `pm_dist`.
    upper : float
        Upper bound to get `mass` % of probability of `pm_dist`.
    init_guess : dict of {str : float}
        Initial guess for ``scipy.optimize.least_squares`` to find the
        optimal parameters of `pm_dist` fitting the interval constraint.
        Must be a dictionary with the name of the PyMC distribution's
        parameter as keys and the initial guess as values.
    mass : float, default 0.95
        Share of the probability mass we want between ``lower`` and ``upper``.
        Defaults to 95%.
    fixed_params : str or float, optional, default None
        Only used when `pm_dist` has at least three parameters.
        Dictionary of fixed parameters, so that there are only 2 to optimize.
        For instance, for a StudentT, you fix nu to a constant and get the optimized
        mu and sigma.
    mass_below_lower : float, optional, default None
        The probability mass below the ``lower`` bound. If ``None``,
        defaults to ``(1 - mass) / 2``, which implies that the probability
        mass below the ``lower`` value will be equal to the probability
        mass above the ``upper`` value.

    Returns
    -------
    opt_params : dict
        The optimized distribution parameters as a dictionary.
        Dictionary keys are the parameter names and
        dictionary values are the optimized parameter values.

    Notes
    -----
    Optional keyword arguments can be passed to ``find_constrained_prior``. These will be
    delivered to the underlying call to :external:py:func:`scipy.optimize.minimize`.

    Examples
    --------
    .. code-block:: python

        # get parameters obeying constraints
        opt_params = pm.find_constrained_prior(
            pm.Gamma, lower=0.1, upper=0.4, mass=0.75, init_guess={"alpha": 1, "beta": 10}
        )

        # use these parameters to draw random samples
        samples = pm.Gamma.dist(**opt_params, size=100).eval()

        # use these parameters in a model
        with pm.Model():
            x = pm.Gamma("x", **opt_params)

        # specify fixed values before optimization
        opt_params = pm.find_constrained_prior(
            pm.StudentT,
            lower=0,
            upper=1,
            init_guess={"mu": 5, "sigma": 2},
            fixed_params={"nu": 7},
        )

    Under some circumstances, you might not want to have the same cumulative
    probability below the ``lower`` threshold and above the ``upper`` threshold.
    For example, you might want to constrain an Exponential distribution to
    find the parameter that yields 90% of the mass below the ``upper`` bound,
    and have zero mass below ``lower``. You can do that with the following call
    to ``find_constrained_prior``

    .. code-block:: python

        opt_params = pm.find_constrained_prior(
            pm.Exponential,
            lower=0,
            upper=3.0,
            mass=0.9,
            init_guess={"lam": 1},
            mass_below_lower=0,
        )
    """
    warnings.warn(
        "find_constrained_prior is deprecated and will be removed in a future version. "
        "Please use maxent function from PreliZ. "
        "https://preliz.readthedocs.io/en/latest/api_reference.html#preliz.unidimensional.maxent",
        FutureWarning,
        stacklevel=2,
    )

    assert 0.01 <= mass <= 0.99, (
        "This function optimizes the mass of the given distribution +/- "
        f"1%, so `mass` has to be between 0.01 and 0.99. You provided {mass}."
    )
    if mass_below_lower is None:
        mass_below_lower = (1 - mass) / 2

    # exit when any parameter is not scalar:
    if np.any(np.asarray(distribution.rv_op.ndims_params) != 0):
        raise NotImplementedError(
            "`pm.find_constrained_prior` does not work with non-scalar parameters yet.\n"
            "Feel free to open a pull request on PyMC repo if you really need this feature."
        )

    dist_params = pt.vector("dist_params")
    params_to_optim = {
        arg_name: dist_params[i] for arg_name, i in zip(init_guess.keys(), range(len(init_guess)))
    }

    if fixed_params is not None:
        params_to_optim.update(fixed_params)

    dist = distribution.dist(**params_to_optim)

    try:
        logcdf_lower = pm.logcdf(dist, pm.floatX(lower))
        logcdf_upper = pm.logcdf(dist, pm.floatX(upper))
    except AttributeError:
        raise AttributeError(
            f"You cannot use `find_constrained_prior` with {distribution} -- it doesn't have a logcdf "
            "method yet.\nOpen an issue or, even better, a pull request on PyMC repo if you really "
            "need it."
        )

    target = (pt.exp(logcdf_lower) - mass_below_lower) ** 2
    target_fn = pm.pytensorf.compile_pymc([dist_params], target, allow_input_downcast=True)

    constraint = pt.exp(logcdf_upper) - pt.exp(logcdf_lower)
    constraint_fn = pm.pytensorf.compile_pymc([dist_params], constraint, allow_input_downcast=True)

    jac: str | Callable
    constraint_jac: str | Callable
    try:
        pytensor_jac = pm.gradient(target, [dist_params])
        jac = pm.pytensorf.compile_pymc([dist_params], pytensor_jac, allow_input_downcast=True)
        pytensor_constraint_jac = pm.gradient(constraint, [dist_params])
        constraint_jac = pm.pytensorf.compile_pymc(
            [dist_params], pytensor_constraint_jac, allow_input_downcast=True
        )
    # when PyMC cannot compute the gradient
    except (NotImplementedError, NullTypeGradError):
        jac = "2-point"
        constraint_jac = "2-point"
    cons = optimize.NonlinearConstraint(constraint_fn, lb=mass, ub=mass, jac=constraint_jac)

    opt = optimize.minimize(
        target_fn, x0=list(init_guess.values()), jac=jac, constraints=cons, **kwargs
    )
    if not opt.success:
        raise ValueError(
            f"Optimization of parameters failed.\nOptimization termination details:\n{opt}"
        )

    # save optimal parameters
    opt_params = dict(zip(init_guess.keys(), opt.x))
    if fixed_params is not None:
        opt_params.update(fixed_params)
    return opt_params
