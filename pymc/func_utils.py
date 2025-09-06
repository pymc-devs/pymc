#   Copyright 2024 - present The PyMC Developers
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

from sys import modules

try:
    from preliz.unidimensional import maxent
except ImportError as e:
    raise ImportError(
        "The `find_constrained_prior` function requires the `preliz` package. "
        "You can install it via `pip install preliz` or `conda install -c conda-forge preliz`."
    ) from e

import pymc as pm

__all__ = ["find_constrained_prior"]


def find_constrained_prior(
    distribution: pm.Distribution,
    lower: float,
    upper: float,
    init_guess: dict[str, float] | None = None,
    mass: float = 0.95,
    fixed_params: dict[str, float] | None = None,
    fixed_stat: tuple[str, float] | None = None,
    mass_below_lower: float | None = None,
    **kwargs,
) -> dict[str, float]:
    """
    Find the maximum entropy distribution that satisfies the constraints.

    Find the maximum entropy distribution with `mass` in the interval
    defined by the `lower` and `upper` end-points.

    Additional constraints can be set via `fixed_params` and `fixed_stat`.

    Parameters
    ----------
    distribution : Distribution
        PyMC distribution you want to set a prior on.
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
    fixed_stat: tuple
        Summary statistic to fix. The first element should be a name and the second a
        numerical value. Valid names are: "mean", "mode", "median", "var", "std",
        "skewness", "kurtosis". Defaults to None.

    Returns
    -------
    opt_params : dict
        The optimized distribution parameters as a dictionary.
        Dictionary keys are the parameter names and
        dictionary values are the optimized parameter values.

    Examples
    --------
    .. code-block:: python

        # get parameters obeying constraints
        opt_params = pm.find_constrained_prior(pm.Gamma, lower=0.1, upper=0.4, mass=0.75)

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
            fixed_params={"nu": 7},
        )

    .. code-block:: python

        opt_params = pm.find_constrained_prior(
            pm.Exponential,
            lower=0,
            upper=3.0,
            mass=0.9,
        )
    """
    if fixed_params is None:
        fixed_params = {}

    if init_guess is not None:
        warnings.warn(
            "The `init_guess` argument is deprecated and will be removed in a future version. "
            "The initial guess is determined automatically.",
            FutureWarning,
            stacklevel=2,
        )

    if mass_below_lower is not None:
        warnings.warn(
            "The `mass_below_lower` argument is deprecated and will be removed in a future version. "
            "Use `fixed_stat` or `mode` to add additional constraints.",
            FutureWarning,
            stacklevel=2,
        )

    if kwargs:
        warnings.warn(
            "Passing additional keyword arguments is deprecated and will be "
            "removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )

    dist = getattr(modules["preliz.distributions"], distribution.__name__)

    return maxent(
        dist(**fixed_params), lower=lower, upper=upper, mass=mass, fixed_stat=fixed_stat, plot=False
    ).params_dict
