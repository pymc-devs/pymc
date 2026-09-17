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

import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("error")

import pymc as pm


@pytest.mark.parametrize(
    "distribution, lower, upper, init_guess, fixed_params, mass_below_lower",
    [
        (pm.Gamma, 0.1, 0.4, {"alpha": 1, "beta": 10}, {}, None),
        (pm.Normal, 155, 180, {"mu": 170, "sigma": 3}, {}, None),
        (pm.StudentT, 0.1, 0.4, {"mu": 10, "sigma": 3}, {"nu": 7}, None),
        (pm.StudentT, 0, 1, {"mu": 5, "sigma": 2, "nu": 7}, {}, None),
        (pm.Exponential, 0, 1, {"lam": 1}, {}, 0),
        (pm.HalfNormal, 0, 1, {"sigma": 1}, {}, 0),
        (pm.Binomial, 0, 8, {"p": 0.5}, {"n": 10}, None),
        (pm.Poisson, 1, 15, {"mu": 10}, {}, None),
        (pm.Poisson, 19, 41, {"mu": 30}, {}, None),
    ],
)
@pytest.mark.parametrize("mass", [0.5, 0.75, 0.95])
def test_find_constrained_prior(
    distribution, lower, upper, init_guess, fixed_params, mass, mass_below_lower
):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        with pytest.warns(FutureWarning, match="find_constrained_prior is deprecated"):
            opt_params = pm.find_constrained_prior(
                distribution,
                lower=lower,
                upper=upper,
                mass=mass,
                init_guess=init_guess,
                fixed_params=fixed_params,
                mass_below_lower=mass_below_lower,
            )
    assert isinstance(opt_params, dict)


@pytest.mark.parametrize(
    "distribution, lower, upper, init_guess, fixed_params",
    [
        (pm.Gamma, 0.1, 0.4, {"alpha": 1}, {"beta": 10}),
        (pm.Exponential, 0.1, 1, {"lam": 1}, {}),
        (pm.Binomial, 0, 2, {"p": 0.8}, {"n": 10}),
    ],
)
def test_find_constrained_prior_error_too_large(
    distribution, lower, upper, init_guess, fixed_params
):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        with pytest.warns(FutureWarning, match="find_constrained_prior is deprecated"):
            with pytest.raises(
                ValueError, match="Optimization of parameters failed.\nOptimization termination details:\n"
            ):
                pm.find_constrained_prior(
                    distribution,
                    lower=lower,
                    upper=upper,
                    mass=0.95,
                    init_guess=init_guess,
                    fixed_params=fixed_params,
                )


def test_find_constrained_prior_input_errors():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        with pytest.warns(FutureWarning, match="find_constrained_prior is deprecated"):
            with pytest.raises(TypeError, match="required positional argument"):
                pm.find_constrained_prior(
                    pm.StudentT,
                    lower=0.1,
                    upper=0.4,
                    mass=0.95,
                    init_guess={"mu": 170, "sigma": 3},
                )