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
import pytest

import pymc as pm


@pytest.mark.parametrize(
    "distribution, lower, upper, init_guess, fixed_params",
    [
        (pm.Gamma, 0.1, 0.4, {"alpha": 1, "beta": 10}, {}),
        (pm.Normal, 155, 180, {"mu": 170, "sigma": 3}, {}),
        (pm.StudentT, 0.1, 0.4, {"mu": 10, "sigma": 3}, {"nu": 7}),
        (pm.StudentT, 0, 1, {"mu": 5, "sigma": 2, "nu": 7}, {}),
        (pm.Exponential, 0, 1, {"lam": 1}, {}),
        (pm.HalfNormal, 0, 1, {"sigma": 1}, {}),
        (pm.Binomial, 0, 8, {"p": 0.5}, {"n": 10}),
        (pm.Poisson, 1, 15, {"mu": 10}, {}),
        (pm.Poisson, 19, 41, {"mu": 30}, {}),
    ],
)
@pytest.mark.parametrize("mass", [0.5, 0.75, 0.95])
def test_find_constrained_prior(distribution, lower, upper, init_guess, fixed_params, mass):
    with pytest.warns(None) as record:
        opt_params = pm.find_constrained_prior(
            distribution,
            lower=lower,
            upper=upper,
            mass=mass,
            init_guess=init_guess,
            fixed_params=fixed_params,
        )
    assert len(record) == 0

    opt_distribution = distribution.dist(**opt_params)
    mass_in_interval = (
        pm.math.exp(pm.logcdf(opt_distribution, upper))
        - pm.math.exp(pm.logcdf(opt_distribution, lower))
    ).eval()
    assert np.abs(mass_in_interval - mass) <= 1e-5


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
    with pytest.warns(UserWarning, match="instead of the requested 95%"):
        pm.find_constrained_prior(
            distribution,
            lower=lower,
            upper=upper,
            mass=0.95,
            init_guess=init_guess,
            fixed_params=fixed_params,
        )


def test_find_constrained_prior_input_errors():
    # missing param
    with pytest.raises(TypeError, match="required positional argument"):
        pm.find_constrained_prior(
            pm.StudentT,
            lower=0.1,
            upper=0.4,
            mass=0.95,
            init_guess={"mu": 170, "sigma": 3},
        )

    # mass too high
    with pytest.raises(AssertionError, match="has to be between 0.01 and 0.99"):
        pm.find_constrained_prior(
            pm.StudentT,
            lower=0.1,
            upper=0.4,
            mass=0.995,
            init_guess={"mu": 170, "sigma": 3},
            fixed_params={"nu": 7},
        )

    # mass too low
    with pytest.raises(AssertionError, match="has to be between 0.01 and 0.99"):
        pm.find_constrained_prior(
            pm.StudentT,
            lower=0.1,
            upper=0.4,
            mass=0.005,
            init_guess={"mu": 170, "sigma": 3},
            fixed_params={"nu": 7},
        )

    # non-scalar params
    with pytest.raises(NotImplementedError, match="does not work with non-scalar parameters yet"):
        pm.find_constrained_prior(
            pm.MvNormal,
            lower=0,
            upper=1,
            mass=0.95,
            init_guess={"mu": 5, "cov": np.asarray([[1, 0.2], [0.2, 1]])},
        )
