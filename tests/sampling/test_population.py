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

import pytest

import pymc as pm

from pymc.step_methods.metropolis import DEMetropolis


class TestPopulationSamplers:
    steppers = [DEMetropolis]

    def test_checks_population_size(self):
        """Test that population samplers check the population size."""
        with pm.Model() as model:
            n = pm.Normal("n", mu=0, sigma=1)
            for stepper in TestPopulationSamplers.steppers:
                step = stepper()
                with pytest.raises(ValueError, match="requires at least 3 chains"):
                    pm.sample(draws=10, tune=10, chains=1, cores=1, step=step)
                # don't parallelize to make test faster
                pm.sample(
                    draws=10,
                    tune=10,
                    chains=4,
                    cores=1,
                    step=step,
                    compute_convergence_checks=False,
                )

    def test_demcmc_warning_on_small_populations(self):
        """Test that a warning is raised when n_chains <= n_dims"""
        with pm.Model() as model:
            pm.Normal("n", mu=0, sigma=1, size=(2, 3))
            with pytest.warns(UserWarning, match="more chains than dimensions"):
                pm.sample(
                    draws=5,
                    tune=5,
                    chains=6,
                    step=DEMetropolis(),
                    # make tests faster by not parallelizing; disable convergence warning
                    cores=1,
                    compute_convergence_checks=False,
                )

    def test_nonparallelized_chains_are_random(self):
        with pm.Model() as model:
            x = pm.Normal("x", 0, 1)
            for stepper in TestPopulationSamplers.steppers:
                step = stepper()
                idata = pm.sample(
                    chains=4,
                    cores=1,
                    draws=20,
                    tune=0,
                    step=DEMetropolis(),
                    compute_convergence_checks=False,
                )
                samples = idata.posterior["x"].values[:, 5]

                assert len(set(samples)) == 4, f"Parallelized {stepper} chains are identical."

    def test_parallelized_chains_are_random(self):
        with pm.Model() as model:
            x = pm.Normal("x", 0, 1)
            for stepper in TestPopulationSamplers.steppers:
                step = stepper()
                idata = pm.sample(
                    chains=4,
                    cores=4,
                    draws=20,
                    tune=0,
                    step=DEMetropolis(),
                    compute_convergence_checks=False,
                )
                samples = idata.posterior["x"].values[:, 5]

                assert len(set(samples)) == 4, f"Parallelized {stepper} chains are identical."
