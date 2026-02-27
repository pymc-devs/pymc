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
import pytest

import pymc as pm


def test_incompatible_idata_coords_raise_before_sampling(monkeypatch):
    def fail_if_sampling_starts(*args, **kwargs):
        raise AssertionError("Sampling should not start when idata metadata is invalid")

    monkeypatch.setattr(pm.sampling.mcmc, "_sample_many", fail_if_sampling_starts)
    monkeypatch.setattr(pm.sampling.mcmc, "_mp_sample", fail_if_sampling_starts)

    with pm.Model(coords={"group": range(3)}):
        pm.Normal("x", dims="group")

        with pytest.raises(ValueError, match="Incompatible `idata_kwargs`"):
            pm.sample(
                draws=1,
                tune=1,
                chains=1,
                cores=1,
                idata_kwargs={"coords": {"group": range(2)}},
                compute_convergence_checks=False,
            )
