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

import numpy as np
import numpy.testing as npt
import pytest

from pymc import Data, Model, Normal, sample


@pytest.mark.parametrize("nuts_sampler", ["pymc", "nutpie", "blackjax", "numpyro"])
def test_external_nuts_sampler(recwarn, nuts_sampler):
    if nuts_sampler != "pymc":
        pytest.importorskip(nuts_sampler)

    with Model():
        x = Normal("x", 100, 5)
        y = Data("y", [1, 2, 3, 4])
        Data("z", [100, 190, 310, 405])

        Normal("L", mu=x, sigma=0.1, observed=y)

        kwargs = dict(
            nuts_sampler=nuts_sampler,
            random_seed=123,
            chains=2,
            tune=500,
            draws=500,
            progressbar=False,
            initvals={"x": 0.0},
        )

        idata1 = sample(**kwargs)
        idata2 = sample(**kwargs)

        reference_kwargs = kwargs.copy()
        reference_kwargs["nuts_sampler"] = "pymc"
        idata_reference = sample(**reference_kwargs)

    warns = {
        (warn.category, warn.message.args[0])
        for warn in recwarn
        if warn.category not in (FutureWarning, DeprecationWarning, RuntimeWarning)
    }
    expected = set()
    if nuts_sampler == "nutpie":
        expected.add(
            (
                UserWarning,
                "`initvals` are currently not passed to nutpie sampler. "
                "Use `init_mean` kwarg following nutpie specification instead.",
            )
        )
    assert warns == expected
    assert "y" in idata1.constant_data
    assert "z" in idata1.constant_data
    assert "L" in idata1.observed_data
    assert idata1.posterior.chain.size == 2
    assert idata1.posterior.draw.size == 500
    assert idata1.posterior.tuning_steps == 500
    np.testing.assert_array_equal(idata1.posterior.x, idata2.posterior.x)

    assert idata_reference.posterior.attrs.keys() == idata1.posterior.attrs.keys()


@pytest.mark.parametrize("nuts_sampler", ["blackjax", "numpyro"])
def test_external_nuts_chunking(nuts_sampler):
    # blackjax should have same sampling whether chunked or not
    pytest.importorskip(nuts_sampler)

    with Model():
        x = Normal("x", 100, 5)
        y = Data("y", [1, 2, 3, 4])

        Normal("L", mu=x, sigma=0.1, observed=y)

        base_kwargs = dict(
            nuts_sampler=nuts_sampler,
            random_seed=123,
            chains=2,
            tune=500,
            draws=500,
            progressbar=False,
            initvals={"x": 0.0},
        )
        chunk_kwargs = {**base_kwargs, **{"nuts_sampler_kwargs": {"num_chunks": 10}}}

        idata1 = sample(**base_kwargs)
        idata2 = sample(**chunk_kwargs)

    np.testing.assert_array_equal(idata1.posterior.x, idata2.posterior.x)
    assert idata1.posterior.attrs.keys() == idata2.posterior.attrs.keys()


def test_step_args():
    with Model() as model:
        a = Normal("a")
        idata = sample(
            nuts_sampler="numpyro",
            target_accept=0.5,
            nuts={"max_treedepth": 10},
            random_seed=1410,
        )

    npt.assert_almost_equal(idata.sample_stats.acceptance_rate.mean(), 0.5, decimal=1)
