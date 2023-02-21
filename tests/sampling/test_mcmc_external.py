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
import pytest

from pymc import Model, Normal, sample

pytest.importorskip("nutpie")
pytest.importorskip("blackjax")
pytest.importorskip("numpyro")

# turns all warnings into errors for this module
pytestmark = pytest.mark.filterwarnings("error")


@pytest.mark.parametrize("nuts_sampler", ["pymc", "nutpie", "blackjax", "numpyro"])
def test_external_nuts_sampler(recwarn, nuts_sampler):
    with Model():
        Normal("x")

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

    warns = {
        (warn.category, warn.message.args[0])
        for warn in recwarn
        if warn.category is not FutureWarning
    }
    expected = set()
    if nuts_sampler != "pymc":
        expected.add((UserWarning, "Use of external NUTS sampler is still experimental"))
    if nuts_sampler == "nutpie":
        expected.add(
            (
                UserWarning,
                "`initvals` are currently not passed to nutpie sampler. "
                "Use `init_mean` kwarg following nutpie specification instead.",
            )
        )
    assert warns == expected

    assert idata1.posterior.chain.size == 2
    assert idata1.posterior.draw.size == 500
    np.testing.assert_array_equal(idata1.posterior.x, idata2.posterior.x)
