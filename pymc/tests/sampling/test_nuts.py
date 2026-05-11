#   Copyright 2026 - present The PyMC Developers
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
import pymc as pm


def test_advi_plus_adapt_diag_grad_runs():
    with pm.Model():
        x = pm.Normal("x", mu=0, sigma=1)

        idata = pm.sample(
            draws=10,
            tune=10,
            chains=1,
            init="advi+adapt_diag_grad",
            progressbar=False,
        )

    assert "posterior" in idata


def test_backward_compat_adapt_diag_runs():
    with pm.Model():
        x = pm.Normal("x", mu=0, sigma=1)

        idata = pm.sample(
            draws=10,
            tune=10,
            chains=1,
            init="adapt_diag",
            progressbar=False,
        )

    assert "posterior" in idata


def test_auto_still_works():
    with pm.Model():
        x = pm.Normal("x", mu=0, sigma=1)

        idata = pm.sample(
            draws=10,
            tune=10,
            chains=1,
            init="auto",
            progressbar=False,
        )

    assert "posterior" in idata
