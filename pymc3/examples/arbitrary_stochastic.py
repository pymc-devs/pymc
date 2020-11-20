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
import pymc3 as pm
import theano.tensor as tt


# custom log-liklihood
def logp(failure, lam, value):
    return tt.sum(failure * tt.log(lam) - lam * value)


def build_model():
    # data
    failure = np.array([0.0, 1.0])
    value = np.array([1.0, 0.0])

    # model
    with pm.Model() as model:
        lam = pm.Exponential("lam", 1.0)
        pm.DensityDist("x", logp, observed={"failure": failure, "lam": lam, "value": value})
    return model


def run(n_samples=3000):
    model = build_model()
    with model:
        trace = pm.sample(n_samples)
    return trace


if __name__ == "__main__":
    run()
