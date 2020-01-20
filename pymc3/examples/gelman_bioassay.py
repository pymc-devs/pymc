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
import pymc3 as pm
from numpy import ones, array

# Samples for each dose level
n = 5 * ones(4, dtype=int)
# Log-dose
dose = array([-.86, -.3, -.05, .73])

with pm.Model() as model:

    # Logit-linear model parameters
    alpha = pm.Normal('alpha', 0, sigma=100.)
    beta = pm.Normal('beta', 0, sigma=1.)

    # Calculate probabilities of death
    theta = pm.Deterministic('theta', pm.math.invlogit(alpha + beta * dose))

    # Data likelihood
    deaths = pm.Binomial('deaths', n=n, p=theta, observed=[0, 1, 3, 5])


def run(n=1000):
    if n == "short":
        n = 50
    with model:
        pm.sample(n, tune=1000)

if __name__ == '__main__':
    run()
