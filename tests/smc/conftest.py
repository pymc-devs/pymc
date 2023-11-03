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
import pytensor.tensor as pt

import pymc as pm


def two_gaussians_model():
    n = 4
    mu1 = np.ones(n) * 0.5
    mu2 = -mu1

    stdev = 0.1
    sigma = np.power(stdev, 2) * np.eye(n)
    isigma = np.linalg.inv(sigma)
    dsigma = np.linalg.det(sigma)

    w1 = stdev
    w2 = 1 - stdev

    def two_gaussians(x):
        """
        Mixture of gaussians likelihood
        """
        log_like1 = (
            -0.5 * n * pt.log(2 * np.pi)
            - 0.5 * pt.log(dsigma)
            - 0.5 * (x - mu1).T.dot(isigma).dot(x - mu1)
        )
        log_like2 = (
            -0.5 * n * pt.log(2 * np.pi)
            - 0.5 * pt.log(dsigma)
            - 0.5 * (x - mu2).T.dot(isigma).dot(x - mu2)
        )
        return pt.log(w1 * pt.exp(log_like1) + w2 * pt.exp(log_like2))

    with pm.Model() as m:
        X = pm.Uniform("X", lower=-2, upper=2.0, shape=n)
        llk = pm.Potential("muh", two_gaussians(X))

    return m, mu1


def fast_model():
    with pm.Model() as m:
        x = pm.Normal("x", 0, 1)
        y = pm.Normal("y", x, 1, observed=0)
    return m
