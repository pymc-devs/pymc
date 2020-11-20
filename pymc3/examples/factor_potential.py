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

"""
You can add an arbitrary factor potential to the model likelihood using
pm.Potential. For example you can added Jacobian Adjustment using pm.Potential
when you do model reparameterization. It's similar to `target += u;` in
Stan.
"""


def build_model():
    with pm.Model() as model:
        x = pm.Normal("x", 1, 1)
        x2 = pm.Potential("x2", -(x ** 2))
    return model


def run(n=1000):
    model = build_model()
    if n == "short":
        n = 50
    with model:
        pm.sample(n)


if __name__ == "__main__":
    run()
