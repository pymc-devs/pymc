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
import pandas as pd
import matplotlib.pyplot as plt

from pymc3 import HalfCauchy, Model, Normal, get_data, sample
from pymc3.distributions.timeseries import GaussianRandomWalk

data = pd.read_csv(get_data("pancreatitis.csv"))
countries = ["CYP", "DNK", "ESP", "FIN", "GBR", "ISL"]
data = data[data.area.isin(countries)]

age = data["age"] = np.array(data.age_start + data.age_end) / 2
rate = data.value = data.value * 1000
group, countries = pd.factorize(data.area, order=countries)


ncountries = len(countries)

for i, country in enumerate(countries):
    plt.subplot(2, 3, i + 1)
    plt.title(country)
    d = data[data.area == country]
    plt.plot(d.age, d.value, ".")

    plt.ylim(0, rate.max())


nknots = 10
knots = np.linspace(data.age_start.min(), data.age_end.max(), nknots)


def interpolate(x0, y0, x, group):
    x = np.array(x)
    group = np.array(group)

    idx = np.searchsorted(x0, x)
    dl = np.array(x - x0[idx - 1])
    dr = np.array(x0[idx] - x)
    d = dl + dr
    wl = dr / d

    return wl * y0[idx - 1, group] + (1 - wl) * y0[idx, group]


with Model() as model:
    coeff_sd = HalfCauchy("coeff_sd", 5)

    y = GaussianRandomWalk("y", sigma=coeff_sd, shape=(nknots, ncountries))

    p = interpolate(knots, y, age, group)

    sd = HalfCauchy("sd", 5)

    vals = Normal("vals", p, sigma=sd, observed=rate)


def run(n=3000):
    if n == "short":
        n = 150
    with model:
        trace = sample(n, tune=int(n / 2), init="advi+adapt_diag")

    for i, country in enumerate(countries):
        plt.subplot(2, 3, i + 1)
        plt.title(country)

        d = data[data.area == country]
        plt.plot(d.age, d.value, ".")
        plt.plot(knots, trace[y][::5, :, i].T, color="r", alpha=0.01)

        plt.ylim(0, rate.max())


if __name__ == "__main__":
    run()
