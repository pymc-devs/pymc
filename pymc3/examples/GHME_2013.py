# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>
import pandas as pd
from pylab import *

from pymc3 import StudentT, Model, NUTS, Normal, find_MAP, trace, get_data_file
from pymc3.distributions.timeseries import GaussianRandomWalk

# <markdowncell>

# Data
# ----

# <codecell>

data = pd.read_csv(get_data_file('pymc3.examples', 'data/pancreatitis.csv'))
countries = ['CYP', 'DNK', 'ESP', 'FIN', 'GBR', 'ISL']
data = data[data.area.isin(countries)]

age = data['age'] = np.array(data.age_start + data.age_end) / 2
rate = data.value = data.value * 1000
group, countries = pd.factorize(data.area, order=countries)


ncountries = len(countries)

# <codecell>

for i, country in enumerate(countries):
    subplot(2, 3, i + 1)
    title(country)
    d = data[data.area == country]
    plot(d.age, d.value, '.')

    ylim(0, rate.max())

# <markdowncell>

# Model Specification
# -------------------

# <codecell>

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
    coeff_sd = StudentT('coeff_sd', 10, 1, 5**-2)

    y = GaussianRandomWalk('y', sd=coeff_sd, shape=(nknots, ncountries))

    p = interpolate(knots, y, age, group)

    sd = StudentT('sd', 10, 2, 5**-2)

    vals = Normal('vals', p, sd=sd, observed=rate)

# <markdowncell>

# Model Fitting
# -------------

# <codecell>

with model:
    s = find_MAP(vars=[sd, y])

    step = NUTS(scaling=s)
    trace = sample(100, step, s)

    s = trace[-1]

    step = NUTS(scaling=s)


def run(n=3000):
    if n == "short":
        n = 150
    with model:
        trace = sample(n, step, s)
    # <codecell>

    for i, country in enumerate(countries):
        subplot(2, 3, i + 1)
        title(country)

        d = data[data.area == country]
        plot(d.age, d.value, '.')
        plot(knots, trace[y][::5, :, i].T, color='r', alpha=.01)

        ylim(0, rate.max())


if __name__ == '__main__':
    run()
