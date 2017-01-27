import numpy as np
import pandas as pd
from pylab import plot, ylim, subplot, title

import pymc3 as pm

data = pd.read_csv(pm.get_data_file('pymc3.examples', 'data/pancreatitis.csv'))
countries = ['CYP', 'DNK', 'ESP', 'FIN', 'GBR', 'ISL']
data = data[data.area.isin(countries)]

age = data['age'] = np.array(data.age_start + data.age_end) / 2
rate = data.value = data.value * 1000
group, countries = pd.factorize(data.area, order=countries)


for i, country in enumerate(countries):
    subplot(2, 3, i + 1)
    title(country)
    d = data[data.area == country]
    plot(d.age, d.value, '.')

    ylim(0, rate.max())

n_steps = 3000
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


with pm.Model() as model:
    coeff_sd = pm.StudentT('coeff_sd', 10, 1, 5**-2)
    y = pm.GaussianRandomWalk('y', sd=coeff_sd, shape=(nknots, len(countries)))
    p = interpolate(knots, y, age, group)
    sd = pm.StudentT('sd', 10, 2, 5**-2)
    vals = pm.Normal('vals', p, sd=sd, observed=rate)
    start = pm.find_MAP(vars=[sd, y])
    step = pm.NUTS(scaling=start)
    trace = pm.sample(100, step, start)
    start = trace[-1]
    step = pm.NUTS(scaling=start)
    trace = pm.sample(n_steps, step, step)

for i, country in enumerate(countries):
    subplot(2, 3, i + 1)
    title(country)

    d = data[data.area == country]
    plot(d.age, d.value, '.')
    plot(knots, trace[y][::5, :, i].T, color='r', alpha=.01)

    ylim(0, rate.max())
