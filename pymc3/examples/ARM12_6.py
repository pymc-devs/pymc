import numpy as np
from pymc3 import *
import pandas as pd

data = pd.read_csv(get_data_file('pymc3.examples', 'data/srrs2.dat'))

cty_data = pd.read_csv(get_data_file('pymc3.examples', 'data/cty.dat'))

data = data[data.state == 'MN']

data['fips'] = data.stfips * 1000 + data.cntyfips
cty_data['fips'] = cty_data.stfips * 1000 + cty_data.ctfips


data['lradon'] = np.log(np.where(data.activity == 0, .1, data.activity))


data = data.merge(cty_data, 'inner', on='fips')

unique = data[['fips']].drop_duplicates()
unique['group'] = np.arange(len(unique))
unique.set_index('fips')
data = data.merge(unique, 'inner', on='fips')

obs_means = data.groupby('fips').lradon.mean()
n = len(obs_means)

lradon = np.array(data.lradon)
floor = np.array(data.floor)
group = np.array(data.group)


model = Model()
with model:
    groupmean = Normal('groupmean', 0, 10. ** -2.)
    # as recommended by "Prior distributions for variance parameters in
    # hierarchical models"
    groupsd = Uniform('groupsd', 0, 10.)

    sd = Uniform('sd', 0, 10.)

    floor_m = Normal('floor_m', 0, 5. ** -2.)
    means = Normal('means', groupmean, groupsd ** -2., shape=n)

    lr = Normal(
        'lr', floor * floor_m + means[group], sd ** -2., observed=lradon)


def run(n=3000):
    if n == "short":
        n = 50
    with model:
        start = {'groupmean': obs_means.mean(),
                 'groupsd_interval': 0,
                 'sd_interval': 0,
                 'means': np.array(obs_means),
                 'floor_m': 0.,
                 }

        start = find_MAP(start, [groupmean, sd, floor_m])
        step = NUTS(model.vars, scaling=start)

        trace = sample(n, step, start)

if __name__ == '__main__':
    run()
