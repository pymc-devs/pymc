from  utils import * 
import numpy as np 
from pymc import *


data  = pd.read_csv('srrs2.dat', index_col = 'idnum')

cty_data= pd.read_csv('cty.dat')


mn = data['state'] == 'MN'
mn = data[data.state == 'MN']


radon = data.activity
floor = data.floor

lradon = np.log(np.where(radon ==0, .1, radon))
data['fips'] = data.stfips*1000 + data.cntyfips

cty_data['fips'] = cty_data['stfips']*1000 + cty_data['ctfips']

ufull = cty_data['Uppm'][searchsorted(cty_data['fips'], fips)]


stfips,ctfips,st,cty,lon,lat,Uppm = readtable('cty.dat', delimiter = ',' )

ufips = np.unique(fips)
n = ufips.shape[0]
group = np.searchsorted(ufips, fips)
obs_means = np.array([np.mean(lradon[fips == fip]) for fip in np.unique(fips)])

obs_means = data.groupby('fips').mean()



model = Model()
Var = model.Var
Data = model.Data


groupmean = Var('groupmean', Normal(0, (10)**-2))
#as recommended by "Prior distributions for variance parameters in hierarchical models"
groupsd = Var('groupsd', Uniform(0,10))

sd = Var('sd', Uniform(0, 10))

floor_m = Var('floor_m', Normal(0, 5.** -2))
means = Var('means', Normal(groupmean, groupsd ** -2), n)

start = {'groupmean' : np.mean(obs_means )[None],
         'groupsd' : np.std(obs_means)[None], 
         'sd' : np.std(lradon)[None], 
         'means' : obs_means,
         'floor_m' : np.array([0.]),
         }

#the gradient of indexing into an array is generally slow unless you have the experimental branch of theano
Data(lradon, Normal(floor*floor_m + means[group], sd**-2))

hess = np.diag(approx_hess(model, start))

step_method = hmc_step(model, model.vars, hess, .2, 1.0, is_cov = False)
#step_method = hmc_lowflip_step(model, model.vars, hess, .5, a = .5)

trace, state, t = sample(3000, step_method, start)

print  " took: ", t
