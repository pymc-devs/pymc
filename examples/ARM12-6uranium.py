import numpy as np 
from pymc import *
import pandas as pd


data  = pd.read_csv('data/srrs2.dat')

cty_data= pd.read_csv('data/cty.dat')

data = data[data.state == 'MN']

data['fips'] = data.stfips*1000 + data.cntyfips
cty_data['fips'] = cty_data.stfips*1000 + cty_data.ctfips


data['lradon'] = np.log(np.where(data.activity==0, .1, data.activity))




data = data.merge(cty_data,'inner', on='fips')

unique = data[['fips']].drop_duplicates()   
unique['group'] = np.arange(len(unique))
unique.set_index('fips')
data = data.merge(unique, 'inner', on = 'fips')

obs_means = data.groupby('fips').lradon.mean()
n = len(obs_means)

lradon = np.array(data.lradon)
floor = np.array(data.floor)
group = np.array(data.group)
ufull = np.array(data.Uppm)


model = Model()
Var = model.Var
Data = model.Data


groupmean = Var('groupmean', Normal(0, (10.)**-2.))
#as recommended by "Prior distributions for variance parameters in hierarchical models"
groupsd = Var('groupsd', Uniform(0,10.))

sd = Var('sd', Uniform(0, 10.))

floor_m = Var('floor_m', Normal(0, 5.** -2.))
u_m = Var('u_m', Normal(0, 5.** -2))
means = Var('means', Normal(groupmean, groupsd ** -2.), n)



#the gradient of indexing into an array is generally slow unless you have the experimental branch of theano
Data(lradon, Normal(floor*floor_m + means[group] + ufull*u_m, sd**-2.))

start = {'groupmean' : obs_means.mean(),
         'groupsd' : obs_means.std(), 
         'sd' : data.groupby('group').lradon.std().mean(), 
         'means' : np.array(obs_means),
         'u_m' : np.array([.72]),
         'floor_m' : 0.,
         }

start = find_MAP(model, start, model.vars[:-1])
H = model.d2logpc()
hess = np.diag(H(start))

step = hmc_step(model, model.vars, hess, is_cov = False)

trace, state, t = sample(3000, step, start)

print  " took: ", t
