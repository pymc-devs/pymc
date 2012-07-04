from  utils import * 
import numpy as np 
from mcex import *


data = readtabledict('srrs2.dat', delimiter = ',', quotechar='"', skipinitialspace = True)
def dictmap(f, d):
    return dict((k, f(v)) for k, v in d.iteritems())

mn = data['state'] == 'MN'
data = dictmap(lambda v: v[mn], data)



radon = data['activity']
floor = data['floor']
lradon = np.log(np. where(radon ==0, .1, radon))
fips = data['stfips']*1000 + data['cntyfips']


stfips,ctfips,st,cty,lon,lat,Uppm = readtable('cty.dat', delimiter = ',' )

ufips = np.unique(fips)
n = ufips.shape[0]
group = np.searchsorted(ufips, fips)
obs_means = np.array([np.mean(lradon[fips == fip]) for fip in np.unique(fips)])

model = Model()

chain = {'groupmean' : np.mean(lradon)[None],
         'groupsd' : np.std(obs_means)[None], 
         'sd' : np.std(lradon)[None], 
         'means' : obs_means }

groupmean = AddVar(model, 'groupmean', Normal(0, 10**-2), test = chain['groupmean'])

#as recommended by "Prior distributions for variance parameters in hierarchical models"
groupsd = AddVar(model, 'groupsd', Uniform(0,10), test = chain['groupsd'])

sd = AddVar(model, 'sd', Uniform(0, 10), test = chain['sd'])

means = AddVar(model, 'means', Normal(groupmean, groupsd ** -2), n, test = chain['means'])

AddData(model, lradon, Normal(means[group], sd**-2))


def fn(var, idx, C, logp):
    def lp(x):
        c = C.copy()
        v = c[var].copy()
        v[idx] = x 
        c[var] = v
        return logp(c)
    return lp

MAP, retall = find_MAP(model, chain, retall = True)


C = approx_cov(model, MAP)
step_method = hmc_step(model, model.vars, C)

ndraw = 3e3

history = NpHistory(model.vars, ndraw)
state, t = sample(ndraw, step_method, chain, history)
print "took :", t



