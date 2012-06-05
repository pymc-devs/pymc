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

model = Model()

groupmean = AddVar(model, 'groupmean', Normal(0, 10**-2))

#as recommended by "Prior distributions for variance parameters in hierarchical models"
groupsd = AddVar(model, 'groupsd', Uniform(0,10))

sd = AddVar(model, 'sd', Uniform(0, 10))

means = AddVar(model, 'means', Normal(groupmean, groupsd ** -2), n)

id_means = []
for i, fip in enumerate(ufips):
    d = lradon[fips == fip]
    AddData(model, d , Normal(means[i], sd**-2))
    id_means.append(np.mean(d))
    




logp = model_logp(model)
def fn(var, idx, C):
    def lp(x):
        c = C.copy()
        v = c[var].copy()
        v[idx] = x 
        c[var] = v
        return logp(c)
    return lp

chain = {'groupmean' : np.mean(lradon)[None],
         'groupsd' : np.std(id_means)[None], 
         'sd' : np.std(lradon)[None], 
         'means' : np.array(id_means) }
#MAP, retall = find_MAP(model, chain, retall = True)

hmc_cov = approx_cov(model, chain) 

step_method = hmc_step(model, model.vars, hmc_cov)

ndraw = 3e3

history = NpHistory(model.vars, ndraw)
state, t = sample(ndraw, step_method, chain, history)
print "took :", t



