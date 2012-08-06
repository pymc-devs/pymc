from numpy import *
from theano import *
import theano.tensor as t 
from __builtin__ import map
factors =[]

sd = t.dscalar()

means = t.dvector()

for i in range(85):
    factors.append( -0.5 * sd**-2  * (ones(10)-means[i])**2 + 0.5*t.log(0.5*(sd **-2)/pi))
    


factors = map(t.sum,factors)
logp = t.sum(t.stack(*factors))
#logp = t.add(*factors)

vars =[sd, means]
dlogp = function(vars, [grad(logp, v) for v in vars ] )

