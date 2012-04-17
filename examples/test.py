'''
Created on Apr 16, 2012

@author: jsalvatier
'''
'''
Created on Mar 18, 2011

@author: johnsalvatier
'''
from mcex import * 
import theano as t 
import theano.tensor as ts
import numpy as np 
import pylab as pl


z = FreeVariable('z', (1,), 'float64')
z_prior = Beta(value = z, alpha = 2.5, beta = 3.5)

f = t.function( [z],[ts.sum(z_prior)])


x = np.linspace(-.1, 1.1, 100)

y = np.array([f([v])[0] for v in x])
print x, y
pl.figure()
pl.plot(x, y)
pl.show()