'''
Created on Nov 25, 2009

@author: johnsalvatier
'''

from numpy import *
import pymc
ReData = arange(200, 3000, 25)
measured = 10.2 * (ReData )** .5 + random.normal(scale = 55, size = size(ReData))

def model():
    
    varlist = []
    
    sd =pymc.Uniform('sd', lower = 5, upper = 100) #pymc.Gamma("sd", 60 , beta =  2.0)
    varlist.append(sd)
    
    
    

    
    
    a = pymc.Uniform('a', lower = 0, upper = 100)#pymc.Normal('a', mu =  10, tau = 5**-2)
    b = pymc.Uniform('b', lower = .05, upper = 2.0)
    varlist.append(a)
    varlist.append(b)

    nonlinear = a  * (ReData )** b
    precision = sd **-2

    results = pymc.Normal('results', mu = nonlinear, tau = precision, value = measured, observed = True)
    
    return varlist