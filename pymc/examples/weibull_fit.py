"""
====================================================
Fitting the parameters of a statistical distribution
====================================================

This simple example shows how to fit the parameters of a
statistical distribution given
 - a series of experimental data,
 - priors for the parameters.

The statistical distribution chosen here is the Weibull
distribution, but the same can be done for any other
distribution.
"""

import pymc

"""
First, we will create a fake data set using some
fixed parameters. In real life, of course, you
already have the data  !
"""
alpha = 3
beta = 5
N = 100
dataset = pymc.rweibull(alpha,beta, N)

"""
Now we create a pymc model that defines the likelihood
of the data set and prior assumptions about the value
of the parameters.
"""
a = pymc.Uniform('a', lower=0, upper=10, value=5, doc='Weibull alpha parameter')
b = pymc.Uniform('b', lower=0, upper=10, value=5, doc='Weibull beta parameter')
like = pymc.Weibull('like', alpha=a, beta=b, value=dataset, observed=True)

"""
The last step is simply to sample the parameters a and b and analyze the
results.
"""
if __name__=='__main__':
    import pylab
    M = pymc.MCMC([a,b,like])
    M.sample(10000,5000,2)
    pymc.Matplot.plot(a)
    pymc.Matplot.plot(b)
    pylab.show()
