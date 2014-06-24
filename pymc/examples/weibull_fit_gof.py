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

from pymc import rweibull, Uniform, Weibull

"""
First, we will create a fake data set using some
fixed parameters. In real life, of course, you
already have the data  !
"""
alpha = 3
beta = 5
N = 100
dataset = rweibull(alpha, beta, N)

"""
Now we create a pymc model that defines the likelihood
of the data set and prior assumptions about the value
of the parameters.
"""
a = Uniform('a', lower=0, upper=10, value=5, doc='Weibull alpha parameter')
b = Uniform('b', lower=0, upper=10, value=5, doc='Weibull beta parameter')
like = Weibull('like', alpha=a, beta=b, value=dataset, observed=True)
pred = Weibull('like', alpha=a, beta=b, value=dataset)

if __name__ == '__main__':

    from pymc import MCMC, Matplot

    # Sample the parameters a and b and analyze the results
    M = MCMC([a, b, like])
    M.sample(10000, 5000)
    Matplot.plot(M)
