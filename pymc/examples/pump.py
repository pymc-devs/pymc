# Hierarchical Poisson failure rates example from Clark and Gelfand (2006), pp.23
# ma macneil; 13.03.09
# Import relevant modules
import pdb
from numpy import ones
from pymc import deterministic, stochastic, observed, Gamma, Exponential, poisson_like, MCMC, AdaptiveMetropolis, Matplot

#------------------------------------------------------------------ DATA
# Number of failures for pump system i
Yi = [5,1,5,14,3,19,1,1,4,22]
# Time over which failure measurements were made for pump system i
ti = [94.320, 15.720, 62.880, 125.760, 5.240, 31.440, 1.048, 1.048, 2.096, 10.480]
# Observed failure rate of pump system i
ri = [Yi[n]/ti[n] for n in range(len(Yi))]
# Number of observations
k=len(Yi)

#------------------------------------------------------------------ PRIORS
# alpha is the estimated gamma shape parameter to describe the distribution of thetas
alpha0=Exponential('alpha0', 1.0, value=1.)
# beta is the estimated gamma scale parameter to describe the distribution of thetas
beta0=Gamma('beta0', alpha=0.1, beta=1.0, value=1.)
# Theta values (point esitimates of the failure rate) per pump
theta = Gamma('theta', alpha=alpha0, beta=beta0, value=ones(k))
# overall Lambda = alpha0/beta0*tot_lambda = alpha0/beta0
# overall mtbf = beta0/alpha0*tot__mtbf = beta0/alpha0*

#------------------------------------------------------------------ MODEL
# Relate thetas to time
@deterministic(trace=True)
def L(theta=theta):
    return theta*ti
# Estimated value for shape of Poisson distribution given number of failures observed
@observed(dtype=int)
def y(value=Yi, lambduh=L):
    return poisson_like(value, lambduh) 
