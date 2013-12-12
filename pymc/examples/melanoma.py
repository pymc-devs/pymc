"""
Exponential survival model for melanoma data, taken from
Bayesian Data Analysis (Ibrahim et al 2000) Example 2.1

# JAGS model
model {
    for(i in 1:N) {
        z[i] ~ dinterval(t[i], t.cen[i])
        t[i] ~ dweib(1,mu[i])

        eta[i] <- beta0 + beta[1]*trt[i]
        mu[i] <- exp(eta[i])

     }

     # Priors
     for(j in 1:p) {
         beta[j] ~ dnorm(0,.0001)
     }

     beta0 ~ dnorm(0,.0001)

}
"""

from pymc import Normal, Lambda, observed
from numpy import exp, log
from .melanoma_data import *

# Convert censoring indicators to indicators for failure event
failure = (censored == 0).astype(int)

# Intercept for survival rate
beta0 = Normal('beta0', mu=0.0, tau=0.0001, value=0.0)
# Treatment effect
beta1 = Normal('beta1', mu=0.0, tau=0.0001, value=0.0)

# Survival rates
lam = Lambda('lam', lambda b0=beta0, b1=beta1, t=treat: exp(b0 + b1 * t))


@observed
def survival(value=t, lam=lam, f=failure):
    """Exponential survival likelihood, accounting for censoring"""
    return sum(f * log(lam) - lam * value)

if __name__ == '__main__':
    from pymc import MCMC, Matplot

    # Instantiate model
    M = MCMC([beta0, beta1, lam, survival])
    # Sample
    M.sample(10000, 5000)
    # Plot traces
    Matplot.plot(M)
