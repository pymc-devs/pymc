import pymc as pm
import numpy 


# ===========================
# = Convergence diagnostics =
# ===========================

# Simple dose-response model
n = [5]*4
dose = [-.86,-.3,-.05,.73]
x = [0,1,3,5]

alpha = pm.Normal('alpha', mu=0.0, tau=0.01)
beta = pm.Normal('beta', mu=0.0, tau=0.01)

@pm.deterministic
def theta(a=alpha, b=beta, d=dose):
    """theta = inv_logit(a+b)"""
    return pm.invlogit(a+b*d)

"""deaths ~ binomial(n, p)"""
deaths = pm.Binomial('deaths', n=n, p=theta, value=x, observed=True)

my_model = [alpha, beta, theta, deaths]

# Instantiate and run sampler
S = pm.MCMC(my_model)
S.sample(10000, burn=5000)

# Calculate and plot Geweke scores
scores = pm.geweke(S, intervals=20)
pm.Matplot.geweke_plot(scores)

# Geweke plot for a single parameter
trace = S.trace('alpha')[:]
alpha_scores = pm.geweke(trace, intervals=20)
pm.Matplot.geweke_plot(alpha_scores, 'alpha')

# Calculate Raftery-Lewis diagnostics
pm.raftery_lewis(S, q=0.025, r=0.01)

"""
Sample output:

========================
Raftery-Lewis Diagnostic
========================

937 iterations required (assuming independence) to achieve 0.01 accuracy
with 95 percent probability.

Thinning factor of 1 required to produce a first-order Markov chain.

39 iterations to be discarded at the beginning of the simulation (burn-in).

11380 subsequent iterations required.

Thinning factor of 11 required to produce an independence chain.
"""


# =========================
# = Autocorrelation plots =
# =========================

# Autocorrelation for the entire model
pm.Matplot.autocorrelation(S)

# Autocorrelation for just the slope parameter
pm.Matplot.autocorrelation(beta)


# ===================
# = Goodness-of-fit =
# ===================

# Simulate deaths, using posterior predictive distribution
@pm.deterministic
def deaths_sim(n=n, p=theta):
    """deaths_sim = rbinomial(n, p)"""
    return pm.rbinomial(n, p)

# Expected number of deaths, based on theta
expected_deaths = pm.Lambda('expected_deaths', lambda theta=theta: theta*n)

my_model = [alpha, beta, theta, deaths, deaths_sim, expected_deaths]

# Create MCMC sampler
S = pm.MCMC(my_model)
S.sample(1000)

x_sim = deaths_sim
x_exp = expected_deaths

# Create GOF plot
pm.Matplot.gof_plot(x_sim, x, name='x')

# Calculate and plot discrepancies
D = pm.diagnostics.discrepancy(x, x_sim, x_exp)
pm.Matplot.discrepancy_plot(D, name='D', report_p=True)

