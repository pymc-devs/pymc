import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt

# define gp, true parameter values
with pm.Model() as model:
    l = 0.3
    tau = 2.0
    cov = tau * pm.gp.cov.RBF(1, l)

n = 150
X = np.random.randn(n, 1)
K = theano.function([], cov.K(X, X))()

# generate fake data from GP with white noise (with variance sigma2)
sigma2 = 0.1
y = np.random.multivariate_normal(np.zeros(n), K + sigma2 * np.eye(n))

# infer gp parameter values
with pm.Model() as model:
    l = pm.HalfNormal('l')
    sigma2 = pm.HalfNormal('sigma2')
    tau = pm.HalfNormal('tau')

    f_cov = tau * pm.gp.cov.RBF(1, l)
    n_cov = sigma2 * tt.eye(n)

    y_obs = pm.gp.GP('y_obs', mu=0.0, cov=f_cov, observed=y)

with model:
    trace = pm.sample(2000, init='map')

pm.traceplot(trace, varnames=['l', 'tau', 'sigma2']);

