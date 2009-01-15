from pymc import stoch, data, discrete_stoch
from numpy import arange, ones, eye, sum, zeros, exp, concatenate
from numpy.random import normal
from pymc import normal_like,  uniform_like, JointMetropolis, DiscreteMetropolis

# Generate data
K_true = 5
T=20
A_true = -.4*exp(arange(0,-K_true,-1,dtype=float))
X_true = zeros(T,dtype=float)
X_init_val_true = 2.

X_true[:K_true] = X_init_val_true
for i in xrange(K_true,T):
    X_true[i] = sum(A_true * X_true[i-K_true:i]) + normal()

obs_interval = 5
X_obs_vals = X_true[::obs_interval] + normal(size=T/obs_interval)

if __name__ == '__main__':
    from pylab import *
    plot(X_true)
    show()

# Probability model
mu_x_init = 1.
tau_x_init = 1.

K_min = 0
K_max = 15

@discrete_stoch
def K(value=5, min = K_min, max = K_max):
    """K ~ uniform(min, max)"""
    return uniform_like(value, min, max)

A_init = zeros(K_max,dtype=float)
A_init[:K_true] = A_true
@stoch
def A(value=A_init, mu=-1.*ones(K_max,dtype=float), tau=ones(K_max,dtype=float)):
    """A ~ normal(mu, tau)"""
    return normal_like(value, mu, tau)

@stoch(trace=False)
def X(value=X_true, K=K, A=A, mu = mu_x_init, tau = tau_x_init):
    """Autoregression"""

    # Initial data
    logp=normal_like(value[:K], mu, tau)

    # Difference equation
    for i in xrange(K,T):
        logp += normal_like(value[i], sum(A[:K]*value[i-K:i]), 1.)

    return logp

@data
def X_obs(value=X_obs_vals, mu=X, tau=1.):
    """Data"""
    return normal_like(value, mu[::obs_interval], tau)


# JointMetropolis step method to handle X and A
oneatatime_scales = {X: .2, A: .2}
J = JointMetropolis([X, A], oneatatime_scales = oneatatime_scales)
# S = DiscreteOneAtATimeMetropolis(K, scale=.01)
