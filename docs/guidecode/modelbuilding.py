"""Code from modelbuilding.tex"""

# Imports from tutorial
from pymc import DiscreteUniform, Exponential, deterministic, Poisson, Uniform
import numpy as np


s = DiscreteUniform('s', 1851, 1962, value=1900)


from pymc import stochastic, Stochastic, Binomial, MvNormalCov
#@stochastic(dtype=int)
#def s(value=1900, t_l=1851, t_h=1962):
#    """The switchpoint for the rate of disaster occurrence."""
#    if value > t_h or value < t_l:
#        # Invalid values
#        return -np.inf
#    else:
#        # Uniform log-likelihood
#        return -np.log(t_h - t_l + 1)



@stochastic(dtype=int)
def s(value=1900, t_l=1851, t_h=1962):
    """The switchpoint for the rate of disaster occurrence."""

    def logp(value, t_l, t_h):
        print value, t_l, t_h
        if value > t_h or value < t_l:
            return -np.inf
        else:
            return -np.log(t_h - t_l + 1)

    def random(t_l, t_h):
        return np.round( (t_l - t_h) * random() ) + t_l

    


def s_logp(value, t_l, t_h):
    if value > t_h or value < t_l:
        return -np.inf
    else:
        return -np.log(t_h - t_l + 1)

def s_rand(t_l, t_h):
    return np.round( (t_l - t_h) * random() ) + t_l

s = Stochastic( logp = s_logp,
                doc = 'The switchpoint for the rate of disaster occurrence.',
                name = 's',
                parents = {'t_l': 1851, 't_h': 1962},
                random = s_rand,
                trace = True,
                value = 1900,
                dtype=int,
                rseed = 1.,
                observed = False,
                cache_depth = 2,
                plot=True,
                verbose = 0)




#@stochastic(observed=True)
x = Binomial('x', value=7, n=10, p=.8, observed=True)


#@observed
#@stochastic(dtype=int)




x = MvNormalCov('x',np.ones(3),np.eye(3))
y = MvNormalCov('y',np.ones(3),np.eye(3))

print x+y
#<pymc.PyMCObjects.Deterministic '(x_add_y)' at 0x105c3bd10>

print x[0]
#<pymc.CommonDeterministics.Index 'x[0]' at 0x105c52390>

print x[1]+y[2]
#<pymc.PyMCObjects.Deterministic '(x[1]_add_y[2])' at 0x105c52410>


from pymc.examples.DisasterModel import *
@deterministic
def r(switchpoint = s, early_rate = e, late_rate = l):
    """The rate of disaster occurrence."""
    value = np.zeros(len(D))
    value[:switchpoint] = early_rate
    value[switchpoint:] = late_rate
    return value



def r_eval(switchpoint = s, early_rate = e, late_rate = l):
    value = np.zeros(len(D))
    value[:switchpoint] = early_rate
    value[switchpoint:] = late_rate
    return value

r = Deterministic(  eval = r_eval,
                    name = 'r',
                    parents = {'switchpoint': s, 'early_rate': e, 'late_rate': l},
                    doc = 'The rate of disaster occurrence.',
                    trace = True,
                    verbose = 0,
                    dtype=float,
                    plot=False,
                    cache_depth = 2)



N = 10
x_0 = Normal('x_0', mu=0, tau=1)

# Initialize array of stochastics
x = np.empty(N,dtype=object)
x[0] = x_0

# Loop over number of elements in N
for i in range(1,N):

   # Create Normal stochastic, whose mean is the previous element in x
   x[i] = Normal('x_%i' % i, mu=x[-1], tau=1)

@observed
@stochastic
def y(value = 1, mu = x, tau = 100):
    return normal_like(value, np.sum(mu**2), tau)



@potential
def psi_i(x_lo = x[i], x_hi = x[i+1]):
    """A pair potential"""
    return -(xlo - xhi)**2



def psi_i_logp(x_lo = x[i], x_hi = x[i+1]):
    return -(xlo - xhi)**2

psi_i = Potential(  logp = psi_i_logp,
                    name = 'psi_i',
                    parents = {'xlo': x[i], 'xhi': x[i+1]},
                    doc = 'A pair potential',
                    verbose = 0,
                    cache_depth = 2)



L = LazyFunction(fun, arguments)



fun(**arguments.value)
