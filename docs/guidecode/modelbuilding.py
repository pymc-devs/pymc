
import pymc as pm
import numpy
from pymc.examples.DisasterModel import *
s = pm.DiscreteUniform('s', 1851, 1962, value=1900)



@pm.stochastic(dtype=int)
def s(value=1900, t_l=1851, t_h=1962):
    """The switchpoint for the rate of disaster occurrence."""
    if value > t_h or value < t_l:
        # Invalid values
        return -numpy.inf
    else:
        # Uniform log-likelihood
        return -numpy.log(t_h - t_l + 1)



@pm.stochastic(dtype=int)
def s(value=1900, t_l=1851, t_h=1962):
    """The switchpoint for the rate of disaster occurrence."""

    def logp(value, t_l, t_h):
        if value > t_h or value < t_l:
            return -numpy.inf
        else:
            return -numpy.log(t_h - t_l + 1)

    def random(t_l, t_h):
        return numpy.round( (t_l - t_h) * random() ) + t_l

    


def s_logp(value, t_l, t_h):
    if value > t_h or value < t_l:
        return -numpy.inf
    else:
        return -numpy.log(t_h - t_l + 1)

def s_rand(t_l, t_h):
    return numpy.round( (t_l - t_h) * random() ) + t_l

s = pm.Stochastic( logp = s_logp,
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



    


x = pm.Binomial('x', value=7, n=10, p=.8, observed=True)





x = pm.MvNormalCov('x',numpy.ones(3),numpy.eye(3))
y = pm.MvNormalCov('y',numpy.ones(3),numpy.eye(3))
print x+y
#<pymc.PyMCObjects.Deterministic '(x_add_y)' at 0x105c3bd10>

print x[0]
#<pymc.CommonDeterministics.Index 'x[0]' at 0x105c52390>

print x[1]+y[2]
#<pymc.PyMCObjects.Deterministic '(x[1]_add_y[2])' at 0x105c52410>



@pm.deterministic
def r(switchpoint = s, early_rate = e, late_rate = l):
    """The rate of disaster occurrence."""
    value = numpy.zeros(len(D))
    value[:switchpoint] = early_rate
    value[switchpoint:] = late_rate
    return value



def r_eval(switchpoint = s, early_rate = e, late_rate = l):
    value = numpy.zeros(len(D))
    value[:switchpoint] = early_rate
    value[switchpoint:] = late_rate
    return value

r = pm.Deterministic(  eval = r_eval,
                    name = 'r',
                    parents = {'switchpoint': s, 'early_rate': e, 'late_rate': l},
                    doc = 'The rate of disaster occurrence.',
                    trace = True,
                    verbose = 0,
                    dtype=float,
                    plot=False,
                    cache_depth = 2)



N = 10
x_0 = pm.Normal('x_0', mu=0, tau=1)

# Initialize array of stochastics
x = numpy.empty(N,dtype=object)
x[0] = x_0

# Loop over number of elements in N
for i in range(1,N):

    # Create Normal stochastic, whose mean is the previous element in x
    x[i] = pm.Normal('x_%i' % i, mu=x[i-1], tau=1)



@pm.observed
@pm.stochastic
def y(value = 1, mu = x, tau = 100):
    return pm.normal_like(value, numpy.sum(mu**2), tau)



# i is not specified directly in the text, since it is a general explanation. 
i=1

@pm.potential
def psi_i(x_lo = x[i], x_hi = x[i+1]):
    """A pair potential"""
    return -(x_lo - x_hi)**2



def psi_i_logp(x_lo = x[i], x_hi = x[i+1]):
    return -(x_lo - x_hi)**2

psi_i = pm.Potential(  logp = psi_i_logp,
                    name = 'psi_i',
                    parents = {'x_lo': x[i], 'x_hi': x[i+1]},
                    doc = 'A pair potential',
                    verbose = 0,
                    cache_depth = 2)



# Just made this up to test the bit of code below
def fun(value, a=1):
    return 2*a+ value

arguments = pm.DictContainer(dict(value=5, a=1))

# Here is the code from the paper. 
L = pm.LazyFunction(fun, arguments)

fun(**arguments.value)

