"""
Straight line fit -- linear regression
======================================

This is probably the most widespread problems in statistics: estimating the
slope and ordinate of a linear relationship y = ax+b given some data (x,y).

The standard least-square (SLS) solution to this problem assumes that the input
data x is exact, and that errors only affect output measurements y. In many
instances, this assumption does not hold and using the same SLS method yields
biased parameters: the slope is underestimated and the ordinate overestimated.

Here, both input and output data are assumed to be corrupted by errors of zero
mean and variances of sigma_x and sigma_y respectively. Under these assumptions,
the most general statistical distribution (maximum entropy) is the normal
distribution. In the following, the parameter distribution is sampled by
marginalizing x from

.. math::
    p(a,b,x \mid \tilde{x}, \tilde{y}) = p(\tilde{y} \mid a, b, x) p(\tilde{x} \mid x) p(x) p(a,b),

where p(x) stands for the prior for the true input and p(a,b) the prior for the
regression parameters.
"""
from pymc import stochastic, observed, deterministic, uniform_like, runiform, rnormal, Sampler, Normal, Uniform
from numpy import inf, log, cos,array
import pylab

# ------------------------------------------------------------------------------
# Synthetic values
# Replace by real data
# ------------------------------------------------------------------------------
slope = 1.5
intercept = 4
N = 30
true_x = runiform(0,50, N)
true_y = slope*true_x + intercept
data_y = rnormal(true_y, 2)
data_x = rnormal(true_x, 2)



# ------------------------------------------------------------------------------
# Calibration of straight line parameters from data
# ------------------------------------------------------------------------------


@stochastic
def theta(value=array([2.,5.])):
    """Slope and intercept parameters for a straight line.
    The likelihood corresponds to the prior probability of the parameters."""
    slope, intercept = value
    prob_intercept = uniform_like(intercept, -10, 10)
    prob_slope = log(1./cos(slope)**2)
    return prob_intercept+prob_slope

init_x = data_x.clip(min=0, max=50)

# Inferred true inputs.
x = Uniform('x', lower=0, upper=50, value=init_x)

@deterministic
def modelled_y(x=x, theta=theta):
    """Return y computed from the straight line model, given the
    inferred true inputs and the model paramters."""
    slope, intercept = theta
    return slope*x + intercept


"""
Input error model.

    Define the probability of measuring x knowing the true value. 
"""
measured_input = Normal('measured_input', mu=x, tau=2, value=data_x, observed=True)

"""
Output error model.
    Define the probability of measuring x knowing the true value.
    In this case, the true value is assumed to be given by the model, but
    structural errors could be integrated to the analysis as well.
"""
y = Normal('y', mu=modelled_y, tau=2, value=data_y, observed=True)
