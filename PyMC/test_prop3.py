# Test proposition 3
# Based on the Disaster Sampler

DEBUG = True
from proposition3 import Data, Parameter, Node, BuildModel, JointSampling, Sampler,\
SamplingMethod, LikelihoodError
#from proposition3 import *
#import test_decorator
from test_decorator import normal_like, uniform_like, poisson_like, \
uniform_prior, rnormal

# Define the data and parameters
@Data(value = (4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
    3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
    2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
    1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
    0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
    3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
    0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1))
def disasters():
    """Annual occurences of coal mining disasters."""
    return 0 

@Parameter(init_val=50, discrete=True)
def switchpoint(self, disasters):
    return uniform_like(self, 1, len(disasters)-2)

@Parameter(init_val=1.)
def early_mean(self, disasters, switchpoint):
    """Rate parameter of poisson distribution."""
    return poisson_like(disasters[:switchpoint], self)

@Parameter(init_val=1.)
def late_mean(self, disasters, switchpoint):
    """Rate parameter of poisson distribution."""
    return poisson_like(disasters[switchpoint:], self)

# Create the model
model = BuildModel(early_mean, late_mean)
print 'Parents: ', model.parents
print 'Children: ', model.children
print 'Extended children: ', model.ext_children
print 'Current values: ', model.get_value()
print 'Current likelihoods: ', model.likelihood()
model.switchpoint = 60
print model.switchpoint
print model.late_mean.like()

print '\nNow a couple of Metropolis steps.\n'
s = SamplingMethod(model, 'early_mean', debug=True)
for i in range(3):
    s.step()
#JS = JointSampling(model, ['early_mean', 'late_mean'])
#S = Sampler(JS)
#S.sample()


# Second example ------------------------------------------------------------------


### Define model parameters
##@Parameter(init_val = 4)
##def alpha(self):
##    """Parameter alpha of toy model."""
##    # The return value is the prior. 
##    return uniform_like(self, 0, 10)
##
##@Parameter(init_val=5)
##def beta(self, alpha):
##    """Parameter beta of toy model."""
##    return normal_like(self, alpha, 2)
##
##
### Define the data
##@Data(value = [1,2,3,4])
##def input():
##    """Measured input driving toy model."""
##    like = 0
##    return like
##    
##@Data(value = [45,34,34,65])
##def exp_output():
##    """Experimental output."""
##    # likelihood a value or a function
##    return 0
##    
### Model function
### No decorator is needed, its just a function.
##def model(alpha, beta, input):
##    """Return the simulated output.
##    Usage: sim_output(alpha, beta, input)
##    """
##    self = alpha + beta * input
##    return self
##    
##
### The likelihood node. 
### The keyword specifies the function to call to get the node's value.
##@Node(self=model, shape=input.shape)
##def sim_output(self, exp_output):
##    """Return likelihood of simulation given the experimental data."""
##    return normal_like(self, exp_output, 2)
##
### Just a function we want to compute along the way.
##def residuals(sim_output, exp_output):
##    """Model's residuals"""
##    return sim_output-exp_output
##
### Put everything together
##bunch = Merge(sim_output, residuals)
##print 'alpha: ', bunch.alpha
###print 'alpha_like: ', bunch.alpha_like
##print 'alpha.like(): ', bunch.alpha.like()
##print 'beta: ', bunch.beta
###print 'beta_like: ', bunch.beta_like
##print 'beta.like(): ', bunch.beta.like()
##print 'exp_output: ', bunch.exp_output
##print 'sim_output: ', bunch.sim_output
###print 'sim_output_like: ', bunch.sim_output_like
##print 'sim_output.like(): ', bunch.sim_output.like()
##print 'residuals: ', bunch.residuals


#S = SamplingMethod(bunch, 'alpha')


#print bunch.parent_dic
#print bunch.call_args
# The last step would be to call 
# Sampler(posterior, 'Metropolis')
# i.e. sample the parameters from posterior using a Metropolis algorithm.
# Sampler recursively looks at the parents of posterior, namely sim_output and
# exp_output, identifies the Parameters and sample over them according to the
# posterior likelihood. 
# Since the parents are known for each element, we can find the children of the 
# Parameters, so when one Parameter is sampled, we only need to compute the 
# likelihood of its children, and avoid computing elements that are not modified 
# by the current Parameter. 
