# Proposition for an interface to PyMC.
# David Huard
# November 19, 2006

# The main idea is to separate the problem specification of the MCMC 
# implementation.

# Problem specs

@Parameter
def alpha(self):
    """Parameter alpha of toy model."""
    # The return value is the prior. 
    return uniform_like(self, 0, 10)

@Parameter
def beta(self, alpha):
    """Parameter beta of toy model."""
    return normal_like(self, alpha, 2)

@Data
def input(self):
    """Measured input driving toy model."""
    # input value
    value = [1,2,3,4]
    # input likelihood     
    like = 0
    return locals()
    

@Data
def exp_output(self):
    """Experimental output."""
    # output value
    value = [45,34,34,65]
    # likelihood a value or a function
    like = 0
    return locals()

@Node
def sim_output(self, alpha, beta, input, exp_output):
    """Compute the simulated output and return its likelihood.
    Usage: sim_output(alpha, beta, input, exp_output)
    """
    self = toy_model(alpha, beta, input)
    return normal_like(self, exp_output, 2)

"""
Ok, that's it for the problem specification, now we want to sample alpha and 
beta using MCMC.
"""

##from PyMC import Sampler
##
### alpha and beta will be sampled jointly since we think they are strongly 
### correlated
##J = Joint(alpha, beta)
### Initial values
##J.alpha = 3   
##J.beta = 5
##J.alpha.scale = 2
##
### We create a sampler instance that will sample the parent parameters of output, 
### using the Joint subsampler for alpha and beta. 
##S = Sampler(output, J)
##
### If instead we use simply 
##S1 = Sampler(output)
### Initial values
##S1.alpha = 3
##S1.beta = 5
### The sampler will look at the parents of output (alpha, beta, input, exp_output), 
### and sample the parents defined as parameters. 
##S.sample()
###------------------------------------------------------------------
##
##
### For a model selection problem, we'd do something like
##@Data
##def input(self):
##    input = [...]
##    return 0
##
##@Parameter
##def beta(self):
##    """Common parameter for all models."""
##    return uniform_like(self,0,1)
##
##@Model
##def MODEL1(beta, input):
##    
##    @Parameters
##    def alpha(self):
##        """Parameter specific to Model 1."""
##        return uniform_like(self, 0, 10)
##
##    return model1_like(alpha, beta, input)
##
##@Model
##def MODEL2(beta, input):
##@Parameters
##    def alpha(self):
##        """Parameter specific to Model 2."""
##        return gamma_like(self, 3, 4)
##
##    return model2_like(alpha, beta, input)
##
### ...
##
##MS = Sampler([MODEL1, MODEL2,...])
##MS.sample()

"""
I think this type of interface offers the modularity and readability we were   
looking for. I'm not sure however how to code it... but I'm pretty sure Python 
is powerful enough to do it.
Keeping the problem specs and  the sampling details separate will help write robust code, 
without risk of introducing bugs each time you want to try out a new sampling 
scheme.  

The sampling part is a little bit dull, I'll dig for better ideas. 

Children are determined by the local variables of each @Node and @Parameter. 
That is,  when we call Sampler(node1), the __init__ method looks at the definition of 
node1 and defines the inputs of this node as its parents. For each Parameter, 
a dictionary of children is built.  

The sampler __init__ method looks for SubSamplers that would define sampling
methods for the parameters. If it doesn't find any, a standard subsampler is 
applied, so that each parameter becomes a subsampler instance. The get_likelihood
method of each SubSampler instance is the sum of the __call__() of its children. 
If I'm not fooling myself, this avoids the need for time stamps. 

The Sampler then only has to call the SubSamplers sequentially. This would allow
a very fine grained control of the behavior of each subsampler. You could tell 
one subsampler to sample for 20000 iterations, then set itself to its maximum 
likelihood value for the rest of the simulation. 

This is a draft I wrote while I'm at home, sick and feverish, so excuse me
if its non-sense. 
"""
    

