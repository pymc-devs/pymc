# Proposition 3

# Here is the interface I'm proposing for PyMC. The general idea is to create 
# objects that return their likelihood. The dependencies are stored in an 
# attribute named 'parents'. Once the user has defined his parameters, data and 
# nodes, he instantiates a sampler class using those elements. 
#
# See example below.

import numpy as np
from inspect import getargs

# Decorator to define decorators with arguments... recursion baby !
decorator_with_args = lambda decorator: lambda *args, **kwargs: lambda func: decorator(func, *args, **kwargs)
    

@decorator_with_args
def Parameter(func, *args, **kwds):
    """Decorator function for PyMC parameter.
    
    Input
        func: function returning the likelihood of the parameter, ie its prior. 
        init_val: Initial value to start the sampling.
    
    Example
        @parameter(init_val=56)
        def alpha(self):
            "Parameter alpha of model M."
            return uniform_like(self, 0, 10)
    """
    if len(kwds) == 0:
        kwds['init_val'] = args[0]
    func.__dict__.update(kwds)
    func.parents = getargs(func.func_code)[0]
    func.parents.remove('self')
    return func
    
    
@decorator_with_args
def Data(func, *args,  **kwds):
    """Decorator function for PyMC data.
    
    Input
        'value': The data.
        func: A function returning the likelihood of the data.
        
    Example
        @Data(value=[1,2,3,4])
        def input():
            "Input data to force model."
            return 0
    """
    if len(kwds) == 0:
        kwds['value'] = np.asarray(args[0])
    func.__dict__.update(kwds)
    func.value = np.asarray(func.value)
    func.parents = getargs(func.func_code)[0]
    try:
        func.parents.remove('self')        
    except:pass
    
    return func
    
def Node(func):
    """Decorator function for PyMC node.

    Input
        func: A function returning the likelihood of the node.
        
    Example
        @Node
        def posterior(sim_output, exp_output, var):
            return normal_like(sim_output, exp_output, 1./var)
    
    Note
        All arguments to the likelihood of a Node must be somewhere in the 
        namespace, in order for Node to find the parents (dependencies).
        In the example, sim_output must be a function, and var a constant or a 
        Parameter. 
    
    TODO: Implement higher level of recursion in the parent finding. 
    """
    
    parents = getargs(func.func_code)[0]
    func.parents = {}
    for p in parents:
        try:
            func.parents[p] = globals()[p].parents
        except AttributeError:
            func.parents[p] = getargs(globals()[p].func_code)[0]
    return func

# Example ------------------------------------------------------------------
from test_decorator import normal_like, uniform_like

# Define model parameters
@Parameter(init_val = 4)
def alpha(self):
    """Parameter alpha of toy model."""
    # The return value is the prior. 
    return uniform_like(self, 0, 10)

@Parameter(init_val=5)
def beta(self, alpha):
    """Parameter beta of toy model."""
    return normal_like(self, alpha, 2)


# Define the data
@Data(value = [1,2,3,4])
def input():
    """Measured input driving toy model."""
    like = 0
    return like
    
@Data(value = [45,34,34,65])
def exp_output():
    """Experimental output."""
    # likelihood a value or a function
    return 0
    
# Model function
# No decorator is needed, its just a function.
def sim_output(alpha, beta, input):
    """Return the simulated output.
    Usage: sim_output(alpha, beta, input)
    """
    self = alpha + beta * input
    return like
    

# Finally, the posterior node that combines everything. 
@Node
def posterior(sim_output, exp_output):
    """Return likelihood of simulation given the experimental data."""
    return normal_like(sim_output, exp_output, 2)


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
