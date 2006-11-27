# Proposition 3
# This is proposition 2, but working...

import numpy as np
from inspect import getargs

# Decorator to define decorator with args... recursion baby !
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
    """
    
    parents = getargs(func.func_code)[0]
    func.parents = {}
    for p in parents:
        try:
            func.parents[p] = globals()[p].parents
        except AttributeError:
            func.parents[p] = getargs(globals()[p].func_code)[0]
    return func

# Testing
from test_decorator import normal_like, uniform_like

@parameter(init_val = 4)
def alpha(self):
    """Parameter alpha of toy model."""
    # The return value is the prior. 
    return uniform_like(self, 0, 10)

@parameter(init_val=5)
def beta(self, alpha):
    """Parameter beta of toy model."""
    return normal_like(self, alpha, 2)

@data(value = [1,2,3,4])
def input():
    """Measured input driving toy model."""
    like = 0
    return like
    
@data(value = [45,34,34,65])
def exp_output():
    """Experimental output."""
    # likelihood a value or a function
    return 0
    
def sim_output(alpha, beta, input):
    """Return the simulated output.
    Usage: sim_output(alpha, beta, input)
    """
    self = alpha + beta * input
    return like
    
@Node
def posterior(sim_output, exp_output):
    """Return likelihood of simulation given the experimental data."""
    return normal_like(sim_output, exp_output, 2)

