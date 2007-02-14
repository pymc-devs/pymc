import sys, inspect
from imp import load_dynamic
import distributions
from PyMC2 import Parameter, Node
from copy import copy
from AbstractBase import *
from utils import extend_children, _push, _extract, LikelihoodError
import numpy as np
import types

def parameter(__func__=None, **kwds):
    """
    Decorator function for instantiating parameters. Usages:
    
    Medium:
    
        @parameter
        def A(value = ., parent_name = .,  ...):
            return foo(value, parent_name, ...)
        
        @parameter(trace=trace_object)
        def A(value = ., parent_name = .,  ...):
            return foo(value, parent_name, ...)
            
    Long:

        @parameter
        def A(value = ., parent_name = .,  ...):
            
            def logp(value, parent_name, ...):
                return foo(value, parent_name, ...)
                
            def random(parent_name, ...):
                return bar(parent_name, ...)
                
    
        @parameter(trace=trace_object)
        def A(value = ., parent_name = .,  ...):
            
            def logp(value, parent_name, ...):
                return foo(value, parent_name, ...)
                
            def random(parent_name, ...):
                return bar(parent_name, ...)
                
    where foo() computes the log-probability of the parameter A
    conditional on its value and its parents' values, and bar()
    generates a random value from A's distribution conditional on
    its parents' values.
    """

    def instantiate_p(__func__):
        value, parents = _extract(__func__, kwds, keys)
        return Parameter(value=value, parents=parents, **kwds)      
    keys = ['logp','random','rseed']

    if __func__ is None:
        instantiate_p.kwds = kwds   
        return instantiate_p
    else:
        instantiate_p.kwds = kwds
        return instantiate_p(__func__)

    return instantiate_p


def node(__func__ = None, **kwds):
    """
    Decorator function instantiating nodes. Usage:
    
    @node
    def B(parent_name = ., ...)
        return baz(parent_name, ...)
        
    @node(trace = trace_object)
    def B(parent_name = ., ...)
        return baz(parent_name, ...)        
        
    where baz returns the node B's value conditional
    on its parents.
    """

    def instantiate_n(__func__):
        junk, parents = _extract(__func__, kwds, keys)
        return Node(parents=parents, **kwds)        
    keys = ['eval']
    
    if __func__ is None:
        instantiate_n.kwds = kwds   
        return instantiate_n
    else:
        instantiate_n.kwds = kwds
        return instantiate_n(__func__)

    return instantiate_n


def data(__func__=None, **kwds):
    """
    Decorator instantiating data objects. Usage is just like
    parameter.
    """
    return parameter(__func__, isdata=True, trace = False, **kwds)
    

def create_distribution_instantiator(name, logp=None, random=None, module=distributions):
    """Return a function to instantiate a parameter from a particular distribution.
     
      :Example:
        >>> Exponential = create_distribution_instantiator('exponential')
        >>> A = Exponential('A', value=2.3, beta=4)
    """
    
    if type(module) is types.ModuleType:
        module = copy(module.__dict__)
    elif type(module) is dict:
        module = copy(module)
    else:
        raise AttributeError
        
    if logp is None:
        try:
           logp = module[name+"_like"]
        except:
            raise "No likelihood found with this name ", name+"_like"
    if random is None:
        try: 
            random = module['r'+name]
        except:
            raise "No random generator found with this name ", 'r'+name        
    
    # Build parents dictionary by parsing the __func__tion's arguments.
    (args, varargs, varkw, defaults) = inspect.getargspec(logp)
    parent_names = args[1:]
    try:
        parents_default = dict(zip(args[-len(defaults):], defaults))
    except TypeError: # No parents at all.   
        parents_default = {}
        
        
    def instantiator(name, value=None, trace=True, rseed=False, **kwds):
        """%s(name, value, trace=True, rseed=False, **kwds):
        
        Instantiate a Parameter instance with a %s prior.
        """%(name.capitalize(), name)
        # Deal with keywords
        # Find which are parents
        parents=parents_default
            
        for k in kwds.keys():
            if k in parent_names:
                parents[k] = kwds.pop(k)
        
        if value is None:
            print 'No initial value specified. rseed set to True. Is this allowed ?'
            rseed = True
            value = random(**parents)

        return Parameter(value=value, name=name, parents=parents, logp=logp, random=random, \
        trace=trace, rseed=rseed, isdata=False, children=set())

    #instantiator.__doc__="Instantiate a Parameter instance with a %s prior."%name
    return instantiator

def fortranlike(f, snapshot, mv=False):
    """
    Decorator function for fortran likelihoods
    ==========================================
    
    Wrap function f(*args, **kwds) where f is a likelihood defined in flib.
    
    Assume args = (x, param1, param2, ...)
    Before passing the arguments to the function, the wrapper makes sure that 
    the parameters have the same shape as x.

    mv: multivariate (True/False)
    
    Add compatibility with GoF (Goodness of Fit) tests 
    --------------------------------------------------
    * Add a 'prior' keyword (True/False)
    * If the keyword gof is given and is True, return the GoF (Goodness of Fit)
    points instead of the likelihood. 
    * A 'loss' keyword can be given, to specify the loss function used in the 
    computation of the GoF points. 
    * If the keyword random is given and True, return a random variate instead
    of the likelihood.
    """
    name = f.__name__[:-5]
    # Take a snapshot of the main namespace.
    
    
    # Find the functions needed to compute the gof points.
    expval_func = snapshot[name+'_expval']
    random_func = snapshot['r'+name]
    
    def wrapper(*args, **kwds):
        """This wraps a likelihood."""
        
        # Shape manipulations
        if not mv:
            xshape = np.shape(args[0])
            newargs = [np.asarray(args[0])]
            for arg in args[1:]:
                newargs.append(np.resize(arg, xshape))
            for key in kwds.iterkeys():
                kwds[key] = kwds[key]
        else:
            """x, mu, Tau
            x: (kxN)
            mu: (kxN) or (kx1)
            Tau: (k,k)
            """
            xshape=np.shape(args[0])
            newargs = [np.asarray(args[0])] 
            newargs.append(np.resize(args[1], xshape))
            newargs.append(np.asarray(args[2]))
            
        if kwds.pop('gof', False) and not kwds.pop('prior', False):
            """Return gof points."""            
            loss = kwds.pop('gof', squared_loss)
            #name = kwds.pop('name', name)
            expval = expval_func(*newargs[1:], **kwds)
            y = random_func(*newargs[1:], **kwds)
            gof_points = GOFpoints(newargs[0],y,expval,loss)
            return gof_points
        elif kwds.pop('random', False):
            return random_func(*newargs[1:], **kwds)
        else:
            """Return likelihood."""
            try:
                return f(*newargs, **kwds)
            except LikelihoodError:
                return -np.Inf
        

    # Assign function attributes to wrapper.
    wrapper.__doc__ = f.__doc__
    wrapper._PyMC = True
    wrapper.__name__ = f.__name__
    wrapper.name = name
    return wrapper
