import sys, inspect
from imp import load_dynamic
import distributions
from PyMC2 import Parameter, Node
from copy import copy
from AbstractBase import *
from utils import extend_children, _push, _extract

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
        kwds['children'] = set()
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
        kwds['children'] = set()
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
    
def create_distribution_instantiator(name, logp=None, random=None):
    """Return a function to instantiate a parameter from a particular distribution.
     
      :Example:
        >>> Exponential = create_distribution_instantiator('exponential')
        >>> A = Exponential(name ='A', beta=4)
    """
    

    if logp is None:
        try:
           logp = getattr(distributions, name+"_like")
        except:
            raise "No likelihood found with this name ", name+"_like"
    if random is None:
        try: 
            random = getattr(distributions, 'r'+name)
        except:
            raise "No random generator found with this name ", 'r'+name
        
    
    # Build parents dictionary by parsing the __func__tion's arguments.
    (args, varargs, varkw, defaults) = inspect.getargspec(logp)
    parent_names = args[1:]
    try:
        parents_default = dict(zip(args[-len(defaults):], defaults))
    except TypeError: # No parents at all.   
        parents_default = {}
        
        
    def instantiator(name, trace=True, rseed=False, **kwds):
        # Deal with keywords
        # Find which are parents
        value = kwds.pop('value')
        parents=parents_default
        for k in kwds.keys():
            if k in parent_names:
                parents[k] = kwds.pop(k)
        return Parameter(value=value, name=name, parents=parents, logp=logp, random=random, \
        trace=trace, rseed=rseed, isdata=False, children=set())

    instantiator.__doc__="Instantiate a Parameter instance with a %s prior."%name
    return instantiator
