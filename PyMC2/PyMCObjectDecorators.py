import sys, inspect
from imp import load_dynamic
from PyMC2 import Parameter, Node
from AbstractBase import *
from utils import extend_children, _push, _extract, LikelihoodError
import numpy as np


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
    


