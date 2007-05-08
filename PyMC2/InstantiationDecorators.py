import sys, inspect
from imp import load_dynamic
from PyMCObjects import Parameter, Node, DiscreteParameter, BinaryParameter
from utils import extend_children, _push, _extract
from PyMCBase import ZeroProbability
import numpy as np


def parameter(__func__=None, __class__=Parameter, **kwds):
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
    
    :SeeAlso: Parameter, Node, node, data, Model, Container
    """

    def instantiate_p(__func__):
        value, parents = _extract(__func__, kwds, keys)
        return __class__(value=value, parents=parents, **kwds)      
    keys = ['logp','random','rseed']

    if __func__ is None:
        instantiate_p.kwds = kwds   
        return instantiate_p
    else:
        instantiate_p.kwds = kwds
        return instantiate_p(__func__)

    return instantiate_p
    
def discrete_parameter(__func__=None, **kwds):
    """
    Instantiates a DiscreteParameter instance, which takes only
    integer values.
    
    Same usage as parameter.
    """
    return parameter(__func__=__func__, __class__ = DiscreteParameter, **kwds)
    
def binary_parameter(__func__=None, **kwds):
    """
    Instantiates a BinaryParameter instance, which takes only boolean
    values.
    
    Same usage as parameter.
    """
    return parameter(__func__=__func__, __class__ = BinaryParameter, **kwds)


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
    
    :SeeAlso: Node, Parameter, parameter, data, Model, Container
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


def data(*args, **kwds):
    """
    Decorator function to instantiate data objects.     
    If given a Parameter, sets a the isdata flag to True.
    
    :SeeAlso: parameter, Parameter, node, Node, Model, Container
    """
    if isinstance(args[0], Parameter):
        p = args[0]
    else:
        p = parameter(*args, **kwds)
    p.isdata = True
    return p
    


