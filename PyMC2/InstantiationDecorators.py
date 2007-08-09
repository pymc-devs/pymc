import sys, inspect
from imp import load_dynamic
from PyMCObjects import Parameter, Node, DiscreteParameter, BinaryParameter, Potential
from utils import extend_children, _push
from PyMCBase import ZeroProbability
import numpy as np

def _extract(__func__, kwds, keys): 
    """
    Used by decorators parameter and node to inspect declarations
    """
    
    # Add docs and name
    kwds.update({'doc':__func__.__doc__, 'name':__func__.__name__})
    
    # Instanitate dictionary of parents
    parents = {}
    
    # Define global tracing function (I assume this is for debugging??)
    def probeFunc(frame, event, arg):
        if event == 'return':
            locals = frame.f_locals
            kwds.update(dict((k,locals.get(k)) for k in keys))
            sys.settrace(None)
        return probeFunc

    sys.settrace(probeFunc)
    
    # Get the __func__tions logp and random (complete interface).
    try:
        __func__()
    except:
        if 'logp' in keys:  
            kwds['logp']=__func__
        else:
            kwds['eval'] =__func__

    for key in keys:
        if not kwds.has_key(key):
            kwds[key] = None            
            
    for key in ['logp', 'eval']:
        if key in keys:
            if kwds[key] is None:
                kwds[key] = __func__

    # Build parents dictionary by parsing the __func__tion's arguments.
    (args, varargs, varkw, defaults) = inspect.getargspec(__func__)
    try:
        parents.update(dict(zip(args[-len(defaults):], defaults)))

    # No parents at all     
    except TypeError: 
        pass
        
    if parents.has_key('value'):
        value = parents.pop('value')
    else:
        value = None
        
    return (value, parents)

def parameter(__func__=None, __class__=Parameter, binary=False, discrete=False, **kwds):
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
    
    if binary:
        __class__ = BinaryParameter
    elif discrete:
        __class__ = DiscreteParameter
    
    def instantiate_p(__func__):
        value, parents = _extract(__func__, kwds, keys)
        return __class__(value=value, parents=parents, **kwds)  
            
    keys = ['logp','random','rseed']
    
    instantiate_p.kwds = kwds
    
    if __func__:
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


def potential(__func__ = None, **kwds):
    """
    Decorator function instantiating potentials. Usage:

    @potential
    def B(parent_name = ., ...)
        return baz(parent_name, ...)

    where baz returns the node B's value conditional
    on its parents.

    :SeeAlso: Node, Parameter, Potential, parameter, data, Model, Container
    """
    def instantiate_pot(__func__):
        junk, parents = _extract(__func__, kwds, keys)
        return Potential(parents=parents, **kwds)

    keys = ['logp']

    instantiate_pot.kwds = kwds

    if __func__:
        return instantiate_pot(__func__)

    return instantiate_pot


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
    
    instantiate_n.kwds = kwds
    
    if __func__:
        return instantiate_n(__func__)

    return instantiate_n

# @data(discrete=True): obj-> None,   kwds-> {'discrete':True}
# data(parameter)     : obj-> <PyMC2 Parameter>, kwds->{}
# @data
# def ...             : obj-> <function>,  kwds-> {}


def data(obj=None, **kwds):
    """
    Decorator function to instantiate data objects.     
    If given a Parameter, sets a the isdata flag to True.
    
    Can be used as
    
    @data
    def A(value = ., parent_name = .,  ...):
        return foo(value, parent_name, ...)
    
    or as
    
    @data
    @parameter
    def A(value = ., parent_name = .,  ...):
        return foo(value, parent_name, ...)
        
    
    :SeeAlso: parameter, Parameter, node, Node, Model, Container
    """
    if obj is not None:
        if isinstance(obj, Parameter):
            obj.isdata=True
            return obj
        else:
            p = parameter(__func__=obj, isdata=True, **kwds)
            return p
    
    kwds['isdata']=True
    def instantiate_data(func):
        return parameter(func, **kwds)
        
    return instantiate_data

