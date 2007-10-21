import sys, inspect
from imp import load_dynamic
from PyMCObjects import Stochastic, Deterministic, DiscreteStochastic, BinaryStochastic, Potential
from Node import ZeroProbability, ContainerBase, Node
from Container import Container
import numpy as np

def _extract(__func__, kwds, keys, classname): 
    """
    Used by decorators stoch and dtrm to inspect declarations
    """
    
    # Add docs and name
    kwds.update({'doc':__func__.__doc__, 'name':__func__.__name__})
    
    # Instanitate dictionary of parents
    parents = {}
    
    # Define global tracing function (I assume this is for debugging??)
    # No, it's to get out the logp and random functions, if they're in there.
    def probeFunc(frame, event, arg):
        if event == 'return':
            locals = frame.f_locals
            kwds.update(dict((k,locals.get(k)) for k in keys))
            sys.settrace(None)
        return probeFunc

    sys.settrace(probeFunc)
    
    # Get the functions logp and random (complete interface).
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
    
    if defaults is None:
        defaults = ()
    
    # Make sure all parents were defined
    arg_deficit = (len(args) - ('value' in args)) - len(defaults)
    if arg_deficit > 0:
        err_str =  classname + ' ' + __func__.__name__ + ': no parent provided for the following labels:'
        for i in range(arg_deficit):
            err_str +=  " " + args[i + ('value' in args)]
            if i < arg_deficit-1:
                err_str += ','
        raise ValueError, err_str
    
    # Fill in parent dictionary
    try:
        parents.update(dict(zip(args[-len(defaults):], defaults)))    
    except TypeError: 
        pass
        
    if parents.has_key('value'):
        value = parents.pop('value')
    else:
        value = None
                    
    return (value, parents)

def stoch(__func__=None, __class__=Stochastic, binary=False, discrete=False, **kwds):
    """
    Decorator function for instantiating stochastic variables. Usages:
    
    Medium:
    
        @stoch
        def A(value = ., parent_name = .,  ...):
            return foo(value, parent_name, ...)
        
        @stoch(trace=trace_object)
        def A(value = ., parent_name = .,  ...):
            return foo(value, parent_name, ...)
            
    Long:

        @stoch
        def A(value = ., parent_name = .,  ...):
            
            def logp(value, parent_name, ...):
                return foo(value, parent_name, ...)
                
            def random(parent_name, ...):
                return bar(parent_name, ...)
                
    
        @stoch(trace=trace_object)
        def A(value = ., parent_name = .,  ...):
            
            def logp(value, parent_name, ...):
                return foo(value, parent_name, ...)
                
            def random(parent_name, ...):
                return bar(parent_name, ...)
                
    where foo() computes the log-probability of the variable A
    conditional on its value and its parents' values, and bar()
    generates a random value from A's distribution conditional on
    its parents' values.
    
    :SeeAlso: Stochastic, Deterministic, dtrm, data, Potential, potential, Model, Container
    """
    
    if binary:
        __class__ = BinaryStochastic
    elif discrete:
        __class__ = DiscreteStochastic
    
    def instantiate_p(__func__):
        value, parents = _extract(__func__, kwds, keys, 'Stochastic')
        return __class__(value=value, parents=parents, **kwds)  
            
    keys = ['logp','random','rseed']
    
    instantiate_p.kwds = kwds
    
    if __func__:
        return instantiate_p(__func__)
        
    return instantiate_p
    
def discrete_stoch(__func__=None, **kwds):
    """
    Instantiates a DiscreteStochastic instance, which takes only
    integer values.
    
    Same usage as stoch.
    """
    return stoch(__func__=__func__, __class__ = DiscreteStochastic, **kwds)
    
def binary_stoch(__func__=None, **kwds):
    """
    Instantiates a BinaryStochastic instance, which takes only boolean
    values.
    
    Same usage as stoch.
    """
    return stoch(__func__=__func__, __class__ = BinaryStochastic, **kwds)


def potential(__func__ = None, **kwds):
    """
    Decorator function instantiating potentials. Usage:

    @potential
    def B(parent_name = ., ...)
        return baz(parent_name, ...)

    where baz returns the dtrm B's value conditional
    on its parents.

    :SeeAlso: Deterministic, dtrm, Stochastic, Potential, stoch, data, Model, Container
    """
    def instantiate_pot(__func__):
        junk, parents = _extract(__func__, kwds, keys, 'Potential')
        return Potential(parents=parents, **kwds)

    keys = ['logp']

    instantiate_pot.kwds = kwds

    if __func__:
        return instantiate_pot(__func__)

    return instantiate_pot


def dtrm(__func__ = None, **kwds):
    """
    Decorator function instantiating deterministic variables. Usage:
    
    @dtrm
    def B(parent_name = ., ...)
        return baz(parent_name, ...)
        
    @dtrm(trace = trace_object)
    def B(parent_name = ., ...)
        return baz(parent_name, ...)        
        
    where baz returns the variable B's value conditional
    on its parents.
    
    :SeeAlso: Deterministic, potential, Stochastic, stoch, data, Model, Container
    """
    def instantiate_n(__func__):
        junk, parents = _extract(__func__, kwds, keys, 'Deterministic')
        return Deterministic(parents=parents, **kwds)
        
    keys = ['eval']
    
    instantiate_n.kwds = kwds
    
    if __func__:
        return instantiate_n(__func__)

    return instantiate_n


def data(obj=None, **kwds):
    """
    Decorator function to instantiate data objects.     
    If given a Stochastic, sets a the isdata flag to True.
    
    Can be used as
    
    @data
    def A(value = ., parent_name = .,  ...):
        return foo(value, parent_name, ...)
    
    or as
    
    @data
    @stoch
    def A(value = ., parent_name = .,  ...):
        return foo(value, parent_name, ...)
        
    
    :SeeAlso: stoch, Stochastic, dtrm, Deterministic, potential, Potential, Model, Container
    """
    if obj is not None:
        if isinstance(obj, Stochastic):
            obj.isdata=True
            return obj
        else:
            p = stoch(__func__=obj, isdata=True, **kwds)
            return p
    
    kwds['isdata']=True
    def instantiate_data(func):
        return stoch(func, **kwds)
        
    return instantiate_data

