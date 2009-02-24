"""
The decorators stochastic, deterministic, discrete_stochastic, binary_stochastic, potential and data
are defined here, but the actual objects are defined in PyMCObjects.py
"""

__all__ = ['stochastic', 'stoch', 'deterministic', 'dtrm', 'potential', 'pot', 'data', 'observed']

import sys, inspect, pdb
from imp import load_dynamic
from PyMCObjects import Stochastic, Deterministic, Potential
from Node import ZeroProbability, ContainerBase, Node
from Container import Container
import numpy as np

def _extract(__func__, kwds, keys, classname):
    """
    Used by decorators stochastic and deterministic to inspect declarations
    """

    # Add docs and name
    kwds['doc'] = __func__.__doc__
    if not kwds.has_key('name'):
        kwds['name'] = __func__.__name__
    # kwds.update({'doc':__func__.__doc__, 'name':__func__.__name__})

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

def stochastic(__func__=None, __class__=Stochastic, binary=False, discrete=False, **kwds):
    """
    Decorator function for instantiating stochastic variables. Usages:

    Medium:

        @stochastic
        def A(value = ., parent_name = .,  ...):
            return foo(value, parent_name, ...)

        @stochastic(trace=trace_object)
        def A(value = ., parent_name = .,  ...):
            return foo(value, parent_name, ...)

    Long:

        @stochastic
        def A(value = ., parent_name = .,  ...):

            def logp(value, parent_name, ...):
                return foo(value, parent_name, ...)

            def random(parent_name, ...):
                return bar(parent_name, ...)


        @stochastic(trace=trace_object)
        def A(value = ., parent_name = .,  ...):

            def logp(value, parent_name, ...):
                return foo(value, parent_name, ...)

            def random(parent_name, ...):
                return bar(parent_name, ...)

    where foo() computes the log-probability of the variable A
    conditional on its value and its parents' values, and bar()
    generates a random value from A's distribution conditional on
    its parents' values.

    :SeeAlso:
      Stochastic, Deterministic, deterministic, data, Potential, potential, Model,
      distributions
    """

    def instantiate_p(__func__):
        value, parents = _extract(__func__, kwds, keys, 'Stochastic')
        return __class__(value=value, parents=parents, **kwds)

    keys = ['logp','random','rseed']

    instantiate_p.kwds = kwds

    if __func__:
        return instantiate_p(__func__)

    return instantiate_p

# Shortcut alias
stoch = stochastic

def potential(__func__ = None, **kwds):
    """
    Decorator function instantiating potentials. Usage:

    @potential
    def B(parent_name = ., ...)
        return baz(parent_name, ...)

    where baz returns the deterministic B's value conditional
    on its parents.

    :SeeAlso:
      Deterministic, deterministic, Stochastic, Potential, stochastic, data, Model
    """
    def instantiate_pot(__func__):
        junk, parents = _extract(__func__, kwds, keys, 'Potential')
        return Potential(parents=parents, **kwds)

    keys = ['logp']

    instantiate_pot.kwds = kwds

    if __func__:
        return instantiate_pot(__func__)

    return instantiate_pot
pot = potential

def deterministic(__func__ = None, **kwds):
    """
    Decorator function instantiating deterministic variables. Usage:

    @deterministic
    def B(parent_name = ., ...)
        return baz(parent_name, ...)

    @deterministic(trace = trace_object)
    def B(parent_name = ., ...)
        return baz(parent_name, ...)

    where baz returns the variable B's value conditional
    on its parents.

    :SeeAlso:
      Deterministic, Potential, potential, Stochastic, stochastic, data, Model,
      CommonDeterministics
    """
    def instantiate_n(__func__):
        junk, parents = _extract(__func__, kwds, keys, 'Deterministic')
        return Deterministic(parents=parents, **kwds)

    keys = ['eval']

    instantiate_n.kwds = kwds

    if __func__:
        return instantiate_n(__func__)

    return instantiate_n

# Shortcut alias
dtrm = deterministic

def observed(obj=None, **kwds):
    """
    Decorator function to instantiate data objects.
    If given a Stochastic, sets a the observed flag to True.

    Can be used as

    @observed
    def A(value = ., parent_name = .,  ...):
        return foo(value, parent_name, ...)

    or as

    @stochastic(observed=True)
    def A(value = ., parent_name = .,  ...):
        return foo(value, parent_name, ...)


    :SeeAlso:
      stochastic, Stochastic, dtrm, Deterministic, potential, Potential, Model,
      distributions
    """

    if obj is not None:
        if isinstance(obj, Stochastic):
            obj._observed=True
            return obj
        else:
            p = stochastic(__func__=obj, observed=True, **kwds)
            return p

    kwds['observed']=True
    def instantiate_observed(func):
        return stochastic(func, **kwds)

    return instantiate_observed

data = observed
