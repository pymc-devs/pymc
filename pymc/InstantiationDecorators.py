"""
The decorators stochastic, deterministic, discrete_stochastic, binary_stochastic, potential and data
are defined here, but the actual objects are defined in PyMCObjects.py
"""

__all__ = [
    'stochastic',
    'stoch',
    'deterministic',
    'dtrm',
    'potential',
    'pot',
    'data',
    'observed',
    'robust_init',
    'disable_special_methods',
    'enable_special_methods',
    'check_special_methods']

import sys
import inspect
import pdb
from imp import load_dynamic
from .PyMCObjects import Stochastic, Deterministic, Potential
from .Node import ZeroProbability, ContainerBase, Node, StochasticMeta
from .Container import Container
import numpy as np

special_methods_available = [True]


def disable_special_methods(sma=special_methods_available):
    sma[0] = False


def enable_special_methods(sma=special_methods_available):
    sma[0] = True


def check_special_methods(sma=special_methods_available):
    return sma[0]

from . import six


def _extract(__func__, kwds, keys, classname, probe=True):
    """
    Used by decorators stochastic and deterministic to inspect declarations
    """

    # Add docs and name
    kwds['doc'] = __func__.__doc__
    if not 'name' in kwds:
        kwds['name'] = __func__.__name__
    # kwds.update({'doc':__func__.__doc__, 'name':__func__.__name__})

    # Instanitate dictionary of parents
    parents = {}

    # This gets used by stochastic to check for long-format logp and random:
    if probe:
        cur_status = check_special_methods()
        disable_special_methods()
        # Define global tracing function (I assume this is for debugging??)
        # No, it's to get out the logp and random functions, if they're in
        # there.

        def probeFunc(frame, event, arg):
            if event == 'return':
                locals = frame.f_locals
                kwds.update(dict((k, locals.get(k)) for k in keys))
                sys.settrace(None)
            return probeFunc

        sys.settrace(probeFunc)

        # Get the functions logp and random (complete interface).
        # Disable special methods to prevent the formation of a hurricane of
        # Deterministics
        try:
            __func__()
        except:
            if 'logp' in keys:
                kwds['logp'] = __func__
            else:
                kwds['eval'] = __func__
        # Reenable special methods.
        if cur_status:
            enable_special_methods()

    for key in keys:
        if not key in kwds:
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
        err_str =  classname + ' ' + __func__.__name__ + \
            ': no parent provided for the following labels:'
        for i in range(arg_deficit):
            err_str += " " + args[i + ('value' in args)]
            if i < arg_deficit - 1:
                err_str += ','
        raise ValueError(err_str)

    # Fill in parent dictionary
    try:
        parents.update(dict(zip(args[-len(defaults):], defaults)))
    except TypeError:
        pass

    value = parents.pop('value', None)

    return (value, parents)


def stochastic(__func__=None, __class__=Stochastic,
               binary=False, discrete=False, **kwds):
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

    keys = ['logp', 'random', 'rseed']

    instantiate_p.kwds = kwds

    if __func__:
        return instantiate_p(__func__)

    return instantiate_p

# Shortcut alias
stoch = stochastic


def potential(__func__=None, **kwds):
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
        junk, parents = _extract(
            __func__, kwds, keys, 'Potential', probe=False)
        return Potential(parents=parents, **kwds)

    keys = ['logp']

    instantiate_pot.kwds = kwds

    if __func__:
        return instantiate_pot(__func__)

    return instantiate_pot
pot = potential


def deterministic(__func__=None, **kwds):
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
        junk, parents = _extract(
            __func__, kwds, keys, 'Deterministic', probe=False)
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
            obj._observed = True
            return obj
        else:
            p = stochastic(__func__=obj, observed=True, **kwds)
            return p

    kwds['observed'] = True

    def instantiate_observed(func):
        return stochastic(func, **kwds)

    return instantiate_observed

data = observed


def robust_init(stochclass, tries, *args, **kwds):
    """Robust initialization of a Stochastic.

    If the evaluation of the log-probability returns a ZeroProbability
    error, due for example to a parent being outside of the support for
    this Stochastic, the values of parents are randomly sampled until
    a valid log-probability is obtained.

    If the log-probability is still not valid after `tries` attempts, the
    original ZeroProbability error is raised.

    :Parameters:
    stochclass : Stochastic, eg. Normal, Uniform, ...
      The Stochastic distribution to instantiate.
    tries : int
      Maximum number of times parents will be sampled.
    *args, **kwds
      Positional and keyword arguments to declare the Stochastic variable.

    :Example:
    >>> lower = pymc.Uniform('lower', 0., 2., value=1.5, rseed=True)
    >>> pymc.robust_init(pymc.Uniform, 100, 'data', lower=lower, upper=5, value=[1,2,3,4], observed=True)
    """
    # Find the direct parents
    stochs = [arg for arg in (list(args) + list(kwds.values()))
              if isinstance(arg.__class__, StochasticMeta)]

    # Find the extended parents
    parents = stochs
    for s in stochs:
        parents.extend(s.extended_parents)

    extended_parents = set(parents)

    # Select the parents with a random method.
    random_parents = [
        p for p in extended_parents if p.rseed is True and hasattr(
            p,
            'random')]

    for i in range(tries):
        try:
            return stochclass(*args, **kwds)
        except ZeroProbability:
            exc = sys.exc_info()
            for parent in random_parents:
                try:
                    parent.random()
                except:
                    six.reraise(*exc)

    six.reraise(*exc)
