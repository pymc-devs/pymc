#-------------------------------------------------------------
# Decorators
#-------------------------------------------------------------

import numpy as np
from numpy import inf, random, sqrt
import string
import inspect
import types, copy
import distributions
from Node import ZeroProbability

def deterministic_to_NDarray(arg):
    if isinstance(arg,proposition5.Deterministic):
        return arg.value
    else:
        return arg

def prop(func):
  """Function decorator for defining property attributes

  The decorated function is expected to return a dictionary
  containing one or more of the following pairs:
      fget - function for getting attribute value
      fset - function for setting attribute value
      fdel - function for deleting attribute
  This can be conveniently constructed by the locals() builtin
  function; see:
  http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/205183
  """
  return property(doc=func.__doc__, **func())
  
