"""
pymc.NumpyDeterministics
"""

__docformat__='reStructuredText'
from . import PyMCObjects as pm
import numpy as np
from numpy import sum, ones, zeros, ravel, shape, size, newaxis
from .utils import find_element
import inspect

from pymc import six
xrange = six.moves.xrange

#accumulations 
_boolean_accumulation_deterministics = ['any' , 'all']
_accumulation_deterministics = ['sum']#['sum', 'prod']


#transformations (broadcasted)
_generic = ['abs', 'exp', 'log', 'sqrt','expm1', 'log1p']
_trig = ['sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan']
_hyp_trig = ['sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh']
_transformation_deterministics = _generic + _trig + _hyp_trig
_misc_funcs1 = ['arctan2', 'hypot']

__all__ = _accumulation_deterministics  + _boolean_accumulation_deterministics+ _transformation_deterministics + _misc_funcs1

def deterministic_from_funcs(name, eval, jacobians={}, jacobian_formats={}, dtype=np.float, mv=False):
    """
    Return a Stochastic subclass made from a particular distribution.

    :Parameters:
      name : string
        The name of the new class.
      jacobians : function
        The log-probability function.
      random : function
        The random function
      dtype : numpy dtype
        The dtype of values of instances.
      mv : boolean
        A flag indicating whether this class represents
        array-valued variables.

    """

    (args, varargs, varkw, defaults) = inspect.getargspec(eval)
    parent_names = args[0:]
    try:
        parents_default = dict(zip(args[-len(defaults):], defaults))
    except TypeError: # No parents at all.
        parents_default = {}

    # Build docstring from distribution
    docstr = name[0]+' = '+name + '('.join(parent_names)+')\n\n'
    docstr += 'Deterministic variable with '+name+' distribution.\nParents are: '+', '.join(parent_names) + '.\n\n'
    docstr += 'Docstring of evaluatio function:\n'
    docstr += eval.__doc__

    return new_deterministic_class(dtype, name, parent_names, parents_default, docstr, eval, jacobians, jacobian_formats)
    
def new_deterministic_class(*new_class_args):
    """
    Returns a new class from a distribution.

    :Parameters:
      dtype : numpy dtype
        The dtype values of instances of this class.
      name : string
        Name of the new class.
      parent_names : list of strings
        The labels of the parents of this class.
      parents_default : list
        The default values of parents.
      docstr : string
        The docstring of this class.
      eval : function
        The function for this class.
      jacobians : dictionary of functions
        The dictionary of jacobian functions for the class
      jacobian_formats : dictionary of strings
        A dictionary indicating the format of each jacobian function
    """

    (dtype, name, parent_names, parents_default, docstr, eval, jacobians, jacobian_formats) = new_class_args
    class new_class(pm.Deterministic):
        __doc__ = docstr
        def __init__(self, *args, **kwds):
            (dtype, name, parent_names, parents_default, docstr, eval, jacobians, jacobian_formats) = new_class_args
            parents=parents_default

            # Figure out what argument names are needed.
            arg_keys = [ 'parents',  'trace', 'doc', 'debug', 'plot', 'verbose']
            arg_vals = [ parents,  False, True, None, False, -1]

            arg_dict_out = dict(zip(arg_keys, arg_vals))
            args_needed =  parent_names + arg_keys[2:]

            # Sort positional arguments
            for i in xrange(len(args)):
                try:
                    k = args_needed.pop(0)
                    if k in parent_names:
                        parents[k] = args[i]
                    else:
                        arg_dict_out[k] = args[i]
                except:
                    raise ValueError('Too many positional arguments provided. Arguments for class ' + self.__class__.__name__ + ' are: ' + str(all_args_needed))


            # Sort keyword arguments
            for k in args_needed:
                if k in parent_names:
                    try:
                        parents[k] = kwds.pop(k)
                    except:
                        if k in parents_default:
                            parents[k] = parents_default[k]
                        else:
                            raise ValueError('No value given for parent ' + k)
                elif k in arg_dict_out.keys():
                    try:
                        arg_dict_out[k] = kwds.pop(k)
                    except:
                        pass

            # Remaining unrecognized arguments raise an error.
            if len(kwds) > 0:
                raise TypeError('Keywords '+ str(kwds.keys()) + ' not recognized. Arguments recognized are ' + str(args_needed))

            # Call base class initialization method
            if arg_dict_out.pop('debug'):
                pass
            else:
                parent_strs = []
                for key in parents.keys():
                    parent_strs.append(str(key))
                    
                instance_name = name + '('+','.join(parent_strs)+')'
                
                pm.Deterministic.__init__(self, name = instance_name, eval=eval, jacobians = jacobians, jacobian_formats = jacobian_formats, dtype=dtype, **arg_dict_out)

    new_class.__name__ = name
    new_class.parent_names = parent_names

    return new_class
    

_sum_hist = {}
def sum_jacobian_a (a, axis):
    try:
        return _sum_hist[shape(a)]
    except KeyError:
        j = ones(shape(a))
        _sum_hist[shape(a)] = j
        return j

sum_jacobians = {'a' : sum_jacobian_a}
    
abs_jacobians = {'x' : lambda x : np.sign(x) }
exp_jacobians = {'x' : lambda x : np.exp(x)  }
log_jacobians = {'x' : lambda x : 1.0/x      }
sqrt_jacobians = {'x': lambda x : .5    * x **-.5}
hypot_jacobians = {'x1' : lambda x1, x2 : (x1**2 + x2**2)**-.5 * x1,
                   'x2' : lambda x1, x2 : (x1**2 + x2**2)**-.5 * x2}
expm1_jacobians = exp_jacobians
log1p_jacobians = {'x' : lambda x : 1.0/(1.0 + x)}

sin_jacobians = {'x' : lambda x : np.cos(x)  }
cos_jacobians = {'x' : lambda x : -np.sin(x) }
tan_jacobians = {'x' : lambda x : 1 + np.tan(x)**2}

arcsin_jacobians = {'x' : lambda x :  (1.0-x**2)**-.5}
arccos_jacobians = {'x' : lambda x : -(1.0-x**2)**-.5}
arctan_jacobians = {'x' : lambda x :  1.0/(1.0+x**2) }
arctan2_jacobians = {'x1' : lambda x1, x2 :  x2/ (x2**2 + x1**2),
                     'x2' : lambda x1, x2 : -x1/ (x2**2 + x1**2)}
# found in www.math.smith.edu/phyllo/Assets/pdf/findcenter.pdf p21

sinh_jacobians = {'x' : lambda x : np.cosh(x)}
cosh_jacobians = {'x' : lambda x : np.sinh(x)}
tanh_jacobians = {'x' : lambda x : 1.0 - np.tanh(x)**2}

arcsinh_jacobians = {'x' : lambda x : (1+x**2)**-.5}
arccosh_jacobians = {'x' : lambda x : (x+1)**-.5*(x-1.0)**-.5}
arctanh_jacobians = {'x' : lambda x : 1.0/(1-x**2) }


def wrap_function_accum(function):
    def wrapped_function(a, axis = None):
        return function(a, axis)
    wrapped_function.__doc__ = function.__doc__
    
    return wrapped_function

for function_name in _accumulation_deterministics:
    wrapped_function = wrap_function_accum(find_element(function_name, np, error_on_fail = True))
    
    jacobians = find_element(function_name + "_jacobians", locals(), error_on_fail = True)
    
    locals()[function_name] = deterministic_from_funcs(function_name, wrapped_function, jacobians, jacobian_formats = {'a' : 'accumulation_operation'})


for function_name in _boolean_accumulation_deterministics:
    wrapped_function = wrap_function_accum(find_element(function_name, np, error_on_fail = True))

    locals()[function_name] = deterministic_from_funcs(function_name, wrapped_function)



def wrapped_function_trans(function):
    def wrapped_function(x):
        return function(x)
    wrapped_function.__doc__ = function.__doc__
    
    return wrapped_function

for function_name in _transformation_deterministics:
    wrapped_function = wrapped_function_trans(find_element(function_name, np, error_on_fail = True))
    
    jacobians = find_element(function_name + "_jacobians", locals(), error_on_fail = True)
    locals()[function_name] = deterministic_from_funcs(function_name, wrapped_function, jacobians, jacobian_formats = {'x' : 'transformation_operation'}) 
    

def wrap_function_misc1(function):
    def wrapped_function(x1, x2):
        return function(x1, x2)
    wrapped_function.__doc__ = function.__doc__
    
    return wrapped_function

for function_name in _misc_funcs1:
    wrapped_function = wrap_function_misc1(find_element(function_name, np, error_on_fail = True))
    
    jacobians = find_element(function_name + "_jacobians", locals(), error_on_fail = True)
    
    locals()[function_name] = deterministic_from_funcs(function_name, wrapped_function, jacobians, jacobian_formats = {'x1' : 'broadcast_operation',
                                                                                                                       'x2' : 'broadcast_operation'})    

