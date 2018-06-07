import numpy as np
import theano
import theano.tensor as tt
from ..memoize import memoize
from ..blocking import ArrayOrdering
from theano.gof.graph import inputs
from .common import floatx, has_arg, makeiter
theano.config.floatX = floatx()
from theano.tensor.var import TensorVariable as TV

### Variables and Variable Manipulation ###

class TensorVariable(TV):
    """Main Class for FreeRV and ObservedRV
       It's a TensorVariable in Theano
    """
    def getName(self):
        return None

def TensorVariableType(dtype, shape, broadcastable=None):
    """Returns the the type for the TensorVariable
       Formerly called TensorType
    """
    if broadcastable is None:
        broadcastable = np.atleast_1d(shape) == 1
    return tt.TensorType(str(dtype), broadcastable)

def as_tensor_variable(variable,name=None):
    """Converts data, like numpy array, to a TensorVariable of appropriate type.
       Used for distribution parameters, like mu.
    """
    return(tt.as_tensor_variable(variable,name=name))

def is_shared(var):
    return isinstance(var, tt.sharedvar.SharedVariable)

def is_variable(var):
    return isinstance(var, tt.TensorVariable)

def is_constant(var):
    return isinstance(var, tt.TensorConstant)

def is_graphVariable(data):
    return(isinstance(data, theano.gof.graph.Variable))

def get_val(var):
    """Gets the current value of a symbolic variable
       Theano has different methods for getting the value depending on variable type
    """
    if isinstance(var, tt.TensorVariable):
        return var.tag.test_value
    if isinstance(var, tt.TensorConstant):
        return var.value
    if isinstance(var, tt.SharedVariable):
        return var.get_value()
    return(var)

### Mscl. ###

def floatX(X):
    """Convert a theano tensor or numpy array to config.floatX type."""
    try:
        return X.astype(floatx())
    except AttributeError:
        return np.asarray(X, dtype=floatx())

def set_symbolic_conf(values):
    """From pymc3.

    Old documnetation: Change the theano configuration and return old values.
    This is similar to `theano.configparser.change_flags`, but it
    returns the original values in a pickleable form.
    """
    variables = {}
    unknown = set(values.keys())
    for variable in theano.configparser._config_var_list:
        if variable.fullname in values:
            variables[variable.fullname] = variable
            unknown.remove(variable.fullname)
    if len(unknown) > 0:
        raise ValueError("Unknown theano config settings: %s" % unknown)

    old = {}
    for name, variable in variables.items():
        old_value = variable.__get__(True, None)
        try:
            variable.__set__(None, values[name])
        except Exception:
            for key, old_value in old.items():
                variables[key].__set__(None, old_value)
            raise
        old[name] = old_value
    return old


### Functions ###

class Function(object):
    """
       Runs a computation graph.

       This is a class to make syntax similar between theano.functions
       and tensorflow session.run().

       The initialization compiles the theano function. Calling the Function
       class calls the compiled theano function.

       See Keras.backends for similar implementation.

       Note: @memoize is now gone.

       Inputs:
       -------
       inputs: Feed placeholders to the computation graph.
       outputs: Output tensors to fetch.
       updates: Additional update ops to be run at function call.
       name: a name to help users identify what this function does.
    """
    def __init__(self, inputs, outputs, updates=[], name=None, **kwargs):
        unique_variables_to_update = {}
        for v, nv in updates:
            if v not in unique_variables_to_update:
                unique_variables_to_update[v] = nv
        updates = unique_variables_to_update.items()
        self.function = theano.function(inputs, outputs, updates=updates,
                                        allow_input_downcast=True,
                                        on_unused_input='ignore',
                                        name=name,
                                        **kwargs)
        self.name = name

    def __call__(self, inputs):
        return self.function(**inputs)


def function(inputs, outputs, updates=[], **kwargs):
    """Instantiates a Function class. Also checks validity of arguments. """
    if len(kwargs) > 0:
        for key in kwargs.keys():
            if not has_arg(theano.function, key, True):
                msg = 'Invalid argument "%s" passed to K.function with Theano backend' % key
                raise ValueError(msg)
    return Function(inputs, outputs, updates=updates, **kwargs)



### Math ###

def prod(x,axis=None,keepdims=False):
    return(tt.prod(x,axis=axis,keepdims=keepdims))

def tsum(x):
    return(tt.sum(x))

def sqrt(x):
    x =tt.clip(x, 0., np.inf)
    return(tt.sqrt(x))

def log(x):
   return(tt.log(x))

def grad(x):
    return(tt.grad(x))

def alltrue_scalar(vals):
    return tt.all([tt.all(1 * val) for val in vals])

def alltrue_elemwise(vals):
    ret = 1
    for c in vals:
        ret = ret * (1 * c)
    return ret

def bound(logp, *conditions, **kwargs):
  """
  Originally from pymc3.

  Original documentation:
  Bounds a log probability density with several conditions.

  Parameters
  ----------
  logp : float
  *conditions : booleans
  broadcast_conditions : bool (optional, default=True)
      If True, broadcasts logp to match the largest shape of the conditions.
      This is used e.g. in DiscreteUniform where logp is a scalar constant and the shape
      is specified via the conditions.
      If False, will return the same shape as logp.
      This is used e.g. in Multinomial where broadcasting can lead to differences in the logp.

  Returns
  -------
  logp with elements set to -inf where any condition is False
  """
  broadcast_conditions = kwargs.get('broadcast_conditions', True)

  if broadcast_conditions:
      alltrue = alltrue_elemwise
  else:
      alltrue = alltrue_scalar

  return tt.switch(alltrue(conditions), logp, -np.inf)


### Graph Manipulation ###

def inputvars(a):
    """
    Get the inputs into a theano variables

    Parameters
    ----------
        a : theano variable

    Returns
    -------
        r : list of tensor variables that are inputs
    """
    return [v for v in inputs(makeiter(a)) if isinstance(v, tt.TensorVariable)]


def make_shared_replacements(vars, model):
    """
    Makes shared replacements for all *other* variables than the ones passed.

    This way functions can be called many times without setting unchanging variables. Allows us
    to use func.trust_input by removing the need for DictToArrayBijection and kwargs.

    Parameters
    ----------
    vars : list of variables not to make shared
    model : model

    Returns
    -------
    Dict of variable -> new shared variable
    """
    othervars = set(model.vars) - set(vars)
    return {var: theano.shared(var.tag.test_value, var.name + '_shared') for var in othervars}


### Things required for calculating Delta Logp ###


class CallableTensor(object):
    """Turns a symbolic variable with one input into a function that returns symbolic arguments
    with the one variable replaced with the input.
    """

    def __init__(self, tensor):
        self.tensor = tensor

    def __call__(self, input):
        """ Replaces the single input of symbolic variable to be the passed argument.

        Parameters
        ----------
        input : TensorVariable
        """
        oldinput, = inputvars(self.tensor)
        return theano.clone(self.tensor, {oldinput: input}, strict=False)

def reshape_t(x, shape):
    """Work around fact that x.reshape(()) doesn't work"""
    if shape != ():
        return x.reshape(shape)
    else:
        return x[0]

def join_nonshared_inputs(xs, vars, shared, make_shared=False):
    """
    Takes a list of theano Variables and joins their non shared inputs into a single input.

    Parameters
    ----------
    xs : list of theano tensors
    vars : list of variables to join

    Returns
    -------
    tensors, inarray
    tensors : list of same tensors but with inarray as input
    inarray : vector of inputs
    """
    if not vars:
        raise ValueError('Empty list of variables.')

    joined = tt.concatenate([var.ravel() for var in vars])

    if not make_shared:
        tensor_type = joined.type
        inarray = tensor_type('inarray')
    else:
        inarray = theano.shared(joined.tag.test_value, 'inarray')

    ordering = ArrayOrdering(vars)
    inarray.tag.test_value = joined.tag.test_value

    get_var = {var.name: var for var in vars}
    replace = {
        get_var[var]: reshape_t(inarray[slc], shp).astype(dtyp)
        for var, slc, shp, dtyp in ordering.vmap}

    replace.update(shared)

    xs_special = [theano.clone(x, replace, strict=False) for x in xs]
    return xs_special, inarray

def logp(logp,vars,shared):
    """Not using at the moment in theano."""
    return(None)

def delta_logp(logp, vars, shared):
    """Returns a compiled theano function for change in logp
       The function will take 2 versions of the input that originally
       would go to logp.
    """

    [logp0], inarray0 = join_nonshared_inputs([logp], vars, shared)

    tensor_type = inarray0.type
    inarray1 = tensor_type('inarray1')

    logp1 = CallableTensor(logp0)(inarray1)

    f = theano.function([inarray1, inarray0], logp1 - logp0)
    f.trust_input = True
    return f
