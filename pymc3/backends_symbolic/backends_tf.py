import numpy as np
import tensorflow as tf
from tensorflow import Variable, Tensor
import os
import sys
import inspect

from .common import floatx, has_arg, makeiter

_SESSION = None

### Variables and Variable Manipulation ###

class TensorVariable(Variable):
    """Main Class for FreeRV and ObservedRV
       It's a Variable in TensorFlow
    """
    def getName(self):
        return None

class tag(object):
    '''This will be attached to TensorVariable, because TensorFlow variables don't have tag'''
    def __init__(self):
        self.test_val = np.array([])

def TensorVariableType(dtype=None,shape=None, name=None):
    """Returns the the type for the TensorVariable
       Formerly called TensorType
    """
    ### TODO: may need to add broadcastable to match theano backend.
    if dtype is None:
        dtype = floatx()
    return tf.Variable # returns a class?

def as_tensor_variable(x,name=None,dtype=floatx()):
    """Converts data, like numpy array, to a TensorVariable of appropriate type.
       Used for distribution parameters, like mu.
    """
    return(tf.convert_to_tensor(x, dtype=dtype,name=name))


def is_shared(var):
    ### tf Variables act like shared variables
    return isinstance(var, tf.Variable)

def is_variable(var):
    return isinstance(var, tf.Variable)

def is_constant(var):
    return isinstance(var, tf.constant)

def is_graphVariable(data):
    ### not going to do this at the moment.
    return(False)

def get_val(var):
    """Gets the current value of a symbolic variable
       TensorFlow requires current session in order to get value.
    """
    return(var.eval(session=get_session()))

### Mscl. ###

def floatX(X):
    """Convert a tensor or numpy array to config.floatX type."""
    try:
        return X.astype(floatx())
    except AttributeError:
        return np.asarray(X, dtype=floatx())

def set_symbolic_conf(values):
    """Used for theano. Don't need for tensorflow, but may in the future."""
    old = {}
    return old


### Functions ###

_MANUAL_VAR_INIT = False

def get_session():
    """Returns the TF session to be used by the backend. Required to evaluate
    computations.

    Returns
    -------
        A TensorFlow session.
    """
    global _SESSION

    default_session = tf.get_default_session()

    if default_session is not None:
        session = default_session
    else:
        if _SESSION is None:
            if not os.environ.get('OMP_NUM_THREADS'):
                config = tf.ConfigProto(allow_soft_placement=True)
            else:
                num_thread = int(os.environ.get('OMP_NUM_THREADS'))
                config = tf.ConfigProto(intra_op_parallelism_threads=num_thread,
                                        allow_soft_placement=True)
            _SESSION = tf.Session(config=config)
        session = _SESSION
    if not _MANUAL_VAR_INIT:
        with session.graph.as_default():
            variables = tf.global_variables()
            candidate_vars = []
            for v in variables:
                if not getattr(v, '_keras_initialized', False):
                    candidate_vars.append(v)
            if candidate_vars:
                # This step is expensive, so we only run it on variables
                # not already marked as initialized.
                is_initialized = session.run(
                    [tf.is_variable_initialized(v) for v in candidate_vars])
                uninitialized_vars = []
                for flag, v in zip(is_initialized, candidate_vars):
                    if not flag:
                        uninitialized_vars.append(v)
                    v._keras_initialized = True
                if uninitialized_vars:
                    session.run(tf.variables_initializer(uninitialized_vars))
    # hack for list_devices() function.
    # list_devices() function is not available under tensorflow r1.3.
    if not hasattr(session, 'list_devices'):
        session.list_devices = lambda: device_lib.list_local_devices()
    return session


class Function(object):
    """
       Runs a computation graph.

       This is a class to make syntax similar between theano.functions
       and tensorflow session.run().

       The initialization associates all the inputs, outputs, and update operations.
       Calling the Function evaluates the comptutation.

       See Keras.backends for similar implementation.

       Note: @memoize is now gone.

       Inputs:
       -------
       inputs: Feed placeholders to the computation graph.
       outputs: Output tensors to fetch.
       updates: Additional update ops to be run at function call.
       name: a name to help users identify what this function does.
    """

    def __init__(self, inputs, outputs, updates=None, name=None, **session_kwargs):
        updates = updates or []
        if not isinstance(inputs, (list, tuple)):
            raise TypeError('`inputs` to a TensorFlow backend function '
                            'should be a list or tuple.')
        if not isinstance(outputs, (list, tuple)):
            raise TypeError('`outputs` of a TensorFlow backend function '
                            'should be a list or tuple.')
        if not isinstance(updates, (list, tuple)):
            raise TypeError('`updates` in a TensorFlow backend function '
                            'should be a list or tuple.')
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        with tf.control_dependencies(self.outputs):
            updates_ops = []
            for update in updates:
                if isinstance(update, tuple):
                    p, new_p = update
                    updates_ops.append(tf.assign(p, new_p))
                else:
                    # assumed already an op
                    updates_ops.append(update)
            self.updates_op = tf.group(*updates_ops)
        self.name = name
        # additional tensor substitutions
        self.feed_dict = session_kwargs.pop('feed_dict', {})
        # additional operations
        self.fetches = session_kwargs.pop('fetches', [])
        if not isinstance(self.fetches, list):
            self.fetches = [self.fetches]
        self.session_kwargs = session_kwargs

    def __call__(self, inputs):
        if isinstance(inputs, (list, tuple)):
            feed_dict = self.feed_dict.copy()
            for tensor, value in zip(self.inputs, inputs):
                feed_dict[tensor] = value
        elif isinstance(inputs,(dict)):
            feed_dict = self.feed_dict.copy()
            # This maps inputs to those in self.inputs.
            # The inputs come in in the order of the variables, but self.inputs
            # is ordered based on which variable is main input (ie. which distribution)
            for tensor in self.inputs:
                feed_dict[tensor] = inputs[str(tensor)]
        else:
            print('error: input needs to be dict,list,or tuple')

        fetches = self.outputs + [self.updates_op] + self.fetches
        session = get_session()
        updated = session.run(fetches=fetches, feed_dict=feed_dict)
        return updated[:len(self.outputs)]


def function(inputs, outputs, updates=None, **kwargs):
    """Instantiates a Function class. Also checks validity of arguments. """
    if kwargs:
        for key in kwargs:
            if not (has_arg(tf.Session.run, key, True) or has_arg(Function.__init__, key, True)):
                msg = 'Invalid argument "%s" passed to K.function with TensorFlow backend' % key
                raise ValueError(msg)
    return Function(inputs, outputs, updates=updates, **kwargs)


### Math ###

def prod(x,axis=None, keepdims=False):
    return(tf.reduce_prod(x, axis, keepdims))

def tsum(x,axis=None, keepdims=False):
    return(tf.reduce_sum(x,axis,keepdims))

def sqrt(x):
    zero = _to_tensor(0., x.dtype.base_dtype)
    inf = _to_tensor(np.inf, x.dtype.base_dtype)
    x = tf.clip_by_value(x, zero, inf)
    return(tf.sqrt(x))

def log(x):
   return(tf.log(x))

def grad(x):
    return(tf.gradients(loss, variables, colocate_gradients_with_ops=True))

def bound(logp, condition):
  """
  A version of bound from pymc3 in tensorflow. Unlike the original version,
  this only bounds a log probability density with one condition at the moment.

  """
  def f1(): return logp
  def f2(): return -np.inf

  return tf.cond(condition, f1, f2)



### Graph Manipulation ###

def inputvars(a):
    """
    In theano, this will traverse the graph and find inputs necessary
    to get the value of a current variable.

    I didn't need this functionality in TensorFlow,
    so this function just returns the input variables at the moment that
    are of type Variable.

    Parameters
    ----------
        a : list of tensor variables.

    Returns
    -------
        r : list of tensor variables that are inputs
    """

    return [v for v in makeiter(a) if isinstance(v, tf.Variable)]


### Things required for calculating Delta Logp ###

def make_shared_replacements(vars, model):
    """
    tf.Variables are already shared. All this function does is
    get the other variables in the model, excluding the current
    variable under consideration.

    """
    othervars = set(model.vars) - set(vars) # is this even necessary? Oh mybe so it doesn't pass itself!
    return {var: var for var in othervars} # this is bit odd to have the key the same as value

def logp(logp, vars, shared):
    """Returns a Function for logp with the inputs for the current stepself.
       Order matters. The variable being updated is first, and then the other
       variables in the model are in shared.

    """
    inputs = vars+[v for v in shared.values()]
    outputs = [logp]
    f = function(inputs,outputs)
    return f

def delta_logp(logp,vars,shared):
    """Should returns a Function for change in logp
       Did not make yet. Using two calls to logp instead.
    """
    return(None)
