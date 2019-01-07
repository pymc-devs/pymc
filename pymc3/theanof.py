import numpy as np
import theano
from theano import theano, scalar, tensor as tt
from theano.configparser import change_flags
from theano.gof import Op
from theano.gof.graph import inputs
from theano.sandbox.rng_mrg import MRG_RandomStreams

from .blocking import ArrayOrdering
from .data import GeneratorAdapter
from .vartypes import typefilter, continuous_types

__all__ = ['gradient',
           'hessian',
           'hessian_diag',
           'inputvars',
           'cont_inputs',
           'floatX',
           'intX',
           'smartfloatX',
           'jacobian',
           'CallableTensor',
           'join_nonshared_inputs',
           'make_shared_replacements',
           'generator',
           'set_tt_rng',
           'tt_rng']


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


def cont_inputs(f):
    """
    Get the continuous inputs into a theano variables

    Parameters
    ----------
        a : theano variable

    Returns
    -------
        r : list of tensor variables that are continuous inputs
    """
    return typefilter(inputvars(f), continuous_types)


def floatX(X):
    """
    Convert a theano tensor or numpy array to theano.config.floatX type.
    """
    try:
        return X.astype(theano.config.floatX)
    except AttributeError:
        # Scalar passed
        return np.asarray(X, dtype=theano.config.floatX)


_conversion_map = {'float64': 'int32',
                   'float32': 'int16',
                   'float16': 'int8',
                   'float8': 'int8'}


def intX(X):
    """
    Convert a theano tensor or numpy array to theano.tensor.int32 type.
    """
    intX = _conversion_map[theano.config.floatX]
    try:
        return X.astype(intX)
    except AttributeError:
        # Scalar passed
        return np.asarray(X, dtype=intX)


def smartfloatX(x):
    """
    Convert non int types to floatX
    """
    if str(x.dtype).startswith('float'):
        x = floatX(x)
    return x

"""
Theano derivative functions
"""


def gradient1(f, v):
    """flat gradient of f wrt v"""
    return tt.flatten(tt.grad(f, v, disconnected_inputs='warn'))


empty_gradient = tt.zeros(0, dtype='float32')


def gradient(f, vars=None):
    if vars is None:
        vars = cont_inputs(f)

    if vars:
        return tt.concatenate([gradient1(f, v) for v in vars], axis=0)
    else:
        return empty_gradient


def jacobian1(f, v):
    """jacobian of f wrt v"""
    f = tt.flatten(f)
    idx = tt.arange(f.shape[0], dtype='int32')

    def grad_i(i):
        return gradient1(f[i], v)

    return theano.map(grad_i, idx)[0]


def jacobian(f, vars=None):
    if vars is None:
        vars = cont_inputs(f)

    if vars:
        return tt.concatenate([jacobian1(f, v) for v in vars], axis=1)
    else:
        return empty_gradient


def jacobian_diag(f, x):
    idx = tt.arange(f.shape[0], dtype='int32')

    def grad_ii(i):
        return theano.grad(f[i], x)[i]

    return theano.scan(grad_ii, sequences=[idx],
                       n_steps=f.shape[0],
                       name='jacobian_diag')[0]


@change_flags(compute_test_value='ignore')
def hessian(f, vars=None):
    return -jacobian(gradient(f, vars), vars)


@change_flags(compute_test_value='ignore')
def hessian_diag1(f, v):
    g = gradient1(f, v)
    idx = tt.arange(g.shape[0], dtype='int32')

    def hess_ii(i):
        return gradient1(g[i], v)[i]

    return theano.map(hess_ii, idx)[0]


@change_flags(compute_test_value='ignore')
def hessian_diag(f, vars=None):
    if vars is None:
        vars = cont_inputs(f)

    if vars:
        return -tt.concatenate([hessian_diag1(f, v) for v in vars], axis=0)
    else:
        return empty_gradient


def makeiter(a):
    if isinstance(a, (tuple, list)):
        return a
    else:
        return [a]


class IdentityOp(scalar.UnaryScalarOp):

    @staticmethod
    def st_impl(x):
        return x

    def impl(self, x):
        return x

    def grad(self, inp, grads):
        return grads

    def c_code(self, node, name, inp, out, sub):
        return "{z} = {x};".format(x=inp[0], z=out[0])

    def __eq__(self, other):
        return isinstance(self, type(other))

    def __hash__(self):
        return hash(type(self))


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


def reshape_t(x, shape):
    """Work around fact that x.reshape(()) doesn't work"""
    if shape != ():
        return x.reshape(shape)
    else:
        return x[0]


class CallableTensor:
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


scalar_identity = IdentityOp(scalar.upgrade_to_float, name='scalar_identity')
identity = tt.Elemwise(scalar_identity, name='identity')


class GeneratorOp(Op):
    """
    Generator Op is designed for storing python generators inside theano graph.

    __call__ creates TensorVariable
        It has 2 new methods
        - var.set_gen(gen) : sets new generator
        - var.set_default(value) : sets new default value (None erases default value)

    If generator is exhausted, variable will produce default value if it is not None,
    else raises `StopIteration` exception that can be caught on runtime.

    Parameters
    ----------
    gen : generator that implements __next__ (py3) or next (py2) method
        and yields np.arrays with same types
    default : np.array with the same type as generator produces
    """
    __props__ = ('generator',)

    def __init__(self, gen, default=None):
        super().__init__()
        if not isinstance(gen, GeneratorAdapter):
            gen = GeneratorAdapter(gen)
        self.generator = gen
        self.set_default(default)

    def make_node(self, *inputs):
        gen_var = self.generator.make_variable(self)
        return theano.Apply(self, [], [gen_var])

    def perform(self, node, inputs, output_storage, params=None):
        if self.default is not None:
            output_storage[0][0] = next(self.generator, self.default)
        else:
            output_storage[0][0] = next(self.generator)

    def do_constant_folding(self, node):
        return False

    __call__ = change_flags(compute_test_value='off')(Op.__call__)

    def set_gen(self, gen):
        if not isinstance(gen, GeneratorAdapter):
            gen = GeneratorAdapter(gen)
        if not gen.tensortype == self.generator.tensortype:
            raise ValueError('New generator should yield the same type')
        self.generator = gen

    def set_default(self, value):
        if value is None:
            self.default = None
        else:
            value = np.asarray(value, self.generator.tensortype.dtype)
            t1 = (False,) * value.ndim
            t2 = self.generator.tensortype.broadcastable
            if not t1 == t2:
                raise ValueError('Default value should have the '
                                 'same type as generator')
            self.default = value


def generator(gen, default=None):
    """
    Generator variable with possibility to set default value and new generator.
    If generator is exhausted variable will produce default value if it is not None,
    else raises `StopIteration` exception that can be caught on runtime.

    Parameters
    ----------
    gen : generator that implements __next__ (py3) or next (py2) method
        and yields np.arrays with same types
    default : np.array with the same type as generator produces

    Returns
    -------
    TensorVariable
        It has 2 new methods
        - var.set_gen(gen) : sets new generator
        - var.set_default(value) : sets new default value (None erases default value)
    """
    return GeneratorOp(gen, default)()


_tt_rng = MRG_RandomStreams()


def tt_rng(random_seed=None):
    """
    Get the package-level random number generator or new with specified seed.

    Parameters
    ----------
    random_seed : int
        If not None
        returns *new* theano random generator without replacing package global one

    Returns
    -------
    `theano.sandbox.rng_mrg.MRG_RandomStreams` instance
        `theano.sandbox.rng_mrg.MRG_RandomStreams`
        instance passed to the most recent call of `set_tt_rng`
    """
    if random_seed is None:
        return _tt_rng
    else:
        ret = MRG_RandomStreams(random_seed)
        return ret


def set_tt_rng(new_rng):
    """
    Set the package-level random number generator.

    Parameters
    ----------
    new_rng : `theano.sandbox.rng_mrg.MRG_RandomStreams` instance
        The random number generator to use.
    """
    # pylint: disable=global-statement
    global _tt_rng
    # pylint: enable=global-statement
    if isinstance(new_rng, int):
        new_rng = MRG_RandomStreams(new_rng)
    _tt_rng = new_rng


def floatX_array(x):
    return floatX(np.array(x))


def set_theano_conf(values):
    """Change the theano configuration and return old values.

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


def ix_(*args):
    """
    Theano np.ix_ analog

    See numpy.lib.index_tricks.ix_ for reference
    """
    out = []
    nd = len(args)
    for k, new in enumerate(args):
        if new is None:
            out.append(slice(None))
        new = tt.as_tensor(new)
        if new.ndim != 1:
            raise ValueError("Cross index must be 1 dimensional")
        new = new.reshape((1,)*k + (new.size,) + (1,)*(nd-k-1))
        out.append(new)
    return tuple(out)


def largest_common_dtype(tensors):
    dtypes = set(str(t.dtype) if hasattr(t, 'dtype')
                 else smartfloatX(np.asarray(t)).dtype
                 for t in tensors)
    return np.stack([np.ones((), dtype=dtype) for dtype in dtypes]).dtype
