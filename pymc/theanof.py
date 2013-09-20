from vartypes import typefilter, continuous_types
from theano import theano, tensor as t
from theano.gof.graph import inputs
from memoize import memoize

__all__ = ['gradient', 'hessian', 'hessian_diag', 'inputvars', 'cont_inputs']

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
    return [v for v in inputs(makeiter(a)) if isinstance(v, t.TensorVariable)]

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



"""
Theano derivative functions
"""
def gradient1(f, v):
    """flat gradient of f wrt v"""
    return t.flatten(t.grad(f, v, disconnected_inputs='warn'))


@memoize
def gradient(f, vars=None):
    if vars is None:
        vars = cont_inputs(f)

    return t.concatenate([gradient1(f, v) for v in vars], axis=0)


def jacobian1(f, v):
    """jacobian of f wrt v"""
    f = t.flatten(f)
    idx = t.arange(f.shape[0])

    def grad_i(i):
        return gradient1(f[i], v)

    return theano.map(grad_i, idx)[0]


@memoize
def jacobian(f, vars=None):
    if vars is None:
        vars = cont_inputs(f)

    return t.concatenate([jacobian1(f, v) for v in vars], axis=1)


@memoize
def hessian(f, vars=None):
    return -jacobian(gradient(f, vars), vars)


def hessian_diag1(f, v):

    g = gradient1(f, v)
    idx = t.arange(g.shape[0])

    def hess_ii(i):
        return gradient1(g[i], v)[i]

    return theano.map(hess_ii, idx)[0]


@memoize
def hessian_diag(f, vars=None):
    if vars is None:
        vars = cont_inputs(f)

    return -t.concatenate([hessian_diag1(f, v) for v in vars], axis=0)


def makeiter(a):
    if isinstance(a, (tuple, list)):
        return a
    else:
        return [a]
