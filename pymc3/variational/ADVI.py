
'''
Created on Mar 12, 2011

@author: johnsalvatier
'''
from scipy import optimize
import numpy as np
from ..core import *

import theano
from ..theanof import make_shared_replacements, join_nonshared_inputs, CallableTensor, gradient
from theano.tensor import exp, concatenate, dvector
from theano.sandbox.rng_mrg import MRG_RandomStreams
from collections import OrderedDict

__all__ = ['advi']

def advi(vars=None, start=None, model=None, n=5000):
    model = modelcontext(model)
    if start is None:
        start = model.test_point

    if vars is None:
        vars = model.vars
    vars = inputvars(vars)

    grad, inarray, shared = variational_gradient_estimate(vars, model)

    for var, share in shared.items():
        share.set_value(start[str(var)])

    order = ArrayOrdering(vars)
    bij = DictToArrayBijection(order, start)
    u_start = bij.map(start)
    w_start = np.zeros_like(u_start)
    uw = np.concatenate([u_start, w_start])

    result = run_adagrad(uw, grad, inarray, n)

    l = result.size / 2

    u = bij.rmap(result[:l])
    w = bij.rmap(result[l:])
    # w is in log space
    for var in w.keys():
        w[var] = np.exp(w[var])
    return u, w

def run_adagrad(uw, grad, inarray, n):
    shared_inarray = theano.shared(uw, 'uw_shared')
    grad = CallableTensor(grad)(shared_inarray)

    updates = adagrad(grad, shared_inarray, learning_rate=-.1, epsilon=.1, n=50)

    f = theano.function([], [shared_inarray, grad], updates=updates)

    for i in range(n):
        uw_i, g = f()
    return uw_i

def variational_gradient_estimate(vars, model):
    theano.config.compute_test_value = 'ignore'
    shared = make_shared_replacements(vars, model)
    [logp], inarray = join_nonshared_inputs([model.logpt], vars, shared)
    logp = CallableTensor(logp)

    uw = dvector('uw')
    uw.tag.test_value = np.concatenate([inarray.tag.test_value,
                                        inarray.tag.test_value])

    r = MRG_RandomStreams(seed=1)
    n = r.normal(size=(1,))
    gradient_estimate = inner_gradients(logp, n, uw)

    return gradient_estimate, inarray, shared

def inner_gradients(logp, n, uw):
    l = (uw.size/2).astype('int64')
    u = uw[:l]
    w = uw[l:]

    q = n * exp(w) + u

    duw = gradient(logp(q), [uw])

    return theano.tensor.set_subtensor(duw[l:], duw[l:] + 1)

tprint = theano.printing.Print()

def adagrad(grad, param, learning_rate, epsilon, n):
    i = theano.shared(np.array(0), 'i')
    value = param.get_value(borrow=True)
    accu = theano.shared(np.zeros(value.shape+(n,), dtype=value.dtype))

    accu_new = theano.tensor.set_subtensor(accu[:,i], grad ** 2)
    i_new = theano.tensor.switch((i + 1) < n, i + 1, 0)

    updates = OrderedDict()
    updates[accu] = accu_new
    updates[i] = i_new
    accu_sum = accu_new.sum(axis=1)
    updates[param] = param - (learning_rate * grad /
                              theano.tensor.sqrt(accu_sum + epsilon))
    return updates
