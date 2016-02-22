
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
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from collections import OrderedDict, namedtuple

__all__ = ['advi']

ADVIFit = namedtuple('ADVIFit', 'means, stds, elbo_vals')

def advi(vars=None, start=None, model=None, n=5000, accurate_elbo=False, 
    learning_rate=.001, epsilon=.1, verbose=1):
    model = modelcontext(model)
    if start is None:
        start = model.test_point

    if vars is None:
        vars = model.vars
    vars = inputvars(vars)

    # Create variational gradient tensor
    grad, elbo, inarray, shared = variational_gradient_estimate(vars, model, accurate_elbo=accurate_elbo)

    # Set starting values
    for var, share in shared.items():
        share.set_value(start[str(var)])

    order = ArrayOrdering(vars)
    bij = DictToArrayBijection(order, start)
    u_start = bij.map(start)
    w_start = np.zeros_like(u_start)
    uw = np.concatenate([u_start, w_start])

    result, elbos = run_adagrad(uw, grad, elbo, inarray, n, learning_rate=learning_rate, epsilon=epsilon, verbose=verbose)

    l = result.size / 2

    u = bij.rmap(result[:l])
    w = bij.rmap(result[l:])
    # w is in log space
    for var in w.keys():
        w[var] = np.exp(w[var])
    return ADVIFit(u, w, elbos)

def run_adagrad(uw, grad, elbo, inarray, n, learning_rate=.001, epsilon=.1,
                verbose=1):
    shared_inarray = theano.shared(uw, 'uw_shared')
    grad = CallableTensor(grad)(shared_inarray)
    elbo = CallableTensor(elbo)(shared_inarray)

    updates = adagrad(grad, shared_inarray, learning_rate=learning_rate, epsilon=epsilon, n=10)

    # Create in-place update function
    f = theano.function([], [shared_inarray, grad, elbo], updates=updates)

    # Run adagrad steps
    elbos = np.empty(n)
    for i in range(n):
        uw_i, g, e = f()
        elbos[i] = e
        if verbose and not i % (n//10):
            print('Iteration {0} [{1}%]: ELBO = {2:.2f}'.format(i, 100*i/n, e))

    return uw_i, elbos

def variational_gradient_estimate(vars, model, accurate_elbo=False):
    theano.config.compute_test_value = 'ignore'
    shared = make_shared_replacements(vars, model)
    [logp], inarray = join_nonshared_inputs([model.logpt], vars, shared)
    logp = CallableTensor(logp)

    uw = dvector('uw')
    uw.tag.test_value = np.concatenate([inarray.tag.test_value,
                                        inarray.tag.test_value])

    # Naive Monte-Carlo
    r = MRG_RandomStreams(seed=1)
    n = r.normal(size=inarray.tag.test_value.shape)

    gradient_estimate, elbo_estimate = inner_gradients(logp, n, uw)

    # More accurate estimation of ELBO
    if accurate_elbo:
        elbo_estimate = inner_elbo(logp, uw, 100)

    return gradient_estimate, elbo_estimate, inarray, shared

def inner_gradients(logp, n, uw):
    # uw contains both, mean and diagonals
    l = (uw.size/2).astype('int64')
    u = uw[:l]
    w = uw[l:]

    # Formula 6 in the paper
    q = n * exp(w) + u

    duw = gradient(logp(q), [uw])
    elbo = logp(q) + T.sum(w)

    # Add gradient of entropy term (just equal to element-wise 1 here), formula 6
    duw = theano.tensor.set_subtensor(duw[l:], duw[l:] + 1)
    return duw, elbo

# This function can be used to
def inner_elbo(logp, uw, n_mcsamples_elbo):
    """Compute get more accurate ELBO than the one in inner_gradients().

    Draw the specified number of MC samples.
    """
    # uw contains both, mean and diagonals
    l = (uw.size/2).astype('int64')
    u = uw[:l]
    w = uw[l:]

    r = MRG_RandomStreams(seed=1)
    n = r.normal(size=(n_mcsamples_elbo, u.tag.test_value.shape[0]))
    qs = n * exp(w) + u
    logps, _ = theano.scan(fn=lambda q: logp(q),
                           outputs_info=None,
                           sequences=[qs])
    elbo = T.mean(logps) + T.sum(w)

    return elbo

def adagrad(grad, param, learning_rate, epsilon, n):
    # Compute windowed adagrad using last n gradients
    i = theano.shared(np.array(0), 'i')
    value = param.get_value(borrow=True)
    accu = theano.shared(np.zeros(value.shape+(n,), dtype=value.dtype))

    # Append squared gradient vector to accu_new
    accu_new = theano.tensor.set_subtensor(accu[:,i], grad ** 2)
    i_new = theano.tensor.switch((i + 1) < n, i + 1, 0)

    updates = OrderedDict()
    updates[accu] = accu_new
    updates[i] = i_new

    accu_sum = accu_new.sum(axis=1)
    updates[param] = param - (-learning_rate * grad /
                              theano.tensor.sqrt(accu_sum + epsilon))
    return updates
