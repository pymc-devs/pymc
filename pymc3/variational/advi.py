
'''
Created on Mar 12, 2011

@author: johnsalvatier
'''
import numpy as np
from ..blocking import DictToArrayBijection, ArrayOrdering
from ..theanof import inputvars
from ..model import ObservedRV, modelcontext
from ..vartypes import discrete_types

import theano
from ..theanof import make_shared_replacements, join_nonshared_inputs, CallableTensor, gradient
from theano.tensor import exp, dvector
import theano.tensor as tt
from theano.sandbox.rng_mrg import MRG_RandomStreams
from collections import OrderedDict, namedtuple
from pymc3.sampling import NDArray
from pymc3.backends.base import MultiTrace

__all__ = ['advi']

ADVIFit = namedtuple('ADVIFit', 'means, stds, elbo_vals')

def check_discrete_rvs(vars):
    """Check that vars not include discrete variables, excepting ObservedRVs.
    """
    vars_ = [var for var in vars if not isinstance(var, ObservedRV)]
    if any([var.dtype in discrete_types for var in vars_]):
        raise ValueError('Model should not include discrete RVs for ADVI.')

def advi(vars=None, start=None, model=None, n=5000, accurate_elbo=False,
    learning_rate=.001, epsilon=.1, random_seed=20090425, verbose=1):
    """Run ADVI.

    Parameters
    ----------
    vars : object
        Random variables.
    start : Dict or None
        Initial values of parameters (variational means).
    model : Model
        Probabilistic model.
    n : int
        Number of interations updating parameters.
    accurate_elbo : bool
        If true, 100 MC samples are used for accurate calculation of ELBO.
    learning_rate: float
        Adagrad base learning rate.
    epsilon : float
        Offset in denominator of the scale of learning rate in Adagrad.
    random_seed : int
        Seed to initialize random state.

    Returns
    -------
    ADVIFit
        Named tuple, which includes 'means', 'stds', and 'elbo_vals'.

    'means' and 'stds' include parameters of the variational posterior.
    """

    model = modelcontext(model)
    if start is None:
        start = model.test_point

    if vars is None:
        vars = model.vars
    vars = inputvars(vars)

    check_discrete_rvs(vars)

    n_mcsamples = 100 if accurate_elbo else 1

    # Create variational gradient tensor
    grad, elbo, shared, _ = variational_gradient_estimate(
        vars, model, n_mcsamples=n_mcsamples, random_seed=random_seed)

    # Set starting values
    for var, share in shared.items():
        share.set_value(start[str(var)])

    order = ArrayOrdering(vars)
    bij = DictToArrayBijection(order, start)
    u_start = bij.map(start)
    w_start = np.zeros_like(u_start)
    uw = np.concatenate([u_start, w_start])

    result, elbos = run_adagrad(uw, grad, elbo, n, learning_rate=learning_rate, epsilon=epsilon, verbose=verbose)

    l = int(result.size / 2)

    u = bij.rmap(result[:l])
    w = bij.rmap(result[l:])
    # w is in log space
    for var in w.keys():
        w[var] = np.exp(w[var])
    return ADVIFit(u, w, elbos)

def replace_shared_minibatch_tensors(minibatch_tensors):
    """Replace shared variables in minibatch tensors with normal tensors.
    """
    givens = dict()
    tensors = list()

    for t in minibatch_tensors:
        if isinstance(t, theano.compile.sharedvalue.SharedVariable):
            t_ = t.type()
            tensors.append(t_)
            givens.update({t: t_})
        else:
            tensors.append(t)

    return tensors, givens

def run_adagrad(uw, grad, elbo, n, learning_rate=.001, epsilon=.1, verbose=1):
    """Run Adagrad parameter update.

    This function is only used in batch training.
    """
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
            if not i:
                print('Iteration {0} [{1}%]: ELBO = {2}'.format(i, 100*i//n, e.round(2)))
            else:
                avg_elbo = elbos[i-n//10:i].mean()
                print('Iteration {0} [{1}%]: Average ELBO = {2}'.format(i, 100*i//n, avg_elbo.round(2)))

    if verbose:
        avg_elbo = elbos[-n//10:].mean()
        print('Finished [100%]: Average ELBO = {}'.format(avg_elbo.round(2)))
    return uw_i, elbos

def variational_gradient_estimate(
    vars, model, minibatch_RVs=[], minibatch_tensors=[], total_size=None,
    n_mcsamples=1, random_seed=20090425):
    """Calculate approximate ELBO and its (stochastic) gradient.
    """

    theano.config.compute_test_value = 'ignore'
    shared = make_shared_replacements(vars, model)

    # Correction sample size
    r = 1 if total_size is None else \
        float(total_size) / minibatch_tensors[0].shape[0]

    other_RVs = set(model.basic_RVs) - set(minibatch_RVs)
    factors = [r * var.logpt for var in minibatch_RVs] + \
              [var.logpt for var in other_RVs] + model.potentials
    logpt = tt.add(*map(tt.sum, factors))

    [logp], inarray = join_nonshared_inputs([logpt], vars, shared)

    uw = dvector('uw')
    uw.tag.test_value = np.concatenate([inarray.tag.test_value,
                                        inarray.tag.test_value])

    elbo = elbo_t(logp, uw, inarray, n_mcsamples=n_mcsamples, random_seed=random_seed)

    # Gradient
    grad = gradient(elbo, [uw])

    return grad, elbo, shared, uw

def elbo_t(logp, uw, inarray, n_mcsamples, random_seed):
    """Create Theano tensor of approximate ELBO by Monte Carlo sampling.
    """
    l = (uw.size/2).astype('int64')
    u = uw[:l]
    w = uw[l:]

    # Callable tensor
    logp_ = lambda input: theano.clone(logp, {inarray: input}, strict=False)

    # Naive Monte-Carlo
    r = MRG_RandomStreams(seed=random_seed)

    if n_mcsamples == 1:
        n = r.normal(size=inarray.tag.test_value.shape)
        q = n * exp(w) + u
        elbo = logp_(q) + tt.sum(w) + 0.5 * l * (1 + np.log(2.0 * np.pi))
    else:
        n = r.normal(size=(n_mcsamples, u.tag.test_value.shape[0]))
        qs = n * exp(w) + u
        logps, _ = theano.scan(fn=lambda q: logp_(q),
                               outputs_info=None,
                               sequences=[qs])
        elbo = tt.mean(logps) + tt.sum(w) + 0.5 * l * (1 + np.log(2.0 * np.pi))

    return elbo

def adagrad(grad, param, learning_rate, epsilon, n):
    """Create Theano parameter (tensor) updates by Adagrad.
    """
    # Compute windowed adagrad using last n gradients
    i = theano.shared(np.array(0), 'i')
    value = param.get_value(borrow=True)
    accu = theano.shared(np.zeros(value.shape+(n,), dtype=value.dtype))

    # Append squared gradient vector to accu_new
    accu_new = tt.set_subtensor(accu[:,i], grad ** 2)
    i_new = tt.switch((i + 1) < n, i + 1, 0)

    updates = OrderedDict()
    updates[accu] = accu_new
    updates[i] = i_new

    accu_sum = accu_new.sum(axis=1)
    updates[param] = param - (-learning_rate * grad /
                              tt.sqrt(accu_sum + epsilon))
    return updates

def sample_vp(
    vparams, draws=1000, model=None, local_RVs=None, random_seed=20090425, 
    hide_transformed=True):
    """Draw samples from variational posterior. 

    Parameters
    ----------
    vparams : dict or pymc3.variational.ADVIFit
        Estimated variational parameters of the model.
    draws : int
        Number of random samples.
    model : pymc3.Model
        Probabilistic model.
    random_seed : int
        Seed of random number generator.
    hide_transformed : bool
        If False, transformed variables are also sampled. Default is True. 

    Returns
    -------
    trace : pymc3.backends.base.MultiTrace
        Samples drawn from the variational posterior.
    """
    model = modelcontext(model)

    if isinstance(vparams, ADVIFit):
        vparams = {
            'means': vparams.means,
            'stds': vparams.stds
        }

    ds = model.deterministics
    get_transformed = lambda v: v if v not in ds else v.transformed
    rvs = lambda x: [get_transformed(v) for v in x] if x is not None else []

    global_RVs = list(set(model.free_RVs) - set(rvs(local_RVs)))

    # Make dict for replacements of random variables
    r = MRG_RandomStreams(seed=random_seed)
    updates = {}
    for v in global_RVs:
        u = theano.shared(vparams['means'][str(v)]).ravel()
        w = theano.shared(vparams['stds'][str(v)]).ravel()
        n = r.normal(size=u.tag.test_value.shape)
        updates.update({v: (n * w + u).reshape(v.tag.test_value.shape)})

    if local_RVs is not None:
        ds = model.deterministics
        get_transformed = lambda v: v if v not in ds else v.transformed
        for v_, (uw, _) in local_RVs.items():
            v = get_transformed(v_)
            u = uw[0].ravel()
            w = uw[1].ravel()
            n = r.normal(size=u.tag.test_value.shape)
            updates.update({v: (n * tt.exp(w) + u).reshape(v.tag.test_value.shape)})

    # Replace some nodes of the graph with variational distributions
    vars = model.free_RVs
    samples = theano.clone(vars, updates)
    f = theano.function([], samples)

    # Random variables which will be sampled
    vars_sampled = [v for v in model.unobserved_RVs if not str(v).endswith('_')] \
                   if hide_transformed else \
                   [v for v in model.unobserved_RVs]

    varnames = [str(var) for var in model.unobserved_RVs]
    trace = NDArray(model=model, vars=vars_sampled)
    trace.setup(draws=draws, chain=0)

    for i in range(draws):
        # 'point' is like {'var1': np.array(0.1), 'var2': np.array(0.2), ...}
        point = {varname: value for varname, value in zip(varnames, f())}
        trace.record(point)

    return MultiTrace([trace])
