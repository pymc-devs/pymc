# Refactored by Thomas Wiecki
# Original implementation by Qiang Liu and Dilin Wang: https://github.com/DartML/Stein-Variational-Gradient-Descent/blob/master/python/svgd.py
# Many thanks to Dilin Wang for help with getting this going
# (c) 2016 Qiang Liu and Dilin Wang

import numpy as np
import theano
import theano.tensor as tt
from tqdm import tqdm
from .updates import adagrad

import pymc3 as pm
from pymc3.model import modelcontext

def rbf_kernel(X):
    # TODO. rbf may not be a good choice for high dimension data
    XY = tt.dot(X, X.transpose())
    x2 = tt.reshape(tt.sum(tt.square(X), axis=1), (X.shape[0], 1))
    X2e = tt.repeat(x2, X.shape[0], axis=1)
    H = tt.sub(tt.add(X2e, X2e.transpose()), 2 * XY)

    V = H.flatten()

    # median distance
    h = tt.switch(tt.eq((V.shape[0] % 2), 0),
                  # if even vector
                  tt.mean(tt.sort(V)[ ((V.shape[0] // 2) - 1) : ((V.shape[0] // 2) + 1) ]),
                  # if odd vector
                  tt.sort(V)[V.shape[0] // 2])

    h = tt.sqrt(0.5 * h / tt.log(X.shape[0].astype('float32') + 1.0))

    Kxy = tt.exp(-H / h ** 2 / 2.0)
    dxkxy = -tt.dot(Kxy, X)
    sumkxy = tt.sum(Kxy, axis=1).dimshuffle(0, 'x')
    dxkxy = tt.add(dxkxy, tt.mul(X, sumkxy)) / (h ** 2)

    return (Kxy, dxkxy)


def _make_vectorized_logp_grad(vars, model, X):
    theano.config.compute_test_value = 'ignore'
    shared = pm.make_shared_replacements(vars, model)

    # For some reason can't use model.basic_RVs here as the ordering
    # will not match up with that of vars.
    factors = [var.logpt for var in vars + model.observed_RVs] + model.potentials
    logpt_grad = pm.theanof.gradient(tt.add(*map(tt.sum, factors)))

    [logp_grad], inarray = pm.join_nonshared_inputs([logpt_grad], vars, shared)

    # Callable tensor
    def logp_grad_(input):
        return theano.clone(logp_grad, {inarray: input}, strict=False)

    logp_grad_vec = theano.map(logp_grad_, X)[0]

    return logp_grad_vec


def _svgd_gradient(vars, model, X, logp_grad_vec):
    kxy, dxkxy = rbf_kernel(X)
    svgd_grad = (tt.dot(kxy, logp_grad_vec) + dxkxy) / X.shape[0].astype('float32')  # default

    return svgd_grad


def svgd(vars=None, n=5000, n_particles=100, jitter=.01,
         optimizer=adagrad, start=None, progressbar=True,
         random_seed=None, model=None):

    if random_seed is not None:
        seed(random_seed)

    model = modelcontext(model)
    if vars is None:
        vars = model.vars
    vars = pm.inputvars(vars)

    if start is None:
        start = model.test_point
    start = model.dict_to_array(start)

    # Initialize particles
    x0 = np.tile(start, (n_particles, 1))
    x0 += np.random.normal(0, jitter, x0.shape)

    theta = theano.shared(x0)

    # Create theano svgd gradient expression and function
    logp_grad_vec = _make_vectorized_logp_grad(vars, model, theta)
    svgd_grad = -1 * _svgd_gradient(vars, model, theta, logp_grad_vec) # maximize

    svgd_updates = optimizer([svgd_grad], [theta], learning_rate=1e-3)

    i = tt.iscalar('i')
    svgd_step = theano.function([i], [i],
                                updates=svgd_updates)
    # Run svgd optimization
    if progressbar:
        progress = tqdm(np.arange(n))
    else:
        progress = np.arange(n)

    for ii in progress:
        svgd_step(ii)

    theta_val = theta.get_value()

    # Build trace
    strace = pm.backends.NDArray()
    strace.setup(theta_val.shape[0], 1)

    for p in theta_val:
        strace.record(model.bijection.rmap(p))
    strace.close()

    trace = pm.backends.base.MultiTrace([strace])

    return trace
