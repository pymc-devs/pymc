# Refactored by Thomas Wiecki
# Original implementation by Qiang Liu and Dilin Wang: https://github.com/DartML/Stein-Variational-Gradient-Descent/blob/master/python/svgd.py
# Many thanks to Dilin Wang for help with getting this going
# (c) 2016 Qiang Liu and Dilin Wang

import numpy as np
import theano
import theano.tensor as tt
from tqdm import tqdm

import pymc3 as pm
from pymc3.model import modelcontext

def vgd_kernel(X):
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

    h = tt.sqrt(0.5 * h / tt.log(X.shape[0].astype('float32') + 1.0)) / 2.

    Kxy = tt.exp(-H / h ** 2 / 2.0)
    dxkxy = -tt.dot(Kxy, X)
    sumkxy = tt.sum(Kxy, axis=1).dimshuffle(0, 'x')
    dxkxy = tt.add(dxkxy, tt.mul(X, sumkxy)) / (h ** 2)

    return (Kxy, dxkxy)


def _make_svgd_step(theta, svgd_grad, stepsize=1e-3, bandwidth=-1,
                    alpha=0.9):
    # adagrad with momentum
    fudge_factor = 1e-6
    historical_grad = 0

    # Define optimization step
    i = tt.iscalar('i')
    grad_theta = svgd_grad[0]
    historical_grad = tt.switch(i,
                                historical_grad + grad_theta ** 2,
                                alpha * historical_grad + (1 - alpha) * (grad_theta ** 2))

    adj_grad = grad_theta / (fudge_factor + tt.sqrt(historical_grad))
    svgd_grad_func = theano.function([i], [],
                                     updates=[(theta, theta + stepsize * adj_grad)])

    return svgd_grad_func

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
    kxy, dxkxy = vgd_kernel(X)
    svgd_grad = (tt.dot(kxy, logp_grad_vec) + dxkxy) / X.shape[0].astype('float32')  # default

    return svgd_grad

def svgd(vars=None, n=5000, learning_rate=0.01, epsilon=.1, n_particles=100, jitter=.01,
         optimizer=None, start=None, progressbar=True, random_seed=None, model=None):

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
    svgd_grad = _svgd_gradient(vars, model, theta, logp_grad_vec)

    # Run svgd optimization
    svgd_step = _make_svgd_step(theta, svgd_grad,
                                stepsize=learning_rate)

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
