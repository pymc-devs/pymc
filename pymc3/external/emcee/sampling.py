import pymc3 as pm
from pymc3.backends import NDArray
from pymc3.backends.base import MultiTrace
from pymc3.external.emcee.backends import EnsembleNDArray
from pymc3.external.emcee.step_methods import ExternalEnsembleStepShared, AffineInvariantEnsemble
from pymc3.sampling import _update_start_vals
from theano.gradient import np

__all__ = ['sample', 'build_start_points', 'EnsembleNDArray', 'AffineInvariantEnsemble']

# TODO: create generic EnsembleTrace to use a normal Basetrace (create duplicate parameter names for particles?)


def get_random_starters(nparticles, model):
    return {v.name: np.asarray([v.distribution.random() for i in range(nparticles)]) for v in model.vars}


def build_start_points(nparticles, method='random', model=None, **kwargs):
    if method == 'random':
        return get_random_starters(nparticles, model)
    else:
        start, _ = pm.init_nuts(method, nparticles, model=model, **kwargs)
        return {v: start.get_values(v) for v in start.varnames}


def sample(draws=500, step='affine_invariant', init='random', n_init=200000, start=None,
           trace=None, nparticles=None, tune=500, progressbar=True, model=None, random_seed=-1,
           live_plot=False, discard_tuned_samples=True, **kwargs):

    if start is None:
        start = {}

    model = pm.modelcontext(model)
    vars = pm.inputvars(model.vars)

    if step == 'affine_invariant':
        step = AffineInvariantEnsemble
    elif not isinstance(step, ExternalEnsembleStepShared):
        raise ValueError("Unknown step {}".format(step))

    sampler = step(vars, nparticles, tune > 0, tune, model, **kwargs)
    nparticles = sampler.nparticles

    if trace is None:
        trace = EnsembleNDArray('mcmc', model, vars, nparticles)
    elif not isinstance(trace, EnsembleNDArray):
        raise TypeError("trace must be of type EnsembleNDArray")

    _start = build_start_points(nparticles, init, model)
    _update_start_vals(start, _start, model)
    trace = pm.sample(draws, sampler, init, n_init, start, trace, 0, 1, tune, None, None, progressbar, model,
                      random_seed, live_plot, discard_tuned_samples, **kwargs)

    traces = []
    for i in range(nparticles):
        tr = NDArray('mcmc', model, vars)
        tr.setup(len(trace), i)
        for varname in trace.varnames:
            tr.samples[varname] = trace[varname][:, i]
            tr.draw_idx = len(trace)
        traces.append(tr)
    return MultiTrace(traces)
