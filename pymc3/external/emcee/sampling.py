import pymc3 as pm
from pymc3.backends import NDArray
from pymc3.backends.base import MultiTrace
from pymc3.external.emcee.backends import EnsembleNDArray
from pymc3.external.emcee.step_methods import ExternalEnsembleStepShared, AffineInvariantEnsemble
from pymc3.sampling import _update_start_vals
from theano.gradient import np


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
    """Draw samples from the posterior using the given step methods.

    Multiple step methods are supported via compound step methods.

    Parameters
    ----------
    draws : int
        The number of samples to draw. Defaults to 500. The number of tuned
        samples are discarded by default. See discard_tuned_samples.
    step : at the moment only 'affine_invariant' is supported
    init : str {'ADVI', 'ADVI_MAP', 'MAP', 'NUTS', 'random'}
        Initialization method to use.

        * ADVI: Run ADVI to estimate starting points
        * ADVI_MAP: Initialize ADVI with MAP and use MAP as starting point.
        * MAP: Use the MAP as starting point.
        * NUTS: Run NUTS to estimate starting points and covariance matrix. If
          njobs > 1 it will sample starting points from the estimated posterior,
          otherwise it will use the estimated posterior mean.
        * random: sample from the prior distributions
    n_init : int
        Number of iterations of initializer
        If 'ADVI', number of iterations, if 'nuts', number of draws.
    start : dict
        Starting point in parameter space (or partial point)
        Defaults to trace.point(-1)) if there is a trace provided and
        model.test_point if not (defaults to empty dict).
    trace : backend, list, or MultiTrace
        This should be a backend instance, a list of variables to track,
        or a MultiTrace object with past values. If a MultiTrace object
        is given, it must contain samples for the chain number `chain`.
        If None or a list of variables, the NDArray backend is used.
        Passing either "text" or "sqlite" is taken as a shortcut to set
        up the corresponding backend (with "mcmc" used as the base
        name).
    tune : int
        Number of iterations to tune, if applicable (defaults to 500).
        These samples will be drawn in addition to samples and discarded
        unless discard_tuned_samples is set to True.
    step_kwargs : dict
        Options for step methods. Keys are the lower case names of
        the step method, values are dicts of keyword arguments.
        You can find a full list of arguments in the docstring of
        the step methods. If you want to pass arguments only to nuts,
        you can use `nuts_kwargs`.
    progressbar : bool
        Whether or not to display a progress bar in the command line. The
        bar shows the percentage of completion, the sampling speed in
        samples per second (SPS), and the estimated remaining time until
        completion ("expected time of arrival"; ETA).
    model : Model (optional if in `with` context)
    random_seed : int or list of ints
        A list is accepted if more if `njobs` is greater than one.
    live_plot : bool
        Flag for live plotting the trace while sampling
    live_plot_kwargs : dict
        Options for traceplot. Example: live_plot_kwargs={'varnames': ['x']}
    discard_tuned_samples : bool
        Whether to discard posterior samples of the tune interval.

    Returns
    -------
    trace : pymc3.backends.base.MultiTrace
        A `MultiTrace` object that contains the samples.
    """

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
