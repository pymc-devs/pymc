from collections import defaultdict

from joblib import Parallel, delayed
from numpy.random import randint, seed
import numpy as np

import pymc3 as pm
from .backends.base import merge_traces, BaseTrace, MultiTrace
from .backends.ndarray import NDArray
from .model import modelcontext, Point
from .step_methods import (NUTS, HamiltonianMC, Metropolis, BinaryMetropolis,
                           BinaryGibbsMetropolis, CategoricalGibbsMetropolis,
                           Slice, CompoundStep)
from .plots.traceplot import traceplot
from .util import is_transformed_name, get_untransformed_name
from tqdm import tqdm

import sys
sys.setrecursionlimit(10000)

__all__ = ['sample', 'iter_sample', 'sample_ppc', 'init_nuts']

STEP_METHODS = (NUTS, HamiltonianMC, Metropolis, BinaryMetropolis,
                BinaryGibbsMetropolis, Slice, CategoricalGibbsMetropolis)


def assign_step_methods(model, step=None, methods=STEP_METHODS,
                        step_kwargs=None):
    """Assign model variables to appropriate step methods.

    Passing a specified model will auto-assign its constituent stochastic
    variables to step methods based on the characteristics of the variables.
    This function is intended to be called automatically from `sample()`, but
    may be called manually. Each step method passed should have a
    `competence()` method that returns an ordinal competence value
    corresponding to the variable passed to it. This value quantifies the
    appropriateness of the step method for sampling the variable.

    Parameters
    ----------
    model : Model object
        A fully-specified model object
    step : step function or vector of step functions
        One or more step functions that have been assigned to some subset of
        the model's parameters. Defaults to None (no assigned variables).
    methods : vector of step method classes
        The set of step methods from which the function may choose. Defaults
        to the main step methods provided by PyMC3.
    step_kwargs : dict
        Parameters for the samplers. Keys are the lower case names of
        the step method, values a dict of arguments.

    Returns
    -------
    methods : list
        List of step methods associated with the model's variables.
    """
    if step_kwargs is None:
        step_kwargs = {}
    steps = []
    assigned_vars = set()
    if step is not None:
        try:
            steps += list(step)
        except TypeError:
            steps.append(step)
        for step in steps:
            try:
                assigned_vars = assigned_vars.union(set(step.vars))
            except AttributeError:
                for method in step.methods:
                    assigned_vars = assigned_vars.union(set(method.vars))

    # Use competence classmethods to select step methods for remaining
    # variables
    selected_steps = defaultdict(list)
    for var in model.free_RVs:
        if var not in assigned_vars:
            selected = max(methods, key=lambda method, var=var: method._competence(var))
            pm._log.info('Assigned {0} to {1}'.format(selected.__name__, var))
            selected_steps[selected].append(var)

    # Instantiate all selected step methods
    used_keys = set()
    for step_class, vars in selected_steps.items():
        if len(vars) == 0:
            continue
        args = step_kwargs.get(step_class.name, {})
        used_keys.add(step_class.name)
        step = step_class(vars=vars, **args)
        steps.append(step)

    unused_args = set(step_kwargs).difference(used_keys)
    if unused_args:
        raise ValueError('Unused step method arguments: %s' % unused_args)

    if len(steps) == 1:
        steps = steps[0]

    return steps


def sample(draws=500, step=None, init='auto', n_init=200000, start=None,
           trace=None, chain=0, njobs=1, tune=500, nuts_kwargs=None,
           step_kwargs=None, progressbar=True, model=None, random_seed=-1,
           live_plot=False, discard_tuned_samples=True, live_plot_kwargs=None,
           **kwargs):
    """Draw samples from the posterior using the given step methods.

    Multiple step methods are supported via compound step methods.

    Parameters
    ----------
    draws : int
        The number of samples to draw. Defaults to 500. The number of tuned
        samples are discarded by default. See discard_tuned_samples.
    step : function or iterable of functions
        A step function or collection of functions. If there are variables
        without a step methods, step methods for those variables will
        be assigned automatically.
    init : str {'ADVI', 'ADVI_MAP', 'MAP', 'NUTS', 'auto', None}
        Initialization method to use. Only works for auto-assigned step methods.

        * ADVI: Run ADVI to estimate starting points and diagonal covariance
          matrix. If njobs > 1 it will sample starting points from the estimated
          posterior, otherwise it will use the estimated posterior mean.
        * ADVI_MAP: Initialize ADVI with MAP and use MAP as starting point.
        * MAP: Use the MAP as starting point.
        * NUTS: Run NUTS to estimate starting points and covariance matrix. If
          njobs > 1 it will sample starting points from the estimated posterior,
          otherwise it will use the estimated posterior mean.
        * auto : Auto-initialize, if possible. Currently only works when NUTS
          is auto-assigned as step method (default).
        * None: Do not initialize.
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
    chain : int
        Chain number used to store sample in backend. If `njobs` is
        greater than one, chain numbers will start here.
    njobs : int
        Number of parallel jobs to start. If None, set to number of cpus
        in the system - 2.
    tune : int
        Number of iterations to tune, if applicable (defaults to 500).
        These samples will be drawn in addition to samples and discarded
        unless discard_tuned_samples is set to True.
    nuts_kwargs : dict
        Options for the NUTS sampler. See the docstring of NUTS
        for a complete list of options. Common options are

        * target_accept: float in [0, 1]. The step size is tuned such
          that we approximate this acceptance rate. Higher values like 0.9
          or 0.95 often work better for problematic posteriors.
        * max_treedepth: The maximum depth of the trajectory tree.
        * step_scale: float, default 0.25
          The initial guess for the step size scaled down by `1/n**(1/4)`.

        If you want to pass options to other step methods, please use
        `step_kwargs`.
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

    Examples
    --------
    .. code:: ipython

        >>> import pymc3 as pm
        ... n = 100
        ... h = 61
        ... alpha = 2
        ... beta = 2

    .. code:: ipython

        >>> with pm.Model() as model: # context management
        ...     p = pm.Beta('p', alpha=alpha, beta=beta)
        ...     y = pm.Binomial('y', n=n, p=p, observed=h)
        ...     trace = pm.sample(2000, tune=1000, njobs=4)
        >>> pm.df_summary(trace)
               mean        sd  mc_error   hpd_2.5  hpd_97.5
        p  0.604625  0.047086   0.00078  0.510498  0.694774
    """
    model = modelcontext(model)

    draws += tune

    if init is not None:
        init = init.lower()

    if nuts_kwargs is not None:
        if step_kwargs is not None:
            raise ValueError("Specify only one of step_kwargs and nuts_kwargs")
        step_kwargs = {'nuts': nuts_kwargs}

    if model.ndim == 0:
        raise ValueError('The model does not contain any free variables.')

    if step is None and init is not None and pm.model.all_continuous(model.vars):
        # By default, use NUTS sampler
        pm._log.info('Auto-assigning NUTS sampler...')
        args = step_kwargs if step_kwargs is not None else {}
        args = args.get('nuts', {})
        if init == 'auto':
            init = 'ADVI'
        start_, step = init_nuts(init=init, njobs=njobs, n_init=n_init,
                                 model=model, random_seed=random_seed,
                                 progressbar=progressbar, **args)
        if start is None:
            start = start_
    else:
        step = assign_step_methods(model, step, step_kwargs=step_kwargs)

    if njobs is None:
        import multiprocessing as mp
        njobs = max(mp.cpu_count() - 2, 1)

    sample_args = {'draws': draws,
                   'step': step,
                   'start': start,
                   'trace': trace,
                   'chain': chain,
                   'tune': tune,
                   'progressbar': progressbar,
                   'model': model,
                   'random_seed': random_seed,
                   'live_plot': live_plot,
                   'live_plot_kwargs': live_plot_kwargs,
                   }

    sample_args.update(kwargs)

    if njobs > 1:
        sample_func = _mp_sample
        sample_args['njobs'] = njobs
    else:
        sample_func = _sample

    discard = tune if discard_tuned_samples else 0

    return sample_func(**sample_args)[discard:]


def _sample(draws, step=None, start=None, trace=None, chain=0, tune=None,
            progressbar=True, model=None, random_seed=-1, live_plot=False,
            live_plot_kwargs=None, **kwargs):
    skip_first = kwargs.get('skip_first', 0)
    refresh_every = kwargs.get('refresh_every', 100)

    sampling = _iter_sample(draws, step, start, trace, chain,
                            tune, model, random_seed)
    if progressbar:
        sampling = tqdm(sampling, total=draws)
    try:
        strace = None
        for it, strace in enumerate(sampling):
            if live_plot:
                if live_plot_kwargs is None:
                    live_plot_kwargs = {}
                if it >= skip_first:
                    trace = MultiTrace([strace])
                    if it == skip_first:
                        ax = traceplot(trace, live_plot=False, **live_plot_kwargs)
                    elif (it - skip_first) % refresh_every == 0 or it == draws - 1:
                        traceplot(trace, ax=ax, live_plot=True, **live_plot_kwargs)
    except KeyboardInterrupt:
        pass
    finally:
        if progressbar:
            sampling.close()
    result = [] if strace is None else [strace]
    return MultiTrace(result)


def iter_sample(draws, step, start=None, trace=None, chain=0, tune=None,
                model=None, random_seed=-1):
    """Generator that returns a trace on each iteration using the given
    step method.  Multiple step methods supported via compound step
    method returns the amount of time taken.

    Parameters
    ----------
    draws : int
        The number of samples to draw
    step : function
        Step function
    start : dict
        Starting point in parameter space (or partial point)
        Defaults to trace.point(-1)) if there is a trace provided and
        model.test_point if not (defaults to empty dict)
    trace : backend, list, or MultiTrace
        This should be a backend instance, a list of variables to track,
        or a MultiTrace object with past values. If a MultiTrace object
        is given, it must contain samples for the chain number `chain`.
        If None or a list of variables, the NDArray backend is used.
    chain : int
        Chain number used to store sample in backend. If `njobs` is
        greater than one, chain numbers will start here.
    tune : int
        Number of iterations to tune, if applicable (defaults to None)
    model : Model (optional if in `with` context)
    random_seed : int or list of ints
        A list is accepted if more if `njobs` is greater than one.

    Example
    -------
    ::

        for trace in iter_sample(500, step):
            ...
    """
    sampling = _iter_sample(draws, step, start, trace, chain, tune,
                            model, random_seed)
    for i, strace in enumerate(sampling):
        yield MultiTrace([strace[:i + 1]])


def _iter_sample(draws, step, start=None, trace=None, chain=0, tune=None,
                 model=None, random_seed=-1):
    model = modelcontext(model)
    draws = int(draws)
    if random_seed != -1:
        seed(random_seed)
    if draws < 1:
        raise ValueError('Argument `draws` should be above 0.')

    if start is None:
        start = {}

    strace = _choose_backend(trace, chain, model=model)

    if len(strace) > 0:
        _update_start_vals(start, strace.point(-1), model)
    else:
        _update_start_vals(start, model.test_point, model)

    try:
        step = CompoundStep(step)
    except TypeError:
        pass

    point = Point(start, model=model)

    if step.generates_stats and strace.supports_sampler_stats:
        strace.setup(draws, chain, step.stats_dtypes)
    else:
        strace.setup(draws, chain)

    try:
        for i in range(draws):
            if i == tune:
                step = stop_tuning(step)
            if step.generates_stats:
                point, states = step.step(point)
                if strace.supports_sampler_stats:
                    strace.record(point, states)
                else:
                    strace.record(point)
            else:
                point = step.step(point)
                strace.record(point)
            yield strace
    except KeyboardInterrupt:
        strace.close()
        if hasattr(step, 'report'):
            step.report._finalize(strace)
        raise
    except BaseException:
        strace.close()
        raise
    else:
        strace.close()
        if hasattr(step, 'report'):
            step.report._finalize(strace)


def _choose_backend(trace, chain, shortcuts=None, **kwds):
    if isinstance(trace, BaseTrace):
        return trace
    if isinstance(trace, MultiTrace):
        return trace._straces[chain]
    if trace is None:
        return NDArray(**kwds)

    if shortcuts is None:
        shortcuts = pm.backends._shortcuts

    try:
        backend = shortcuts[trace]['backend']
        name = shortcuts[trace]['name']
        return backend(name, **kwds)
    except TypeError:
        return NDArray(vars=trace, **kwds)
    except KeyError:
        raise ValueError('Argument `trace` is invalid.')


def _make_parallel(arg, njobs):
    if not np.shape(arg):
        return [arg] * njobs
    return arg


def _parallel_random_seed(random_seed, njobs):
    if random_seed == -1 and njobs > 1:
        max_int = np.iinfo(np.int32).max
        return [randint(max_int) for _ in range(njobs)]
    else:
        return _make_parallel(random_seed, njobs)


def _mp_sample(**kwargs):
    njobs = kwargs.pop('njobs')
    chain = kwargs.pop('chain')
    random_seed = kwargs.pop('random_seed')
    start = kwargs.pop('start')

    rseed = _parallel_random_seed(random_seed, njobs)
    start_vals = _make_parallel(start, njobs)

    chains = list(range(chain, chain + njobs))
    pbars = [kwargs.pop('progressbar')] + [False] * (njobs - 1)
    traces = Parallel(n_jobs=njobs)(delayed(_sample)(chain=chains[i],
                                                     progressbar=pbars[i],
                                                     random_seed=rseed[i],
                                                     start=start_vals[i],
                                                     **kwargs) for i in range(njobs))
    return merge_traces(traces)


def stop_tuning(step):
    """ stop tuning the current step method """

    if hasattr(step, 'tune'):
        step.tune = False

    elif hasattr(step, 'methods'):
        step.methods = [stop_tuning(s) for s in step.methods]

    return step

def _update_start_vals(a, b, model):
    """Update a with b, without overwriting existing keys. Values specified for
    transformed variables on the original scale are also transformed and inserted.
    """
    if model is not None:
        for free_RV in model.free_RVs:
            tname = free_RV.name
            for name in a:
                if is_transformed_name(tname) and get_untransformed_name(tname) == name:
                    transform_func = [d.transformation for d in model.deterministics if d.name == name]
                    if transform_func:
                        b[tname] = transform_func[0].forward_val(a[name], point=b).eval()

    a.update({k: v for k, v in b.items() if k not in a})

def sample_ppc(trace, samples=None, model=None, vars=None, size=None,
               random_seed=None, progressbar=True):
    """Generate posterior predictive samples from a model given a trace.

    Parameters
    ----------
    trace : backend, list, or MultiTrace
        Trace generated from MCMC sampling
    samples : int
        Number of posterior predictive samples to generate. Defaults to the
        length of `trace`
    model : Model (optional if in `with` context)
        Model used to generate `trace`
    vars : iterable
        Variables for which to compute the posterior predictive samples.
        Defaults to `model.observed_RVs`.
    size : int
        The number of random draws from the distribution specified by the
        parameters in each sample of the trace.

    Returns
    -------
    samples : dict
        Dictionary with the variables as keys. The values corresponding
        to the posterior predictive samples.
    """
    if samples is None:
        samples = len(trace)

    if model is None:
        model = modelcontext(model)

    if vars is None:
        vars = model.observed_RVs

    seed(random_seed)

    if progressbar:
        indices = tqdm(randint(0, len(trace), samples), total=samples)
    else:
        indices = randint(0, len(trace), samples)

    try:
        ppc = defaultdict(list)
        for idx in indices:
            param = trace[idx]
            for var in vars:
                vals = var.distribution.random(point=param, size=size)
                ppc[var.name].append(vals)
    finally:
        if progressbar:
            indices.close()

    return {k: np.asarray(v) for k, v in ppc.items()}


def init_nuts(init='ADVI', njobs=1, n_init=500000, model=None,
              random_seed=-1, progressbar=True, **kwargs):
    """Initialize and sample from posterior of a continuous model.

    This is a convenience function. NUTS convergence and sampling speed is extremely
    dependent on the choice of mass/scaling matrix. In our experience, using ADVI
    to estimate a diagonal covariance matrix and using this as the scaling matrix
    produces robust results over a wide class of continuous models.

    Parameters
    ----------
    init : str {'ADVI', 'ADVI_MAP', 'MAP', 'NUTS'}
        Initialization method to use.
        * ADVI : Run ADVI to estimate posterior mean and diagonal covariance matrix.
        * ADVI_MAP: Initialize ADVI with MAP and use MAP as starting point.
        * MAP : Use the MAP as starting point.
        * NUTS : Run NUTS and estimate posterior mean and covariance matrix.
    njobs : int
        Number of parallel jobs to start.
    n_init : int
        Number of iterations of initializer
        If 'ADVI', number of iterations, if 'metropolis', number of draws.
    model : Model (optional if in `with` context)
    progressbar : bool
        Whether or not to display a progressbar for advi sampling.
    **kwargs : keyword arguments
        Extra keyword arguments are forwarded to pymc3.NUTS.

    Returns
    -------
    start : pymc3.model.Point
        Starting point for sampler
    nuts_sampler : pymc3.step_methods.NUTS
        Instantiated and initialized NUTS sampler object
    """

    model = pm.modelcontext(model)

    pm._log.info('Initializing NUTS using {}...'.format(init))

    random_seed = int(np.atleast_1d(random_seed)[0])

    if init is not None:
        init = init.lower()
    cb = [
        pm.callbacks.CheckParametersConvergence(tolerance=1e-2, diff='absolute'),
        pm.callbacks.CheckParametersConvergence(tolerance=1e-2, diff='relative'),
    ]
    if init == 'advi':
        approx = pm.fit(
            random_seed=random_seed,
            n=n_init, method='advi', model=model,
            callbacks=cb,
            progressbar=progressbar,
            obj_optimizer=pm.adagrad_window
        )  # type: pm.MeanField
        start = approx.sample(draws=njobs)
        stds = approx.gbij.rmap(approx.std.eval())
        cov = model.dict_to_array(stds) ** 2
        if njobs == 1:
            start = start[0]
    elif init == 'advi_map':
        start = pm.find_MAP()
        approx = pm.MeanField(model=model, start=start)
        pm.fit(
            random_seed=random_seed,
            n=n_init, method=pm.ADVI.from_mean_field(approx),
            callbacks=cb,
            progressbar=progressbar,
            obj_optimizer=pm.adagrad_window
        )
        start = approx.sample(draws=njobs)
        stds = approx.gbij.rmap(approx.std.eval())
        cov = model.dict_to_array(stds) ** 2
        if njobs == 1:
            start = start[0]
    elif init == 'map':
        start = pm.find_MAP()
        cov = pm.find_hessian(point=start)
    elif init == 'nuts':
        init_trace = pm.sample(draws=n_init, step=pm.NUTS(),
                               tune=n_init // 2,
                               random_seed=random_seed)
        cov = np.atleast_1d(pm.trace_cov(init_trace))
        start = np.random.choice(init_trace, njobs)
        if njobs == 1:
            start = start[0]
    else:
        raise NotImplementedError('Initializer {} is not supported.'.format(init))

    step = pm.NUTS(scaling=cov, is_cov=True, **kwargs)

    return start, step
