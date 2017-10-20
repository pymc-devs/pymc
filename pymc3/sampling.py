from collections import defaultdict
import pickle

from joblib import Parallel, delayed
import numpy as np
import theano.gradient as tg

import pymc3 as pm
from .backends.base import BaseTrace, MultiTrace
from .backends.ndarray import NDArray
from .model import modelcontext, Point
from .step_methods import (NUTS, HamiltonianMC, SGFS, Metropolis, BinaryMetropolis,
                           BinaryGibbsMetropolis, CategoricalGibbsMetropolis,
                           Slice, CompoundStep)
from .util import update_start_vals
from .vartypes import discrete_types
from pymc3.step_methods.hmc import quadpotential
from tqdm import tqdm

import sys
sys.setrecursionlimit(10000)

__all__ = ['sample', 'iter_sample', 'sample_ppc', 'sample_ppc_w', 'init_nuts']

STEP_METHODS = (NUTS, HamiltonianMC, SGFS, Metropolis, BinaryMetropolis,
                BinaryGibbsMetropolis, Slice, CategoricalGibbsMetropolis)


def instantiate_steppers(model, steps, selected_steps, step_kwargs=None):
    """Instantiates steppers assigned to the model variables.

    This function is intended to be called automatically from `sample()`, but
    may be called manually.

    Parameters
    ----------
    model : Model object
        A fully-specified model object
    step : step function or vector of step functions
        One or more step functions that have been assigned to some subset of
        the model's parameters. Defaults to None (no assigned variables).
    selected_steps: dictionary of step methods and variables
        The step methods and the variables that have were assigned to them.
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
            # determine if a gradient can be computed
            has_gradient = var.dtype not in discrete_types
            if has_gradient:
                try:
                    tg.grad(model.logpt, var)
                except (AttributeError,
                        NotImplementedError,
                        tg.NullTypeGradError):
                    has_gradient = False
            # select the best method
            selected = max(methods, key=lambda method,
                           var=var, has_gradient=has_gradient:
                           method._competence(var, has_gradient))
            pm._log.info('Assigned {0} to {1}'.format(selected.__name__, var))
            selected_steps[selected].append(var)

    return instantiate_steppers(model, steps, selected_steps, step_kwargs)


def _cpu_count():
    """Try to guess the number of CPUs in the system.

    We use the number provided by psutil if that is installed.
    If not, we use the number provided by multiprocessing, but assume
    that half of the cpus are only hardware threads and ignore those.
    """
    try:
        import psutil
        cpus = psutil.cpu_count(False)
    except ImportError:
        import multiprocessing
        try:
            cpus = multiprocessing.cpu_count() // 2
        except NotImplementedError:
            cpus = 1
    if cpus is None:
        cpus = 1
    return cpus


def sample(draws=500, step=None, init='auto', n_init=200000, start=None,
           trace=None, chain=0, chains=None, njobs=None, tune=500,
           nuts_kwargs=None, step_kwargs=None, progressbar=True, model=None,
           random_seed=None, live_plot=False, discard_tuned_samples=True,
           live_plot_kwargs=None, **kwargs):
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
    init : str
        Initialization method to use for auto-assigned NUTS samplers.

        * auto : Choose a default initialization method automatically.
          Currently, this is `'jitter+adapt_diag'`, but this can change in
          the future. If you depend on the exact behaviour, choose an
          initialization method explicitly.
        * adapt_diag : Start with a identity mass matrix and then adapt
          a diagonal based on the variance of the tuning samples. All
          chains use the test value (usually the prior mean) as starting
          point.
        * jitter+adapt_diag : Same as `adapt_diag`, but add uniform jitter
          in [-1, 1] to the starting point in each chain.
        * advi+adapt_diag : Run ADVI and then adapt the resulting diagonal
          mass matrix based on the sample variance of the tuning samples.
        * advi+adapt_diag_grad : Run ADVI and then adapt the resulting
          diagonal mass matrix based on the variance of the gradients
          during tuning. This is **experimental** and might be removed
          in a future release.
        * advi : Run ADVI to estimate posterior mean and diagonal mass
          matrix.
        * advi_map: Initialize ADVI with MAP and use MAP as starting point.
        * map : Use the MAP as starting point. This is discouraged.
        * nuts : Run NUTS and estimate posterior mean and mass matrix from
          the trace.
    n_init : int
        Number of iterations of initializer
        If 'ADVI', number of iterations, if 'nuts', number of draws.
    start : dict, or array of dict
        Starting point in parameter space (or partial point)
        Defaults to trace.point(-1)) if there is a trace provided and
        model.test_point if not (defaults to empty dict). Initialization
        methods for NUTS (see `init` keyword) can overwrite the default.
    trace : backend, list, or MultiTrace
        This should be a backend instance, a list of variables to track,
        or a MultiTrace object with past values. If a MultiTrace object
        is given, it must contain samples for the chain number `chain`.
        If None or a list of variables, the NDArray backend is used.
        Passing either "text" or "sqlite" is taken as a shortcut to set
        up the corresponding backend (with "mcmc" used as the base
        name).
    chain : int
        Chain number used to store sample in backend. If `chains` is
        greater than one, chain numbers will start here.
    chains : int
        The number of chains to sample. Running independent chains
        is important for some convergence statistics and can also
        reveal multiple modes in the posterior.
    njobs : int
        The number of chains to run in parallel. If None, set to the
        number of CPUs in the system, but at most 4. Keep in mind that
        some chains might themselves be multithreaded via openmp or
        BLAS. In those cases it might be faster to set this to one.
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
        A list is accepted if `njobs` is greater than one.
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

    if njobs is None:
        njobs = min(4, _cpu_count())
    if chains is None:
        chains = max(2, njobs)
    if isinstance(start, dict):
        start = [start] * chains
    if random_seed == -1:
        random_seed = None
    if chains == 1 and isinstance(random_seed, int):
        random_seed = [random_seed]
    if random_seed is None or isinstance(random_seed, int):
        if random_seed is not None:
            np.random.seed(random_seed)
        random_seed = [np.random.randint(2 ** 30) for _ in range(chains)]
    if not isinstance(random_seed, list):
        raise TypeError('Invalid value for `random_seed`. Must be list or int')

    if start is not None:
        for start_vals in start:
            _check_start_shape(model, start_vals)

    draws += tune

    if nuts_kwargs is not None:
        if step_kwargs is not None:
            raise ValueError("Specify only one of step_kwargs and nuts_kwargs")
        step_kwargs = {'nuts': nuts_kwargs}

    if model.ndim == 0:
        raise ValueError('The model does not contain any free variables.')

    if step is None and init is not None and pm.model.all_continuous(model.vars):
        try:
            # By default, try to use NUTS
            pm._log.info('Auto-assigning NUTS sampler...')
            args = step_kwargs if step_kwargs is not None else {}
            args = args.get('nuts', {})
            start_, step = init_nuts(init=init, chains=chains, n_init=n_init,
                                     model=model, random_seed=random_seed,
                                     progressbar=progressbar, **args)
            if start is None:
                start = start_
        except (AttributeError, NotImplementedError, tg.NullTypeGradError):
            # gradient computation failed
            pm._log.info("Initializing NUTS failed. "
                         "Falling back to elementwise auto-assignment.")
            step = assign_step_methods(model, step, step_kwargs=step_kwargs)
    else:
        step = assign_step_methods(model, step, step_kwargs=step_kwargs)

    if start is None:
        start = [None] * chains
    if isinstance(start, dict):
        start = [start] * chains

    sample_args = {
        'draws': draws,
        'step': step,
        'start': start,
        'trace': trace,
        'chain': chain,
        'chains': chains,
        'tune': tune,
        'progressbar': progressbar,
        'model': model,
        'random_seed': random_seed,
        'live_plot': live_plot,
        'live_plot_kwargs': live_plot_kwargs,
        'njobs': njobs,
    }

    sample_args.update(kwargs)

    parallel = njobs > 1 and chains > 1
    if parallel:
        try:
            trace = _mp_sample(**sample_args)
        except pickle.PickleError:
            pm._log.warn("Could not pickle model, sampling sequentially.")
            parallel = False
    if not parallel:
        trace = _sample_many(**sample_args)

    discard = tune if discard_tuned_samples else 0
    return trace[discard:]


def _check_start_shape(model, start):
    if not isinstance(start, dict):
        raise TypeError("start argument must be a dict "
                        "or an array-like of dicts")
    e = ''
    for var in model.vars:
        if var.name in start.keys():
            var_shape = var.shape.tag.test_value
            start_var_shape = np.shape(start[var.name])
            if start_var_shape:
                if not np.array_equal(var_shape, start_var_shape):
                    e += "\nExpected shape {} for var '{}', got: {}".format(
                        tuple(var_shape), var.name, start_var_shape
                    )
            # if start var has no shape
            else:
                # if model var has a specified shape
                if var_shape:
                    e += "\nExpected shape {} for var " \
                         "'{}', got scalar {}".format(
                             tuple(var_shape), var.name, start[var.name]
                         )

    if e != '':
        raise ValueError("Bad shape for start argument:{}".format(e))


def _sample_many(draws, chain, chains, start, random_seed, **kwargs):
    traces = []
    for i in range(chains):
        trace = _sample(draws=draws, chain=chain + i, start=start[i],
                        random_seed=random_seed[i], **kwargs)
        if trace is None:
            if len(traces) == 0:
                raise ValueError('Sampling stopped before a sample was created.')
            else:
                break
        elif len(trace) < draws:
            if len(traces) == 0:
                traces.append(trace)
            break
        else:
            traces.append(trace)
    return MultiTrace(traces)


def _sample(chain, progressbar, random_seed, start, draws=None, step=None,
            trace=None, tune=None, model=None, live_plot=False,
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
                        ax = pm.plots.traceplot(trace, live_plot=False, **live_plot_kwargs)
                    elif (it - skip_first) % refresh_every == 0 or it == draws - 1:
                        pm.plots.traceplot(trace, ax=ax, live_plot=True, **live_plot_kwargs)
    except KeyboardInterrupt:
        pass
    finally:
        if progressbar:
            sampling.close()
    return strace


def iter_sample(draws, step, start=None, trace=None, chain=0, tune=None,
                model=None, random_seed=None):
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
                 model=None, random_seed=None):
    model = modelcontext(model)
    draws = int(draws)
    if random_seed is not None:
        np.random.seed(random_seed)
    if draws < 1:
        raise ValueError('Argument `draws` should be above 0.')

    if start is None:
        start = {}

    strace = _choose_backend(trace, chain, model=model)

    if len(strace) > 0:
        update_start_vals(start, strace.point(-1), model)
    else:
        update_start_vals(start, model.test_point, model)

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


def _mp_sample(**kwargs):
    njobs = kwargs.pop('njobs')
    chain = kwargs.pop('chain')
    rseed = kwargs.pop('random_seed')
    start = kwargs.pop('start')
    chains = kwargs.pop('chains')

    chain_nums = list(range(chain, chain + chains))
    pbars = [kwargs.pop('progressbar')] + [False] * (chains - 1)
    jobs = (delayed(_sample)(*args, **kwargs)
            for args in zip(chain_nums, pbars, rseed, start))
    traces = Parallel(n_jobs=njobs)(jobs)
    return MultiTrace(traces)


def stop_tuning(step):
    """ stop tuning the current step method """

    if hasattr(step, 'tune'):
        step.tune = False

    elif hasattr(step, 'methods'):
        step.methods = [stop_tuning(s) for s in step.methods]

    return step


def sample_ppc(trace, samples=None, model=None, vars=None, size=None,
               random_seed=None, progressbar=True):
    """Generate posterior predictive samples from a model given a trace.

    Parameters
    ----------
    trace : backend, list, or MultiTrace
        Trace generated from MCMC sampling.
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
    random_seed : int
        Seed for the random number generator.
    progressbar : bool
        Whether or not to display a progress bar in the command line. The
        bar shows the percentage of completion, the sampling speed in
        samples per second (SPS), and the estimated remaining time until
        completion ("expected time of arrival"; ETA).

    Returns
    -------
    samples : dict
        Dictionary with the variables as keys. The values corresponding to the
        posterior predictive samples.
    """
    if samples is None:
        samples = len(trace)

    model = modelcontext(model)

    if vars is None:
        vars = model.observed_RVs

    np.random.seed(random_seed)

    indices = np.random.randint(0, len(trace), samples)
    if progressbar:
        indices = tqdm(indices, total=samples)

    try:
        ppc = defaultdict(list)
        for idx in indices:
            param = trace[idx]
            for var in vars:
                ppc[var.name].append(var.distribution.random(point=param,
                                                             size=size))

    except KeyboardInterrupt:
        pass

    finally:
        if progressbar:
            indices.close()

    return {k: np.asarray(v) for k, v in ppc.items()}


def sample_ppc_w(traces, samples=None, models=None, weights=None,
                 random_seed=None, progressbar=True):
    """Generate weighted posterior predictive samples from a list of models and
    a list of traces according to a set of weights.

    Parameters
    ----------
    traces : list
        List of traces generated from MCMC sampling. The number of traces should
        be equal to the number of weights.
    samples : int
        Number of posterior predictive samples to generate. Defaults to the
        length of the shorter trace in traces.
    models : list
        List of models used to generate the list of traces. The number of models
        should be equal to the number of weights and the number of observed RVs
        should be the same for all models.
        By default a single model will be inferred from `with` context, in this
        case results will only be meaningful if all models share the same
        distributions for the observed RVs.
    weights: array-like
        Individual weights for each trace. Default, same weight for each model.
    random_seed : int
        Seed for the random number generator.
    progressbar : bool
        Whether or not to display a progress bar in the command line. The
        bar shows the percentage of completion, the sampling speed in
        samples per second (SPS), and the estimated remaining time until
        completion ("expected time of arrival"; ETA).

    Returns
    -------
    samples : dict
        Dictionary with the variables as keys. The values corresponding to the
        posterior predictive samples from the weighted models.
    """
    np.random.seed(random_seed)

    if models is None:
        models = [modelcontext(models)] * len(traces)

    if weights is None:
        weights = [1] * len(traces)

    if len(traces) != len(weights):
        raise ValueError('The number of traces and weights should be the same')

    if len(models) != len(weights):
        raise ValueError('The number of models and weights should be the same')

    lenght_morv = len(models[0].observed_RVs)
    if not all(len(i.observed_RVs) == lenght_morv for i in models):
        raise ValueError(
            'The number of observed RVs should be the same for all models')

    weights = np.asarray(weights)
    p = weights / np.sum(weights)

    min_tr = min([len(i) for i in traces])

    n = (min_tr * p).astype('int')
    # ensure n sum up to min_tr
    idx = np.argmax(n)
    n[idx] = n[idx] + min_tr - np.sum(n)
    trace = np.concatenate([np.random.choice(traces[i], j)
                            for i, j in enumerate(n)])

    obs = [x for m in models for x in m.observed_RVs]
    variables = np.repeat(obs, n)

    lenghts = list(set([np.shape(np.atleast_1d(o.distribution.default())) for o in obs]))

    if len(lenghts) == 1:
        size = [None for i in variables]
    elif len(lenghts) > 2:
        raise ValueError('Observed variables could not be broadcast together')
    else:
        size = []
        x = np.zeros(shape=lenghts[0])
        y = np.zeros(shape=lenghts[1])
        b = np.broadcast(x, y)
        for var in variables:
            l = np.shape(np.atleast_1d(var.distribution.default()))
            if l != b.shape:
                size.append(b.shape)
            else:
                size.append(None)
    len_trace = len(trace)

    if samples is None:
        samples = len_trace

    indices = np.random.randint(0, len_trace, samples)

    if progressbar:
        indices = tqdm(indices, total=samples)

    try:
        ppc = defaultdict(list)
        for idx in indices:
            param = trace[idx]
            var = variables[idx]
            ppc[var.name].append(var.distribution.random(point=param,
                                                         size=size[idx]))

    except KeyboardInterrupt:
        pass

    finally:
        if progressbar:
            indices.close()

    return {k: np.asarray(v) for k, v in ppc.items()}


def init_nuts(init='auto', chains=1, n_init=500000, model=None,
              random_seed=None, progressbar=True, **kwargs):
    """Set up the mass matrix initialization for NUTS.

    NUTS convergence and sampling speed is extremely dependent on the
    choice of mass/scaling matrix. This function implements different
    methods for choosing or adapting the mass matrix.

    Parameters
    ----------
    init : str
        Initialization method to use.

        * auto : Choose a default initialization method automatically.
          Currently, this is `'jitter+adapt_diag'`, but this can change in
          the future. If you depend on the exact behaviour, choose an
          initialization method explicitly.
        * adapt_diag : Start with a identity mass matrix and then adapt
          a diagonal based on the variance of the tuning samples. All
          chains use the test value (usually the prior mean) as starting
          point.
        * jitter+adapt_diag : Same as `adapt_diag`, but add uniform jitter
          in [-1, 1] to the starting point in each chain.
        * advi+adapt_diag : Run ADVI and then adapt the resulting diagonal
          mass matrix based on the sample variance of the tuning samples.
        * advi+adapt_diag_grad : Run ADVI and then adapt the resulting
          diagonal mass matrix based on the variance of the gradients
          during tuning. This is **experimental** and might be removed
          in a future release.
        * advi : Run ADVI to estimate posterior mean and diagonal mass
          matrix.
        * advi_map: Initialize ADVI with MAP and use MAP as starting point.
        * map : Use the MAP as starting point. This is discouraged.
        * nuts : Run NUTS and estimate posterior mean and mass matrix from
          the trace.
    chains : int
        Number of jobs to start.
    n_init : int
        Number of iterations of initializer
        If 'ADVI', number of iterations, if 'nuts', number of draws.
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

    vars = kwargs.get('vars', model.vars)
    if set(vars) != set(model.vars):
        raise ValueError('Must use init_nuts on all variables of a model.')
    if not pm.model.all_continuous(vars):
        raise ValueError('init_nuts can only be used for models with only '
                         'continuous variables.')

    if not isinstance(init, str):
        raise TypeError('init must be a string.')

    if init is not None:
        init = init.lower()

    if init == 'auto':
        init = 'jitter+adapt_diag'

    pm._log.info('Initializing NUTS using {}...'.format(init))

    if random_seed is not None:
        random_seed = int(np.atleast_1d(random_seed)[0])
        np.random.seed(random_seed)

    cb = [
        pm.callbacks.CheckParametersConvergence(
            tolerance=1e-2, diff='absolute'),
        pm.callbacks.CheckParametersConvergence(
            tolerance=1e-2, diff='relative'),
    ]

    if init == 'adapt_diag':
        start = [model.test_point] * chains
        mean = np.mean([model.dict_to_array(vals) for vals in start], axis=0)
        var = np.ones_like(mean)
        potential = quadpotential.QuadPotentialDiagAdapt(
            model.ndim, mean, var, 10)
    elif init == 'jitter+adapt_diag':
        start = []
        for _ in range(chains):
            mean = {var: val.copy() for var, val in model.test_point.items()}
            for val in mean.values():
                val[...] += 2 * np.random.rand(*val.shape) - 1
            start.append(mean)
        mean = np.mean([model.dict_to_array(vals) for vals in start], axis=0)
        var = np.ones_like(mean)
        potential = quadpotential.QuadPotentialDiagAdapt(
            model.ndim, mean, var, 10)
    elif init == 'advi+adapt_diag_grad':
        approx = pm.fit(
            random_seed=random_seed,
            n=n_init, method='advi', model=model,
            callbacks=cb,
            progressbar=progressbar,
            obj_optimizer=pm.adagrad_window,
        )  # type: pm.MeanField
        start = approx.sample(draws=chains)
        start = list(start)
        stds = approx.bij.rmap(approx.std.eval())
        cov = model.dict_to_array(stds) ** 2
        mean = approx.bij.rmap(approx.mean.get_value())
        mean = model.dict_to_array(mean)
        weight = 50
        potential = quadpotential.QuadPotentialDiagAdaptGrad(
            model.ndim, mean, cov, weight)
    elif init == 'advi+adapt_diag':
        approx = pm.fit(
            random_seed=random_seed,
            n=n_init, method='advi', model=model,
            callbacks=cb,
            progressbar=progressbar,
            obj_optimizer=pm.adagrad_window,
        )  # type: pm.MeanField
        start = approx.sample(draws=chains)
        start = list(start)
        stds = approx.bij.rmap(approx.std.eval())
        cov = model.dict_to_array(stds) ** 2
        mean = approx.bij.rmap(approx.mean.get_value())
        mean = model.dict_to_array(mean)
        weight = 50
        potential = quadpotential.QuadPotentialDiagAdapt(
            model.ndim, mean, cov, weight)
    elif init == 'advi':
        approx = pm.fit(
            random_seed=random_seed,
            n=n_init, method='advi', model=model,
            callbacks=cb,
            progressbar=progressbar,
            obj_optimizer=pm.adagrad_window
        )  # type: pm.MeanField
        start = approx.sample(draws=chains)
        start = list(start)
        stds = approx.bij.rmap(approx.std.eval())
        cov = model.dict_to_array(stds) ** 2
        potential = quadpotential.QuadPotentialDiag(cov)
    elif init == 'advi_map':
        start = pm.find_MAP(include_transformed=True)
        approx = pm.MeanField(model=model, start=start)
        pm.fit(
            random_seed=random_seed,
            n=n_init, method=pm.KLqp(approx),
            callbacks=cb,
            progressbar=progressbar,
            obj_optimizer=pm.adagrad_window
        )
        start = approx.sample(draws=chains)
        start = list(start)
        stds = approx.bij.rmap(approx.std.eval())
        cov = model.dict_to_array(stds) ** 2
        potential = quadpotential.QuadPotentialDiag(cov)
    elif init == 'map':
        start = pm.find_MAP(include_transformed=True)
        cov = pm.find_hessian(point=start)
        start = [start] * chains
        potential = quadpotential.QuadPotentialFull(cov)
    elif init == 'nuts':
        init_trace = pm.sample(draws=n_init, step=pm.NUTS(),
                               tune=n_init // 2,
                               random_seed=random_seed)
        cov = np.atleast_1d(pm.trace_cov(init_trace))
        start = list(np.random.choice(init_trace, chains))
        potential = quadpotential.QuadPotentialFull(cov)
    else:
        raise NotImplementedError('Initializer {} is not supported.'.format(init))

    step = pm.NUTS(potential=potential, **kwargs)

    return start, step
