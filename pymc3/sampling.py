from collections import defaultdict

from joblib import Parallel, delayed
from numpy.random import randint, seed
import numpy as np

import pymc3 as pm
from .backends.base import merge_traces, BaseTrace, MultiTrace
from .backends.ndarray import NDArray, MultiNDArray
from .model import modelcontext, Point
from .step_methods import (NUTS, HamiltonianMC, Metropolis, BinaryMetropolis,
                           BinaryGibbsMetropolis, CategoricalGibbsMetropolis,
                           Slice, CompoundStep, ParticleStep)
from tqdm import tqdm

import sys
sys.setrecursionlimit(10000)

__all__ = ['sample', 'iter_sample', 'sample_ppc', 'init_nuts']


def assign_step_methods(model, step=None, methods=(NUTS, HamiltonianMC, Metropolis,
                                                   BinaryMetropolis, BinaryGibbsMetropolis,
                                                   Slice, CategoricalGibbsMetropolis)):
    '''
    Assign model variables to appropriate step methods. Passing a specified
    model will auto-assign its constituent stochastic variables to step methods
    based on the characteristics of the variables. This function is intended to
    be called automatically from `sample()`, but may be called manually. Each
    step method passed should have a `competence()` method that returns an
    ordinal competence value corresponding to the variable passed to it. This
    value quantifies the appropriateness of the step method for sampling the
    variable.

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

    Returns
    -------
    List of step methods associated with the model's variables.
    '''
    if isinstance(step, ParticleStep):
        return [step]

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
    steps += [step(vars=selected_steps[step]) for step in selected_steps if selected_steps[step]]

    if len(steps) == 1:
        steps = steps[0]

    return steps


def sample(draws, step=None, init='advi', n_init=200000, start=None,
           trace=None, chain=0, njobs=1, tune=None, progressbar=True,
           model=None, random_seed=-1, init_start=None):
    """
    Draw a number of samples using the given step method.
    Multiple step methods supported via compound step method
    returns the amount of time taken.

    Parameters
    ----------

    draws : int
        The number of samples to draw.
    step : function or iterable of functions
        A step function or collection of functions. If no step methods are
        specified, or are partially specified, they will be assigned
        automatically (defaults to None).
    init : str {'ADVI', 'ADVI_MAP', 'MAP', 'NUTS', None}
        Initialization method to use.
        * ADVI : Run ADVI to estimate starting points and diagonal covariance
        matrix. If njobs > 1 it will sample starting points from the estimated
        posterior, otherwise it will use the estimated posterior mean.
        * ADVI_MAP: Initialize ADVI with MAP and use MAP as starting point.
        * MAP : Use the MAP as starting point.
        * NUTS : Run NUTS to estimate starting points and covariance matrix. If
        njobs > 1 it will sample starting points from the estimated posterior,
        otherwise it will use the estimated posterior mean.
        * None : Do not initialize.
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
        Number of iterations to tune, if applicable (defaults to None)
    progressbar : bool
        Whether or not to display a progress bar in the command line. The
        bar shows the percentage of completion, the sampling speed in
        samples per second (SPS), and the estimated remaining time until
        completion ("expected time of arrival"; ETA).
    model : Model (optional if in `with` context)
    random_seed : int or list of ints
        A list is accepted if more if `njobs` is greater than one.

    Returns
    -------
    MultiTrace object with access to sampling values
    """
    model = modelcontext(model)

    if init is not None:
        init = init.lower()

    if isinstance(step, ParticleStep):
        trace = MultiNDArray(step.nparticles)

    _start = None
    if init is not None and pm.model.all_continuous(model.vars):
        if step is None:
            # By default, use NUTS sampler
            pm._log.info('Auto-assigning NUTS sampler...')
            _start, cov = do_init(init=init, njobs=njobs * step.nparticles, n_init=n_init, model=model, start=init_start,
                                  random_seed=random_seed)
            step = pm.NUTS(scaling=cov, is_cov=True)
        elif isinstance(step, ParticleStep):
            if start is None and init == 'random':
                _start = get_random_starters(njobs * step.nparticles, model)
            elif init is not None:
                _start, _ = do_init(init=init, njobs=njobs * step.nparticles, n_init=n_init, model=model,
                                        random_seed=random_seed, start=init_start)
            if trace is None:
                trace = MultiNDArray(step.nparticles)
        else:
            step = assign_step_methods(model, step)
    else:
        step = assign_step_methods(model, step)

    start = transform_start_particles(start, step.nparticles, njobs, model)
    _start = transform_start_particles(_start, step.nparticles, njobs, model)
    if start is None:
        start = [{}]
    if _start is None:
        _start = [{}]
    for pair in zip(start, _start):
        _soft_update(*pair)

    if njobs is None:
        import multiprocessing as mp
        njobs = max(mp.cpu_count() - 2, 1)
    elif njobs == 1 and isinstance(start, (list, tuple)):
        start = start[0]

    sample_args = {'draws': draws,
                   'step': step,
                   'start': start,
                   'trace': trace,
                   'chain': chain,
                   'tune': tune,
                   'progressbar': progressbar,
                   'model': model,
                   'random_seed': random_seed}

    if njobs > 1:
        sample_func = _mp_sample
        sample_args['njobs'] = njobs
    else:
        sample_func = _sample

    return sample_func(**sample_args)


def _sample(draws, step=None, start=None, trace=None, chain=0, tune=None,
            progressbar=True, model=None, random_seed=-1):
    sampling = _iter_sample(draws, step, start, trace, chain,
                            tune, model, random_seed)
    if progressbar:
        sampling = tqdm(sampling, total=draws)
    try:
        strace = None
        for strace in sampling:
            pass
    except KeyboardInterrupt:
        pass
    finally:
        if progressbar:
            sampling.close()
    if strace is not None:
        strace.close()
    result = [] if strace is None else [strace]
    return MultiTrace(result)


def iter_sample(draws, step, start=None, trace=None, chain=0, tune=None,
                model=None, random_seed=-1):
    """
    Generator that returns a trace on each iteration using the given
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

    strace = _choose_backend(trace, chain, model=model, nparticles=step.nparticles)

    if len(strace) > 0:
        _soft_update(start, strace.point(-1))
    else:
        if hasattr(step, 'nparticles'):
            _soft_update(start, {k: _make_parallel(v, step.nparticles) for k, v in model.test_point.iteritems()})
        else:
            _soft_update(start, model.test_point)

    try:
        step = CompoundStep(step)
    except TypeError:
        pass

    point = Point(start, model=model)

    if step.generates_stats and strace.supports_sampler_stats:
        strace.setup(draws, chain, step.stats_dtypes)
    else:
        strace.setup(draws, chain)
    methods = step.methods if isinstance(step, CompoundStep) else [step]
    for s in methods:
        s.expected_ndraws = draws
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
    else:
        strace.close()


def _choose_backend(trace, chain, shortcuts=None, nparticles=None, **kwds):
    if isinstance(trace, BaseTrace):
        return trace
    if isinstance(trace, MultiTrace):
        return trace._straces[chain]

    if nparticles is None:
        if trace is None:
            return MultiNDArray(nparticles=nparticles, **kwds)
        if shortcuts is None:
            shortcuts = pm.backends._particle_shortcuts
    else:
        if trace is None:
            return NDArray(**kwds)
        if shortcuts is None:
            shortcuts = pm.backends._shortcuts

    try:
        backend = shortcuts[trace]['backend']
        name = shortcuts[trace]['name']
        return backend(name, **kwds)
    except TypeError:
        if nparticles is None:
            return NDArray(vars=trace, **kwds)
        return MultiNDArray(vars=trace, **kwds)
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


def _soft_update(a, b):
    """As opposed to dict.update, don't overwrite keys if present.
    """
    a.update({k: v for k, v in b.items() if k not in a})


def sample_ppc(trace, samples=None, model=None, vars=None, size=None, random_seed=None, progressbar=True):
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
    Dictionary keyed by `vars`, where the values are the corresponding
    posterior predictive samples.
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

    ppc = defaultdict(list)
    for idx in indices:
        param = trace[idx]
        for var in vars:
            ppc[var.name].append(var.distribution.random(point=param,
                                                         size=size))

    return {k: np.asarray(v) for k, v in ppc.items()}


def do_init(init='ADVI', njobs=1, n_init=500000, model=None, random_seed=-1, randomiser=None, start=None):
    """Initialize and sample from posterior of a continuous model.
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
        **kwargs : keyword arguments
            Extra keyword arguments are forwarded to pymc3.NUTS.
        randomiser : str ('cov', 'sample', 'duplicate')
            How inits with njobs > 1 obtain samples around the single inititalized guess
            * cov : Generate a covariance array for all parameters and sample from the ensuing multivariate gaussian
            * sample : For all init methods except `map`, 'sample' will sample using advi or from the initial nuts trace
            * duplicate : Just copy the same values for each job

        Returns
        -------
        start, nuts_sampler

        start : pymc3.model.Point
            Starting point for sampler
        """
    model = pm.modelcontext(model)

    pm._log.info('Initializing using {}...'.format(init))

    random_seed = int(np.atleast_1d(random_seed)[0])

    if randomiser is None:
        if init == 'map':
            randomiser = 'cov'
        else:
            randomiser = 'sample'

    if randomiser != 'sample':
        _njobs = njobs
        njobs = 1

    if init is not None:
        init = init.lower()

    if init.startswith('advi'):
        if init == 'advi_map':
            start = pm.find_MAP(start=start)
        elif init != 'advi':
            raise NotImplemented('Initializer {} is not supported.'.format(init))
        v_params = pm.variational.advi(n=n_init, start=start, random_seed=random_seed)
        cov = np.power(model.dict_to_array(v_params.stds), 2)
        if randomiser == 'sample':
            start = pm.variational.sample_vp(v_params, njobs, progressbar=False,
                                             hide_transformed=False,
                                             random_seed=random_seed)  # multitrace
            start = [start[i] for i in range(njobs)]
    elif init == 'map':
        start = pm.find_MAP(start=start)  # dict
        cov = pm.find_hessian(point=start)
        start = [start]
        if randomiser == 'sample':
            raise NotImplementedError(
                'Randomising from MAP using "sample" is not supported. Use "random", "cov", or "duplicate".')
    elif init == 'nuts':
        init_trace = pm.sample(step=pm.NUTS(), draws=n_init,
                               random_seed=random_seed, start=start)[n_init // 2:]
        cov = np.atleast_1d(pm.trace_cov(init_trace))
        start = np.random.choice(init_trace, njobs)  # list of dicts
    else:
        raise NotImplementedError('Initializer {} is not supported.'.format(init))

    if njobs == 1:
        return start[0], cov

    if randomiser == 'cov':
        model = pm.modelcontext(model)
        order = pm.ArrayOrdering(model.vars)
        bij = pm.DictToArrayBijection(order, start[0])
        sarray = bij.map(start[0])
        start = [bij.rmap(np.random.multivariate_normal(sarray, cov)) for i in range(_njobs)]
    elif randomiser == 'duplicate':
        start = [start[0]] * _njobs
    elif randomiser != 'sample':
        raise NotImplemented('Randomiser {} is not supported.'.format(randomiser))

    return start, cov


def init_nuts(init='ADVI', njobs=1, n_init=500000, model=None,
              random_seed=-1, **kwargs):
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
    **kwargs : keyword arguments
        Extra keyword arguments are forwarded to pymc3.NUTS.

    Returns
    -------
    start, nuts_sampler

    start : pymc3.model.Point
        Starting point for sampler
    nuts_sampler : pymc3.step_methods.NUTS
        Instantiated and initialized NUTS sampler object
    """

    start, cov = do_init(init, njobs, n_init, model, random_seed)
    step = pm.NUTS(scaling=cov, is_cov=True, **kwargs)
    return start, step


def get_random_starters(nwalkers, model=None):
    model = pm.modelcontext(model)
    return {v.name: v.distribution.random(size=nwalkers) for v in model.vars}


def transform_start_particles(start, nparticles, njobs, model=None):
    """
    Transforms lists/dicts of starting values to the same format of [{...}, {...}, ...] where the list index is the job index
     and each dictionary variable has the same first dimension (nwalkers) or if they don't, duplicate those variables

     if the given start has already specified the correct number of jobs (length of list) and nparticles, it is returned unchanged
     if the given start has specified only one particle the required number of particles will be achieved by duplication
     if the given start is a
     if the given start is a list with len > 1 and njobs > 1 this will raise a TypeError
     if the given start has mismatched walkers  this will raise a TypeError

    :param start: dict or list of dicts
    :param nparticles: int
    :param njobs: int
    :param model: model
    :return:
    """
    model = modelcontext(model)
    if start is None:
        return None

    if isinstance(start, list):
        ns = [np.atleast_1d(v).shape[0] for d in start for k, v in d.iteritems()]
        if nparticles is not None:
            if not all(n == ns[0] for n in ns):  # inconsistent shapes means no particle specification within the dict
                # expects a list of dicts jobs/particles
                l = []
                for njob in range(njobs):
                    d = defaultdict(list)
                    for job_particle in start[njob * nparticles:(njob + 1) * nparticles]:
                        for varname, value in job_particle.iteritems():
                            if varname in [i.name for i in model.vars]:
                                d[varname].append(value)
                    l.append(d)
                start = l
            elif ns[0] == 1:
                if len(start) == nparticles*njobs:
                    l = []
                    for j in range(njobs):
                        d = defaultdict(list)
                        slcs = start[j:(j+1)*nparticles]
                        for s in slcs:
                            for k, v in s.iteritems():
                                d[k].append(v)
                        l.append(d)
                    start = l

                elif len(start) == 1:
                    pm._log.warning("Your given start is specified for 1 job/1 particle when you wanted {} jobs and {} particles, duplicating starting "
                                    "values...".format(njobs, nparticles))
                    start = [{k: [v]*nparticles for d in start for k, v in d.iteritems()}]*njobs

            elif ns[0] == nparticles:
                if len(start) == 1:
                    pm._log.warning("Your given start is specified for 1 job when you wanted {}, duplicating starting "
                                    "values between jobs...".format(njobs))
                    start = start * njobs
                elif len(start) != njobs:
                    raise TypeError("Your given start specified {} jobs when you wanted {}, this is ambiguous".format(len(start), njobs))
            else:
                raise TypeError("Your given start is specified for {} particles when you wanted {}, this is "
                                "ambiguous".format(ns[0], nparticles))
        else:
            if len(start) == 1:
                pm._log.warning("Your given start specified 1 job when you wanted {}, duplicating starting values "
                                "between jobs...".format(njobs))
                start = start * njobs
            elif len(start) != njobs:
                raise TypeError("Your given start specified {} jobs when you wanted {}, this is ambiguous".format(len(start), njobs))

    elif isinstance(start, dict):
        ns = [np.atleast_1d(v).shape[0] for k, v in start.iteritems()]
        if all(n == ns[0] for n in ns):
            if nparticles is not None:
                if ns[0] == nparticles * njobs:
                    l = []
                    for njob in range(njobs):
                        d = {}
                        for varname, value in start.iteritems():
                            d[varname] = value[njob * nparticles:(njob + 1) * nparticles]
                        l.append(d)
                    start = l
                elif ns[0] == nparticles:
                    start = [start]*njobs
                elif ns[0] == 1:  # single guess
                    pm._log.warning(
                        "Your given start is specified for 1 job/1 particle when you wanted {} jobs and {} particles, duplicating starting "
                        "values...".format(njobs, nparticles))
                    d = {}
                    for varname, value in start.iteritems():
                        d[varname] = np.asarray([value]*nparticles)
                    start = [d]*njobs
                else:
                    raise TypeError("Your given start is specified for {} particles when you wanted {}, this is "
                                    "ambiguous".format(ns[0], nparticles))
            elif ns[0] == njobs:
                l = []
                for njob in range(njobs):
                    d = {}
                    for varname, value in start.iteritems():
                        d[varname] = value[njob:njob + 1]
                    l.append(d)
                start = l
        else:
            start = [start] * njobs
    else:
        raise TypeError("Start type {} not understood".format(type(start)))

    names = [i.name for i in model.vars]
    _s = []
    for i, d in enumerate(start):
        _d = {}
        for varname, value in d.iteritems():
            if varname in names:
                _d[varname] = value
        _s.append(_d)
    return _s