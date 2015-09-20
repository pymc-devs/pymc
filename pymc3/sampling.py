from . import backends
from .backends.base import merge_traces, BaseTrace, MultiTrace
from .backends.ndarray import NDArray
import multiprocessing as mp
from time import time
from .core import *
from . import step_methods
from .progressbar import progress_bar
from numpy.random import seed

__all__ = ['sample', 'iter_sample']

def assign_step_methods(model, step):
    
    steps = []
    assigned_vars = set()
    if step is not None:
        steps = np.append(steps, step).tolist()
        try:
            assigned_vars = assigned_vars | set(step.vars)
        except AttributeError:
            for s in step.methods:
                assigned_vars = assigned_vars | set(s.vars)
    
    # Use competence classmethods to select step methods for remaining variables
    selected_steps = {s:[] for s in step_methods.step_method_registry}
    for var in model.free_RVs:
        if not var in assigned_vars:
               
            competences = {s:s._competence(var) for s in
                             step_methods.step_method_registry}
            
            selected = max(competences.keys(), key=(lambda k: competences[k]))
        
            print('Assigned {0} to {1}'.format(selected, var))
            selected_steps[selected].append(var)
    
    # Instantiate all selected step methods
    steps += [s(vars=selected_steps[s]) for s in selected_steps if selected_steps[s]]
    
    if len(steps)==1:
        steps = steps[0]    
                
    return steps

def sample(draws, step=None, start=None, trace=None, chain=0, njobs=1, tune=None,
           progressbar=True, model=None, random_seed=None):
    """
    Draw a number of samples using the given step method.
    Multiple step methods supported via compound step method
    returns the amount of time taken.

    Parameters
    ----------

    draws : int
        The number of samples to draw
    step : function or iterable of functions
        A step function or collection of functions. If no step methods are 
        specified, or are partially specified, they will be assigned 
        automatically (defaults to None).
    start : dict
        Starting point in parameter space (or partial point)
        Defaults to trace.point(-1)) if there is a trace provided and
        model.test_point if not (defaults to empty dict)
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
        Flag for progress bar
    model : Model (optional if in `with` context)
    random_seed : int or list of ints
        A list is accepted if more if `njobs` is greater than one.

    Returns
    -------
    MultiTrace object with access to sampling values
    """
    model = modelcontext(model)
    
    step = assign_step_methods(model, step)

    if njobs is None:
        njobs = max(mp.cpu_count() - 2, 1)
    if njobs > 1:
        try:
            if not len(random_seed) == njobs:
                random_seeds = [random_seed] * njobs
            else:
                random_seeds = random_seed
        except TypeError:  # None, int
            random_seeds = [random_seed] * njobs

        chains = list(range(chain, chain + njobs))

        pbars = [progressbar] + [False] * (njobs - 1)

        argset = zip([draws] * njobs,
                     [step] * njobs,
                     [start] * njobs,
                     [trace] * njobs,
                     chains,
                     [tune] * njobs,
                     pbars,
                     [model] * njobs,
                     random_seeds)
        sample_func = _mp_sample
        sample_args = [njobs, argset]
    else:
        sample_func = _sample
        sample_args = [draws, step, start, trace, chain,
                       tune, progressbar, model, random_seed]
    return sample_func(*sample_args)


def _sample(draws, step=None, start=None, trace=None, chain=0, tune=None,
            progressbar=True, model=None, random_seed=None):
    sampling = _iter_sample(draws, step, start, trace, chain,
                            tune, model, random_seed)
    progress = progress_bar(draws)
    try:
        for i, strace in enumerate(sampling):
            if progressbar:
                progress.update(i)
    except KeyboardInterrupt:
        strace.close()
    return MultiTrace([strace])


def iter_sample(draws, step, start=None, trace=None, chain=0, tune=None,
                model=None, random_seed=None):
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
                 model=None, random_seed=None):
    model = modelcontext(model)
    draws = int(draws)
    seed(random_seed)
    if draws < 1:
        raise ValueError('Argument `draws` should be above 0.')

    if start is None:
        start = {}

    strace = _choose_backend(trace, chain, model=model)

    if len(strace) > 0:
        _soft_update(start, strace.point(-1))
    else:
        _soft_update(start, model.test_point)

    try:
        step = step_methods.CompoundStep(step)
    except TypeError:
        pass

    point = Point(start, model=model)

    strace.setup(draws, chain)
    for i in range(draws):
        if i == tune:
            step = stop_tuning(step)
        point = step.step(point)
        strace.record(point)
        yield strace
    else:
        strace.close()


def _choose_backend(trace, chain, shortcuts=None, **kwds):
    if isinstance(trace, BaseTrace):
        return trace
    if isinstance(trace, MultiTrace):
        return trace._straces[chain]
    if trace is None:
        return NDArray(**kwds)

    if shortcuts is None:
        shortcuts = backends._shortcuts

    try:
        backend = shortcuts[trace]['backend']
        name = shortcuts[trace]['name']
        return backend(name, **kwds)
    except TypeError:
        return NDArray(vars=trace, **kwds)
    except KeyError:
        raise ValueError('Argument `trace` is invalid.')


def _mp_sample(njobs, args):
    p = mp.Pool(njobs)
    traces = p.map(argsample, args)
    p.close()
    return merge_traces(traces)


def stop_tuning(step):
    """ stop tuning the current step method """

    if hasattr(step, 'tune'):
        step.tune = False

    elif hasattr(step, 'methods'):
        step.methods = [stop_tuning(s) for s in step.methods]

    return step


def argsample(args):
    """ defined at top level so it can be pickled"""
    return _sample(*args)


def _soft_update(a, b):
    """As opposed to dict.update, don't overwrite keys if present.
    """
    a.update({k: v for k, v in b.items() if k not in a})
