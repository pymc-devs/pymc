from .point import *
from pymc.backends.ndarray import NDArray
from pymc.backends.base import merge_chains
import multiprocessing as mp
from time import time
from .core import *
from . import step_methods
from .progressbar import progress_bar
from numpy.random import seed

__all__ = ['sample', 'iter_sample']


def sample(draws, step, start=None, db=None, chain=0, threads=1, tune=None,
           progressbar=True, model=None, variables=None, random_seed=None):
    """Draw samples using the given step method

    Parameters
    ----------
    draws : int
        The number of samples to draw
    step : step method or list of step methods
    start : dict
        Starting point in parameter space (or partial point). Defaults
        to model.test_point.
    db : backend
        If None, NDArray is used.
    chain : int
        Chain number used to store sample in trace. If threads greater
        than one, chain numbers will start here
    threads : int
        Number of parallel traces to start. If None, set to number of
        cpus in the system - 2.
    tune : int
        Number of iterations to tune, if applicable
    progressbar : bool
        Flag for progress bar
    model : Model (optional if in `with` context)
    variables : list
        Variables to sample. If None, defaults to model.unobserved_RVs.
        Ignored if model argument is supplied.
    random_seed : int or list of ints
        List accepted if more than one thread.

    Returns
    -------
    Backend object with access to sampling values
    """
    if threads is None:
        threads = max(mp.cpu_count() - 2, 1)
    if threads > 1:
        try:
            if not len(random_seed) == threads:
                random_seeds = [random_seed] * threads
            else:
                random_seeds = random_seed
        except TypeError:  # None, int
            random_seeds = [random_seed] * threads

        chains = list(range(chain, chain + threads))
        argset = zip([draws] * threads,
                     [step] * threads,
                     [start] * threads,
                     [db] * threads,
                     chains,
                     [tune] * threads,
                     [False] * threads,
                     [model] * threads,
                     [variables] * threads,
                     random_seeds)
        sample_func = _thread_sample
        sample_args = [threads, argset]
    else:
        sample_func = _sample
        sample_args = [draws, step, start, db, chain,
                       tune, progressbar, model, variables, random_seed]
    return sample_func(*sample_args)


def _sample(draws, step, start=None, db=None, chain=0, tune=None,
            progressbar=True, model=None, variables=None, random_seed=None):
    sampling = _iter_sample(draws, step, start, db, chain,
                            tune, model, variables, random_seed)
    if progressbar:
        sampling = enumerate_progress(sampling, draws)
    else:
        sampling = enumerate(sampling)

    try:
        for i, trace in enumerate(sampling):
            if progressbar:
                progress.update(i)
    except KeyboardInterrupt:
        trace.backend.clean_interrupt(i)
        trace.backend.close()
    return trace


def iter_sample(draws, step, start=None, db=None, chain=0, tune=None,
                model=None, variables=None, random_seed=None):
    """
    Generator that returns a trace on each iteration using the given step
    method.

    Parameters
    ----------

    draws : int
        The number of samples to draw
    step : step method or list of step methods
    start : dict
        Starting point in parameter space (or partial point). Defaults
        to model.test_point.
    db : backend
        If None, NDArray is used.
    chain : int
        Chain number used to store sample in trace. If threads greater
        than one, chain numbers will start here
    tune : int
        Number of iterations to tune, if applicable
    model : Model (optional if in `with` context)
    variables : list
        Variables to sample. If None, defaults to model.unobserved_RVs.
        Ignored if model argument is supplied.
    random_seed : int or list of ints
        List accepted if more than one thread.

    Example
    -------

    for trace in iter_sample(500, step):
        ...
    """
    sampling = _iter_sample(draws, step, start, db, chain,
                            tune, model, variables, random_seed)
    for i, trace in enumerate(sampling):
        yield trace[:i + 1]


def _iter_sample(draws, step, start=None, db=None, chain=0, tune=None,
                 model=None, variables=None, random_seed=None):
    seed(random_seed)
    model = modelcontext(model)
    draws = int(draws)
    if draws < 1:
        raise ValueError('Argument `draws` should be above 0')

    if start is None:
        start = {}
    _soft_update(start, model.test_point)

    try:
        step = step_methods.CompoundStep(step)
    except TypeError:
        pass

    point = Point(start, model=model)

    if db is None:
        db = NDArray(model=model, variables=variables)
    db.setup_samples(draws, chain)

    for i in range(draws):
        if i == tune:
            step = stop_tuning(step)
        point = step.step(point)
        db.record(point, i)
        if not i % 1000:
            db.commit()
        yield db.trace
    else:
        db.close()


def _thread_sample(threads, args):
    p = mp.Pool(threads)
    traces = p.map(_argsample, args)
    p.close()
    return merge_chains(traces)


def _argsample(args):
    """Defined at top level so it can be pickled"""
    return _sample(*args)


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
