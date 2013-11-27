from .point import *
from .trace import NpTrace, MultiTrace
import multiprocessing as mp
from time import time
from .core import *
from . import step_methods
from .progressbar import progress_bar
from numpy.random import seed

__all__ = ['sample', 'psample']


def sample(draws, step, start=None, trace=None, tune=None, progressbar=True, model=None, random_seed=None):
    """
    Draw a number of samples using the given step method.
    Multiple step methods supported via compound step method
    returns the amount of time taken.

    Parameters
    ----------

    draws : int
        The number of samples to draw
    step : function
        A step function
    start : dict
        Starting point in parameter space (or partial point)
        Defaults to trace.point(-1)) if there is a trace provided and
        model.test_point if not (defaults to empty dict)
    trace : NpTrace or list
        Either a trace of past values or a list of variables to track
        (defaults to None)
    tune : int
        Number of iterations to tune, if applicable (defaults to None)
    progressbar : bool
        Flag for progress bar
    model : Model (optional if in `with` context)

    """
    model = modelcontext(model)
    draws = int(draws)
    seed(random_seed)

    if start is None:
        start = {}

    if isinstance(trace, NpTrace) and len(trace) > 0:

        trace_point = trace.point(-1)
        trace_point.update(start)
        start = trace_point

    else:

        test_point = model.test_point.copy()
        test_point.update(start)
        start = test_point

        if not isinstance(trace, NpTrace):
            if trace is None:
                trace = model.named_vars
            try:
                trace = NpTrace(list(trace.values()))
            except AttributeError:
                trace = NpTrace(list(trace))

    try:
        step = step_methods.CompoundStep(step)
    except TypeError:
        pass

    point = Point(start, model=model)

    progress = progress_bar(draws)

    try:
        for i in range(draws):
            if (i == tune):
                step = stop_tuning(step)
            point = step.step(point)
            trace = trace.record(point)
            if progressbar:
                progress.update(i)
    except KeyboardInterrupt:
        pass
    finally:
        return trace

def stop_tuning(step):
    """ stop tuning the current step method """

    if hasattr(step, 'tune'):
        step.tune = False

    elif hasattr(step, 'methods'):
        step.methods = [stop_tuning(s) for s in step.methods]

    return step


def argsample(args):
    """ defined at top level so it can be pickled"""
    return sample(*args)


def psample(draws, step, start=None, trace=None, tune=None, model=None, threads=None,
    random_seeds=None):
    """draw a number of samples using the given step method.
    Multiple step methods supported via compound step method
    returns the amount of time taken

    Parameters
    ----------

    draws : int
        The number of samples to draw
    step : function
        A step function
    start : dict
        Starting point in parameter space (Defaults to trace.point(-1))
    trace : MultiTrace or list
        Either a trace of past values or a list of variables to track (defaults to None)
    tune : int
        Number of iterations to tune, if applicable (defaults to None)
    model : Model (optional if in `with` context)
    threads : int
        Number of parallel traces to start

    Examples
    --------

    >>> an example

    """

    model = modelcontext(model)

    if not threads:
        threads = max(mp.cpu_count() - 2, 1)

    if start is None:
        start = {}

    if isinstance(start, dict):
        start = threads * [start]

    if trace is None:
        trace = model.named_vars

    if type(trace) is MultiTrace:
        mtrace = trace
    else:
        try:
            mtrace = MultiTrace(threads, list(trace.values()))
        except AttributeError:
            mtrace = MultiTrace(threads, trace)

    p = mp.Pool(threads)

    if random_seeds is None:
        random_seeds = [None] * threads

    argset = zip([draws] * threads, [step] * threads, start, mtrace.traces,
                 [tune] * threads, [False] * threads, [model] * threads,
                 random_seeds)

    traces = p.map(argsample, argset)

    p.close()

    return MultiTrace(traces)
