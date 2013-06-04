from point import *
from trace import NpTrace, MultiTrace
import multiprocessing as mp
from time import time
from core import *
import step_methods
from progressbar import progress_bar
from numpy.random import seed

__all__ = ['sample', 'psample']


def sample(draws, step, start={}, trace=None, progressbar=True, model=None, random_seed=None):
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
    progressbar : bool
        Flag for progress bar
    model : Model (optional if in `with` context)

    """
    model = modelcontext(model)
    draws = int(draws)
    seed(random_seed)

    if trace is not None and len(trace) > 0:

        start = trace.point(-1)

    else:

        test_point = model.test_point.copy()
        test_point.update(start)
        start = test_point

    if not hasattr(trace, 'record'):
        if trace is None:
            trace = model.vars
        trace = NpTrace(list(trace))

    try:
        step = step_methods.CompoundStep(step)
    except TypeError:
        pass

    point = Point(start, model=model)

    progress = progress_bar(draws)

    for i in xrange(draws):
        point = step.step(point)
        trace = trace.record(point)
        if progressbar:
            progress.update(i)

    return trace


def argsample(args):
    """ defined at top level so it can be pickled"""
    return sample(*args)


def psample(draws, step, start, trace=None, model=None, threads=None):
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

    if isinstance(start, dict):
        start = threads * [start]

    if trace is None:
        trace = model.vars

    if type(trace) is MultiTrace:
        mtrace = trace
    else:
        mtrace = MultiTrace(threads, trace)

    p = mp.Pool(threads)

    argset = zip([draws] * threads, [step] * threads, start, mtrace.traces,
                 [False] * threads, [model] * threads)

    traces = p.map(argsample, argset)

    return MultiTrace(traces)
