from point import *
from trace import NpTrace, MultiTrace
import multiprocessing as mp
from time import time
from core import *
import step_methods
from progressbar import progress_bar

__all__ = ['sample', 'psample']

def sample(draws, step, start=None, trace=None, track_progress=True,
        tune_interval=100, model=None):
    """
    Draw a number of samples using the given step method.
    Multiple step methods supported via compound step method
    returns the amount of time taken.

    Parameters
    ----------

    model : Model (optional if in `with` context)
    draws : int
        The number of samples to draw
    step : function
        A step function
    start : dict
        Starting point in parameter space (Defaults to trace.point(-1))
    trace : NpTrace
        A trace of past values (defaults to None)
    track : list of vars
        The variables to follow

    Examples
    --------

    >>> an example

    """
    model = modelcontext(model)
    draws = int(draws)
    if start is None:
        start = trace[-1]
    point = Point(start, model = model)

    if not hasattr(trace, 'record'):
        if trace is None:
            trace = model.vars
        trace = NpTrace(list(trace))


    try:
        step = step_methods.CompoundStep(step)
    except TypeError:
        pass

    progress = progress_bar(draws)

    for i in xrange(draws):
        point = step.step(point)
        trace = trace.record(point)
        if i and not (i % tune_interval) and step.tune:
            step = tune(step, tune_interval)
        if track_progress:
            progress.update(i)

    return trace


def tune(step, tune_interval):
    """
    Tunes the scaling parameter for the proposal distribution
    according to the acceptance rate over the last tune_interval:

    Rate    Variance adaptation
    ----    -------------------
    <0.001        x 0.1
    <0.05         x 0.5
    <0.2          x 0.9
    >0.5          x 1.1
    >0.75         x 2
    >0.95         x 10

    """

    # Calculate acceptance rate
    acc_rate = step.accepted / float(tune_interval)

    # Switch statement
    if acc_rate<0.001:
        # reduce by 90 percent
        step.scaling *= 0.1
    elif acc_rate<0.05:
        # reduce by 50 percent
        step.scaling *= 0.5
    elif acc_rate<0.2:
        # reduce by ten percent
        step.scaling *= 0.9
    elif acc_rate>0.95:
        # increase by factor of ten
        step.scaling *= 10.0
    elif acc_rate>0.75:
        # increase by double
        step.scaling *= 2.0
    elif acc_rate>0.5:
        # increase by ten percent
        step.scaling *= 1.1

    # Re-initialize rejection count
    step.accepted = 0

    return step

def argsample(args):
    """ defined at top level so it can be pickled"""
    return sample(*args)

def psample(draws, step, start, mtrace=None, track=None, model=None, threads=None):
    """draw a number of samples using the given step method. Multiple step methods supported via compound step method
    returns the amount of time taken"""

    model = modelcontext(model)

    if not threads:
        threads = max(mp.cpu_count() - 2, 1)

    if isinstance(start, dict) :
        start = threads * [start]

    if track is None:
        track = model.vars

    if not mtrace:
        mtrace = MultiTrace(threads, track)

    p = mp.Pool(threads)

    argset = zip([draws]*threads, [step]*threads, start, mtrace.traces, [False]*threads, [model] *threads)

    traces = p.map(argsample, argset)

    return MultiTrace(traces)
