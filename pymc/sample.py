from point import *
from trace import NpTrace, MultiTrace
import multiprocessing as mp
from time import time
from core import *
import step_methods
from progressbar import progress_bar

__all__ = ['sample', 'psample']

@withmodel
def sample(model, draws, step, start = None, trace = None, track_progress = True): 
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
        Starting point in parameter space (Defaults to trace.point(-1))
    trace : NpTrace
        A trace of past values (defaults to None)
    track : list of vars
        The variables to follow
        
    Examples
    --------
        
    >>> an example
        
    """
    draws = int(draws)
    if start is None: 
        start = trace[-1]
    point = Point(model, start)

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
        if track_progress:
            progress.update(i)

    return trace

def argsample(args):
    """ defined at top level so it can be pickled"""
    return sample(*args)
  
@withmodel
def psample(model, draws, step, start, mtrace = None, threads = None, track = None):
    """draw a number of samples using the given step method. Multiple step methods supported via compound step method
    returns the amount of time taken"""

    if not threads:
        threads = max(mp.cpu_count() - 2, 1)

    if isinstance(start, dict) :
        start = threads * [start]

    if track is None: 
        track = model.vars

    if not mtrace:
        mtrace = MultiTrace(threads, track)

    p = mp.Pool(threads)

    argset = zip([model] *threads, [draws]*threads, [step]*threads, start, mtrace.traces)
    
    traces = p.map(argsample, argset)
        
    return MultiTrace(traces)
