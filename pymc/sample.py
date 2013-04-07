from point import *
from trace import NpTrace, MultiTrace
import multiprocessing as mp
from time import time
from core import *
import step_methods

__all__ = ['sample', 'psample']

@withmodel
def sample(model, draws, step, start = None, trace = None, vars = None): 
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
    state : 
        The current state of the sampler (defaults to None)
        
    Examples
    --------
        
    >>> an example
        
    """
    if start is None: 
        start = trace[-1]
    point = Point(start)

    if vars is None: 
        vars = model.vars

    if trace is None: 
        trace = NpTrace(vars)

    try:
        step = step_methods.CompoundStep(step)
    except TypeError:
        pass

    for _ in xrange(int(draws)):
        point = step.step(point)
        trace = trace.record(point)

    return trace

def argsample(args):
    """ defined at top level so it can be pickled"""
    return sample(*args)
  
@withmodel
def psample(model, draws, step, start, mtrace = None, threads = None, vars = None):
    """draw a number of samples using the given step method. Multiple step methods supported via compound step method
    returns the amount of time taken"""

    if not threads:
        threads = max(mp.cpu_count() - 2, 1)

    if isinstance(start, dict) :
        start = threads * [start]

    if vars is None: 
        vars = model.vars

    if not mtrace:
        mtrace = MultiTrace(threads, vars)

    p = mp.Pool(threads)

    argset = zip([model] *threads, [draws]*threads, [step]*threads, start, mtrace.traces)
    
    traces = p.map(argsample, argset)
        
    return MultiTrace(traces)
