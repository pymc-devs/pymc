from point import *
from trace import NpTrace, MultiTrace
import multiprocessing as mp
from time import time

__all__ = ['sample', 'psample']

def sample(draws, step, start = None, trace = None, state = None): 
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
        Starting point in parameter space
    trace : NpTrace
        A trace of past values (defaults to None)
    state : 
        The current state of the sampler (defaults to None)
        
    Examples
    --------
        
    >>> an example
        
    """

    point = clean_point(start)
    if trace is None: 
        trace = NpTrace()
    # Keep track of sampling time  
    tstart = time() 
    for _ in xrange(int(draws)):
        state, point = step.step(state, point)
        trace = trace + point

    return trace, state, time() - tstart

def argsample(args):
    """ defined at top level so it can be pickled"""
    return sample(*args)
  
def psample(draws, step, start, mtrace = None, state = None, threads = None):
    """draw a number of samples using the given step method. Multiple step methods supported via compound step method
    returns the amount of time taken"""

    if not threads:
        threads = max(mp.cpu_count() - 2, 1)

    if isinstance(start, dict) :
        start = threads * [start]

    if not mtrace:
        mtrace = MultiTrace([NpTrace() for _ in xrange(threads)])

    if not state: 
        state = threads*[None]

    p = mp.Pool(threads)

    argset = zip([draws]*threads, [step]*threads, start, mtrace.traces, state)
    
    # Keep track of sampling time  
    tstart = time() 

    res = p.map(argsample, argset)
    trace, state, _ = zip(*res)
        
    return MultiTrace(trace), state, (time() - tstart)
