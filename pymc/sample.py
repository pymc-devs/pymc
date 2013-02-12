from point import *
from history import NpHistory, MultiHistory
import multiprocessing as mp
from time import time

__all__ = ['sample', 'psample']

# TODO Can we change `sample_history` to `trace`?
def sample(draws, step, point, sample_history = None, state = None): 
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
    point : float or vector
        The current sample index
    sample_history : NpHistory
        A trace of past values (defaults to None)
    state : 
        The current state of the sampler (defaults to None)
        
    Examples
    --------
        
    >>> an example
        
    """

    point = clean_point(point)
    if sample_history is None: 
        sample_history = NpHistory()
    # Keep track of sampling time  
    tstart = time() 
    for _ in xrange(int(draws)):
        state, point = step.step(state, point)
        sample_history = sample_history + point

    return sample_history, state, time() - tstart

def argsample(args):
    """ defined at top level so it can be pickled"""
    return sample(*args)
  
def psample(draws, step, point, msample_history = None, state = None, threads = None):
    """draw a number of samples using the given step method. Multiple step methods supported via compound step method
    returns the amount of time taken"""

    if not threads:
        threads = max(mp.cpu_count() - 2, 1)

    if isinstance(point, dict) :
        point = threads * [point]

    if not msample_history:
        msample_history = MultiHistory([NpHistory() for _ in xrange(threads)])

    if not state: 
        state = threads*[None]

    p = mp.Pool(threads)

    argset = zip([draws]*threads, [step]*threads, point, msample_history.histories, state)
    
    # Keep track of sampling time  
    tstart = time() 

    res = p.map(argsample, argset)
    trace, state, _ = zip(*res)
        
    return MultiHistory(trace), state, (time() - tstart)
