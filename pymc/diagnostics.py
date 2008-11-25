# Convergence diagnostics
# Heidelberger and Welch (1983) ?

__all__ = ['geweke', 'gelman_rubin', 'raftery_lewis']

import numpy as np
import pymc
import pdb

def diagnostic(f):
    """
    This decorator allows for PyMC arguments of various types to be passed to
    the diagnostic functions. It identifies the type of object and locates its 
    trace(s), then passes the data to the wrapped diagnostic function.
    
    """
    
    def wrapper(pymc_obj, *args, **kwargs):
    
        # Figure out what type of object it is
        try:
            values = {}
            # First try Model type
            for variable in pymc_obj._variables_to_tally:            
                data = variable.trace()
                name = variable.__name__
                values[name] = f(data, *args, **kwargs)
            return values
        except AttributeError:
            pass
            
        try:
            # Then try Node type
            data = pymc_obj.trace()
            name = pymc_obj.__name__
            return f(data, *args, **kwargs)
        except AttributeError:
            pass
        
        # If others fail, assume that raw data is passed
        return f(pymc_obj, *args, **kwargs)
    
    return wrapper

@diagnostic
def geweke(x, first=.1, last=.5, intervals=20):
    """Return z-scores for convergence diagnostics.
    
    Compare the mean of the first % of series with the mean of the last % of 
    series. x is divided into a number of segments for which this difference is
    computed. If the series is converged, this score should oscillate between 
    -1 and 1.

    Parameters
    ----------
    x : array-like
      The trace of some stochastic parameter.
    first : float
      The fraction of series at the beginning of the trace.
    last : float
      The fraction of series at the end to be compared with the section
      at the beginning.
    intervals : int
      The number of segments. 
      
    Returns
    -------
    scores : list [[]]
      Return a list of [i, score], where i is the starting index for each 
      interval and score the Geweke score on the interval.
      
    Notes
    -----
    
    The Geweke score on some series x is computed by:
    
      .. math:: \frac{E[x_s] - E[x_e]}{\sqrt{V[x_s] + V[x_e]}}
    
    where :math:`E` stands for the mean, :math:`V` the variance, 
    :math:`x_s` a section at the start of the series and 
    :math:`x_e` a section at the end of the series.
    
    References
    ----------
    Geweke (1992)
    """
    # Filter out invalid intervals
    if first + last >= 1:
        raise "Invalid intervals for Geweke convergence analysis",(first,last)
        
    # Initialize list of z-scores
    zscores = []
    
    # Last index value
    end = len(x) - 1
    
    # Calculate starting indices
    sindices = np.arange(0, end/2, step = int((end / 2) / (intervals-1)))
    
    # Loop over start indices
    for start in sindices:
        
        # Calculate slices
        first_slice = x[start : start + int(first * (end - start))]
        last_slice = x[int(end - last * (end - start)):]
        
        z = (first_slice.mean() - last_slice.mean())
        z /= np.sqrt(first_slice.std()**2 + last_slice.std()**2)
        
        zscores.append([start, z])
    
    if intervals == None:
        return zscores[0]    
    else:
        return zscores
    
# From StatLib -- gibbsit.f
@diagnostic
def raftery_lewis(x, q, r, s=.95, epsilon=.001, verbose=1):
    """
    Return the number of iterations needed to achieve a given 
    precision.
    
    :Parameters:
        x : sequence
            Sampled series.
        q : float
            Quantile.
        r : float
            Accuracy requested for quantile.
        s (optional): float
            Probability of attaining the requested accuracy (defaults to 0.95).
        epsilon (optional) : float
             Half width of the tolerance interval required for the q-quantile (defaults to 0.001).
        verbose (optional) : int
            Verbosity level for output (defaults to 1).
             
    :Return:
        nmin : int
            Minimum number of independent iterates required to achieve
            the specified accuracy for the q-quantile.                 
        kthin : int
            Skip parameter sufficient to produce a first-order Markov
            chain.      
        nburn : int
            Number of iterations to be discarded at the beginning of the
            simulation, i.e. the number of burn-in iterations.     
        nprec : int
            Number of iterations not including the burn-in iterations which  
            need to be obtained in order to attain the precision specified
            by the values of the q, r and s input parameters.  
        kmind : int
            Minimum skip parameter sufficient to produce an independence
            chain.
    
    :Example:
        >>> raftery_lewis(x, q=.025, r=.005)
    
    :Reference:
        Raftery, A.E. and Lewis, S.M. (1995).  The number of iterations,
        convergence diagnostics and generic Metropolis algorithms.  In
        Practical Markov Chain Monte Carlo (W.R. Gilks, D.J. Spiegelhalter
        and S. Richardson, eds.). London, U.K.: Chapman and Hall. 
        
        See the fortran source file `gibbsit.f` for more details and references.
    """
    output = nmin, kthin, nburn, nprec, kmind = pymc.flib.gibbmain(x, q, r, s, epsilon)
    
    if verbose:
        
        print "========================"
        print "Raftery-Lewis Diagnostic"
        print "========================"
        print
        print "%s iterations required (assuming independence) to achieve %s accuracy with %i percent probability." % (nmin, r, 100*s)
        print
        print "Thinning factor of %i required to produce a first-order Markov chain." % kthin
        print
        print "%i iterations to be discarded at the beginning of the simulation (burn-in)." % nburn
        print
        print "%s subsequent iterations required." % nprec
        print
        print "Thinning factor of %i required to produce an independence chain." % kmind
        
    return output

def batch_means(x, f=lambda y:y, theta=.5, q=.95, burn=0):
    """
    TODO: Use Bayesian CI.
    
    Returns the half-width of the frequentist confidence interval
    (q'th quantile) of the Monte Carlo estimate of E[f(x)]. 
    
    :Parameters:
        x : sequence
            Sampled series. Must be a one-dimensional array.
        f : function
            The MCSE of E[f(x)] will be computed.
        theta : float between 0 and 1
            The batch length will be set to len(x) ** theta.
        q : float between 0 and 1
            The desired quantile.
            
    :Example: 
        >>>batch_means(x, f=lambda x: x**2, theta=.5, q=.95)
        
    :Reference:
        Flegal, James M. and Haran, Murali and Jones, Galin L. (2007).
        Markov chain Monte Carlo: Can we trust the third significant figure?
        <Publication>
        
    :Note:
        Requires SciPy
    """
    
    try:
        import scipy
        from scipy import stats
    except ImportError:
        raise ImportError, 'SciPy must be installed to use batch_means.'
    
    x=x[burn:]
    
    n = len(x)
        
    b = np.int(n**theta)
    a = n/b
    
    t_quant = stats.t.isf(1-q,a-1)
    
    Y = np.array([np.mean(f(x[i*b:(i+1)*b])) for i in xrange(a)])
    sig = b / (a-1.) * sum((Y - np.mean(f(x))) ** 2)
    
    return t_quant * sig / np.sqrt(n)

def gelman_rubin(x):
    raise NotImplementedError
# x contains multiple chains
# Transform positive or [0,1] variables using a logarithmic/logittranformation.
# 
