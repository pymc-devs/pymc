# Convergence diagnostics
# Heidelberger and Welch (1983) ?

import numpy as np
import pymc

def geweke(x, first=.1, last=.5, intervals=20):
    """Return z-scores for convergence diagnostics.
    
    Compare the mean of the first % of series with the mean of the last % of 
    series. x is divided into a number of segments for which this difference is
    computed. 

    :Stochastics:
      `x` : series of data,
      `first` : first fraction of series,
      `last` : last fraction of series to compare with first,
      `intervals` : number of segments. 
      
    :Note: Geweke (1992)
      """
    # Filter out invalid intervals
    if first + last >= 1:
        raise "Invalid intervals for Geweke convergence analysis",(first,last)
        
    # Initialize list of z-scores
    zscores = []
    
    # Last index value
    end = len(x) - 1
    
    # Calculate starting indices
    sindices = np.arange(0, end/2, step = int((end / 2) / intervals))
    
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
def raftery_lewis(x, q, r, s=.95, epsilon=.001):
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
        s : float
            Probability of attaining the requested accuracy.
        epsilon : float
             Half width of the tolerance interval required for the q-quantile.
             
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
    return pymc.flib.gibbmain(x, q, r, s, epsilon)

def gelman_rubin(x):
    pass
# x contains multiple chains
# Transform positive or [0,1] variables using a logarithmic/logittranformation.
# 
