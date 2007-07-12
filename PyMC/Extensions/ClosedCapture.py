"""
Closed-population capture-mark-recapture (CMR) estimation using MCMC with PyMC.
"""

from numpy import  *
from PyMC import  *
from numpy.random import *
from pylab import *

capture_data = {(1,0,0,0,0): 6, 
(1,0,0,0,1): 1, 
(1,0,1,0,1): 1, 
(1,0,1,0,0): 5, 
(0,1,1,0,1): 3, 
(0,1,1,0,0): 5, 
(1,1,1,1,1): 1, 
(1,1,1,1,0): 3, 
(1,1,0,1,0): 4, 
(1,1,0,1,1): 2, 
(0,1,0,0,0): 2, 
(0,1,0,0,1): 2, 
(0,0,1,1,1): 10, 
(0,0,1,1,0): 5, 
(0,0,0,1,0): 9, 
(0,0,0,1,1): 3, 
(1,0,0,1,1): 2, 
(1,0,0,1,0): 5, 
(1,0,1,1,0): 2, 
(1,0,1,1,1): 3, 
(0,1,1,1,1): 4, 
(1,1,0,0,1): 3, 
(1,1,0,0,0): 4, 
(1,1,1,0,0): 4, 
(1,1,1,0,1): 2, 
(0,1,0,1,1): 4, 
(0,1,0,1,0): 2, 
(0,0,0,0,1): 11, 
(0,0,1,0,0): 3, 
(0,0,1,0,1): 3}

def import_data(datafile):
    """
    Import capture histories from a text file. Records shoulld look 
    like this:
    
    100000
    001000
    110000
    100000
    111111
    
    :Arguments:
        datafile : string
            Name of data file to import.
            
    :Note:
        Returns a dictionary of capture history frequencies for use 
        with open capture estimation models.
        
    :SeeAlso: CormackJollySeber, Pradel
    
    """
    
    # Opein specified input file
    ifile = open(datafile)
    
    # Initialize dictionary
    datadict = {}
    
    # Iterate over lines
    for line in ifile:
        if line:
            # Parse string
            hist = tuple([int(i) for i in line.strip()])
            # Add to data dictionary
            try:
                datadict[hist] += 1
            except KeyError:
                datadict[hist] = 1
    
    return datadict

gammln_cof = array([76.18009173, -86.50532033, 24.01409822,
   -1.231739516e0, 0.120858003e-2, -0.536382e-5])
gammln_stp = 2.50662827465

def gammln(xx):
	"""Logarithm of the gamma function"""
	global gammln_cof, gammln_stp
	x = xx - 1.
	tmp = x + 5.5
	tmp = (x + 0.5)*log(tmp) - tmp
	ser = 1.
	for j in range(6):
		x = x + 1.
		ser = ser + gammln_cof[j]/x
	return tmp + log(gammln_stp*ser)
	
def factrl(n, ntop=0, prev=ones((33),Float)):
    """Factorial of N.
    The first 33 values are stored as they are calculated to
    speed up subsequent calculations."""
    if n < 0:
        raise ValueError, 'Negative argument!'
    elif n <= ntop:
        return prev[n]
    elif n <= 32:
        for j in range(ntop+1, n+1):
            prev[j] = j * prev[j-1]
        ntop = n
        return prev[n]
    else:
        return exp(gammln(n+1.))

def factln(n, prev=array(101*(-1.,))):
    """Log factorial of N.
    Values for n=0 to 100 are stored as they are calculated to
    speed up subsequent calls."""
    if n < 0:
        raise ValueError, 'Negative argument!'
    elif n <= 100:
        if prev[n] < 0:
            prev[n] = gammln(n+1.)
        return prev[n]
    else:
        return gammln(n+1.)

class M0(MetropolisHastings):
    # Null model.
    
    def __init__(self, data):
        
        # Class initialization
        MetropolisHastings.__init__(self)
        
        self.data = data
        self.k = len(data.keys()[0])
        self.n = sum([sum(i)*data[i] for i in data])
        self.M = sum(data.values())
        
        # Specify parameters and nodes
        self.parameter('p', init_val=0.5)
        self.parameter('N', init_val=self.M)
        
    def _add_to_post(like):
        # Adds the outcome of the likelihood or prior to self._post
        
        def wrapped_like(*args, **kwargs):
            
            # Initialize multiplier factor for likelihood
            factor = 1.0
            try:
                # Look for multiplier in keywords
                factor = kwargs.pop('factor')
            except KeyError:
                pass
                
            # Call likelihood
            value = like(*args, **kwargs)
            
            # Increment posterior total
            args[0]._post += factor * value 
            
            return factor * value
        
        return wrapped_like
        
    @_add_to_post
    def capture_like(self, x, N, n, M, k, p, name='capture'):
        """Capture log-likelihood"""
        
        llike = factln(N) - sum(map(factln, x)) - factln(N - M)
        llike += n * log(p)
        llike += (k*N - n) * log(1.-p)
        
        return llike
    
    def calculate_likelihood(self):
        # Joint log-likelihood
        self.constrain(self.N, lower=self.M)
        self.beta_prior(self.p, 1, 1)
        self.capture_like(self.data.values(), self.N, self.n, self.M, self.k, self.p)

def run():
    # Run model
    
    sampler = M0(capture_data)
    
    results = sampler.sample(50000, burn=10000)

if __name__  ==  '__main__':
   # Runs if called from command line
   
   run()
