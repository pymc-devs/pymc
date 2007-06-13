"""
Markov chain Monte Carlo (MCMC) simulation, implementing an adaptive form of random walk Metropolis-Hastings sampling.
"""

# Import system functions
import sys, time, unittest, pdb, csv

# Import numpy functions
from numpy import random, linalg
# Generalized inverse
inverse = linalg.pinv
from numpy import absolute, any, arange, around, array, asarray
from numpy import concatenate
from numpy import diagonal, diag
from numpy import exp, eye
from numpy import log
from numpy import mean, cov, std
from numpy import ndim, ones, outer
from numpy import pi
from numpy import ravel, resize
from numpy import searchsorted, shape, sqrt, sort, swapaxes, where
from numpy import tan, transpose, vectorize, zeros
permutation = random.permutation

from TimeSeries import autocorr as acf

try:
    from Matplot import PlotFactory
except ImportError:
    print 'Matplotlib module not detected ... plotting disabled.'

# Import statistical functions and random number generators
random_number = random.random
random_integers = random.random_integers
uniform = random.uniform
randint = random.randint
from flib import categor as _categorical
from flib import rcat as rcategorical
rexponential = random.exponential
from flib import binomial as fbinomial
rbinomial = random.binomial
from flib import bernoulli as fbernoulli
from flib import hyperg as fhyperg
from flib import mvhyperg as fmvhyperg
from flib import geometric as fgeometric
rgeometric = random.geometric
from flib import negbin as fnegbin
from flib import negbin2 as fnegbin2
rnegbin = random.negative_binomial
from flib import normal as fnormal
rnormal = random.normal
from flib import mvnorm as fmvnorm
rmvnormal = random.multivariate_normal
from flib import poisson as fpoisson
rpoisson = random.poisson
from flib import wishart as fwishart
from flib import gamma as fgamma
rgamma = random.gamma
rchi2 = random.chisquare
from flib import beta as fbeta
rbeta = random.beta
from flib import hnormal as fhalfnormal
from flib import dirichlet as fdirichlet
from flib import dirmultinom as fdirmultinom
from flib import multinomial as fmultinomial
rmultinomial = random.multinomial
from flib import weibull as fweibull
from flib import cauchy as fcauchy
from flib import lognormal as flognormal
from flib import igamma as figamma
from flib import wshrt
from flib import gamfun
from flib import chol
from flib import logit as _logit, invlogit as _invlogit

# Shorthand of some verbose local variables

t = transpose

# Vectorize invlogit and logit
invlogit = vectorize(_invlogit)
logit = vectorize(_logit)

# Not sure why this doesnt work on Windoze!
try:
    inf = float('inf')
except ValueError:
    inf = 1.e100

class DivergenceError(ValueError):
    # Exception class for catching divergent models
    pass

class ParameterError(ValueError):
    # Exception class for catching parameters with invalid values
    pass

class LikelihoodError(ValueError):
    # Log-likelihood is invalid or negative informationnite
    pass

""" Random number generation """

def randint(upper, lower, size=None):
    """Returns a random integer. Accepts float arguments."""
    
    return randint(int(upper), int(lower), size=size)

def runiform(lower, upper, n=None):
    """Returns uniform random numbers"""
    
    if n:
        return uniform(lower, upper, n)
    else:
        return uniform(lower, upper)


def rmixeduniform(a,m,b,n=None):
    """2-level uniform random number generator
    Generates a random number in the range (a,b)
    with median m, such that the distribution
    of values above m is uniform, and the distribution
    of values below m is also uniform, although
    with a different frequency.
    
    a, m, and b should be scalar.  The fourth
    parameter, n, specifies the number of
    iid replicates to generate."""
    
    if n:
        u = random_number(n)
    else:
        u = random_number()
    
    return (u<=0.5)*(2.*u*(m-a)+a) + (u>0.5)*(2.*(u-0.5)*(b-m)+m)

def rdirichlet(alphas, n=None):
    """Returns Dirichlet random variates"""
    
    if n:
        gammas = transpose([rgamma(alpha,1,n) for alpha in alphas])
        
        return array([g/sum(g) for g in gammas])
    else:
        gammas = array([rgamma(alpha,1) for alpha in alphas])
        
        return gammas/sum(gammas)

def rdirmultinom(thetas, N, n=None):
    """Returns Dirichlet-multinomial random variates"""
    
    if n:
        
        return array([rmultinomial(N, p) for p in rdirichlet(thetas, n=n)])
    
    else:
        p = rdirichlet(thetas)
        
        return rmultinomial(N, p)

def rcauchy(alpha, beta, n=None):
    """Returns Cauchy random variates"""
    
    if n:
        return alpha + beta*tan(pi*random_number(n) - pi/2.0)
    
    else:
        return alpha + beta*tan(pi*random_number() - pi/2.0)

def rwishart(n, sigma, m=None):
    """Returns Wishart random matrices"""
    
    D = [i for i in ravel(t(chol(sigma))) if i]
    np = len(sigma)
    
    if m:
        return [expand_triangular(wshrt(D, n, np), np) for i in range(m)]
    else:
        return expand_triangular(wshrt(D, n, np), np)

def rhyperg(draws, red, total, n=None):
    """Returns n hypergeometric random variates of size 'draws'"""
    
    urn = [1]*red + [0]*(total-red)
    
    if n:
        return [sum(urn[i] for i in permutation(total)[:draws]) for j in range(n)]
    else:
        return sum(urn[i] for i in permutation(total)[:draws])

def rmvhyperg(draws, colors, n=None):
    """ Returns n multivariate hypergeometric draws of size 'draws'"""
    
    urn = concatenate([[i]*count for i,count in enumerate(colors)])
    
    if n:
        draw = [[urn[i] for i in permutation(len(urn))[:draws]] for j in range(n)]
        
        return [[sum(draw[j]==i) for i in range(len(colors))] for j in range(n)]
    else:
        draw = [urn[i] for i in permutation(len(urn))[:draws]]
        
        return [sum(draw==i) for i in range(len(colors))]

""" Loss functions """

absolute_loss = lambda o,e: absolute(o - e)

squared_loss = lambda o,e: (o - e)**2

chi_square_loss = lambda o,e: (1.*(o - e)**2)/e


""" Support functions """

def make_indices(dimensions):
    # Generates complete set of indices for given dimensions
    
    level = len(dimensions)
    
    if level==1: return range(dimensions[0])
    
    indices = [[]]
    
    while level:
        
        _indices = []
        
        for j in range(dimensions[level-1]):
            
            _indices += [[j]+i for i in indices]
        
        indices = _indices
        
        level -= 1
    
    try:
        return [tuple(i) for i in indices]
    except TypeError:
        return indices

def expand_triangular(X,k):
    # Expands flattened triangular matrix
    
    # Convert to list
    X = X.tolist()
    
    # Unflatten matrix
    Y = array([[0] * i + X[i * k - (i * (i - 1)) / 2 : i * k + (k - i)] for i in range(k)])
    
    # Loop over rows
    for i in range(k):
        # Loop over columns
        for j in range(k):
            Y[j, i] = Y[i, j]
    
    return Y

# Centered normal random deviate
normal_deviate = lambda var, shape : rnormal(0, var, size=shape)

# Centered uniform random deviate
uniform_deviate = lambda half_width, shape: uniform(-half_width, half_width, size=shape)

# Centered discrete uniform random deviate
discrete_uniform_deviate = lambda half_width, shape: randint(-half_width, half_width, size=shape)

def double_exponential_deviate(beta, size):
    """Centered double-exponential random deviate"""
    
    u = random_number(size)
    
    if u<0.5:
        return beta*log(2*u)
    return -beta*log(2*(1-u))

""" Core MCMC classes """

class Node:
    """
    A class for stochastic process variables that are updated through
    stochastic simulation. The current value is stored as an element in the
    dictionary of the MCMC sampler object, while past values are stored in
    a trace array. Includes methods for generating statistics and plotting.
    
    This class is usually instantiated from within the MetropolisHastings
    class, or its subclasses.
    """
    
    def __init__(self, name, sampler, shape=None, plot=True):
        """Class initialization"""
        
        self.name = name
        
        # Specify sampler
        self._sampler = sampler
        
        # Flag for plotting
        self._plot = plot
        
        if shape:
            self.set_value(zeros(shape, 'd'))
        
        # Empty list of traces
        self._traces = []
    
    def init_trace(self, size):
        """Initialize trace"""
        
        self._traces.append(zeros(size, 'd'))
    
    def get_value(self):
        """Retrieves the current value of node from sampler dictionary"""
        
        return self._sampler.__dict__[self.name]
    
    def clear_trace(self):
        """Removes last trace"""
        
        self._traces.pop()
    
    def set_value(self, value):
        """Stores new value for node in sampler dictionary"""
        
        # Cast to array if value is not scalar (Please do not change this behaviour)
        if shape(value):
            self._sampler.__dict__[self.name] = array(value)
        else:
            self._sampler.__dict__[self.name] = value
    
    
    def tally(self, index):
        """Adds current value to trace"""
        
        # Need to make a copy of arrays
        try:
            self._traces[-1][index] = self.get_value().copy()
        except AttributeError:
            self._traces[-1][index] = self.get_value()
    
    def get_trace(self, burn=0, thin=1, chain=-1, composite=False):
        """Return the specified trace (last by default)"""
        
        try:
            if composite:
                
                return concatenate([trace[arange(burn, len(trace), step=thin)] for trace in self._traces])
            
            return array(self._traces[chain][arange(burn, len(self._traces[chain]), step=thin)])
        
        except IndexError:
            
            return
    
    def trace_count(self):
        """Return number of stored traces"""
        
        return len(self._traces)
    
    def quantiles(self, qlist=[2.5, 25, 50, 75, 97.5], burn=0, thin=1, chain=-1, composite=False):
        """Returns a dictionary of requested quantiles"""
        
        # Make a copy of trace
        trace = self.get_trace(burn, thin, chain, composite)
        
        # For multivariate node
        if ndim(trace)>1:
            # Transpose first, then sort, then transpose back
            trace = t(sort(t(trace)))
        else:
            # Sort univariate node
            trace = sort(trace)
        
        try:
            # Generate specified quantiles
            quants = [trace[int(len(trace)*q/100.0)] for q in qlist]
            
            return dict(zip(qlist, quants))
        
        except IndexError:
            print "Too few elements for quantile calculation"
    
    def print_quantiles(self, qlist=[2.5, 25, 50, 75, 97.5], burn=0, thin=1):
        """Pretty-prints quantiles to screen"""
        
        # Generate quantiles
        quants = self.quantiles(qlist, burn=burn, thin=thin)
        
        # Sort and print quantiles
        if quants:
            keys = quants.keys()
            keys.sort()
            
            print 'Quantiles of', self.name
            for k in keys:
                print '\t', k, ':', quants[k]
    
    def plot(self, plotter, burn=0, thin=1, chain=-1, color='b', composite=False):
        """Plot trace and histogram using Matplotlib"""
        
        if self._plot:
            # Call plotting support function
            try:
                trace = self.get_trace(burn, thin, chain, composite)
                plotter.plot(trace, self.name, color=color)
                del trace
            except Exception:
                print 'Could not generate %s plots' % self.name
    
    def mean(self, burn=0, thin=1, chain=-1, composite=False):
        """Calculate mean of sampled values"""
        
        # Make a copy of trace
        trace = self.get_trace(burn, thin, chain, composite)
        
        # For multivariate node
        if ndim(trace)>1:
            
            # Transpose first, then sort
            traces = t(trace, range(ndim(trace))[1:]+[0])
            dims = shape(traces)
            
            # Container list for intervals
            means = resize(0.0, dims[:-1])
            
            for index in make_indices(dims[:-1]):
                
                means[index] = traces[index].mean()
            
            return means
        
        else:
            
            return trace.mean()
    
    def std(self, burn=0, thin=1, chain=-1, composite=False):
        """Calculate standard deviation of sampled values"""
        
        # Make a copy of trace
        trace = self.get_trace(burn, thin, chain, composite)
        
        # For multivariate node
        if ndim(trace)>1:
            
            # Transpose first, then sort
            traces = t(trace, range(ndim(trace))[1:]+[0])
            dims = shape(traces)
            
            # Container list for intervals
            means = resize(0.0, dims[:-1])
            
            for index in make_indices(dims[:-1]):
                
                means[index] = traces[index].std()
            
            return means
        
        else:
            
            return trace.std()
    
    def mcerror(self, burn=0, thin=1, chain=-1, composite=False):
        """Calculate MC error of chain"""
        
        sigma = self.std(burn, thin, chain, composite)
        n = len(self.get_trace(burn, thin, chain, composite))
        
        return sigma/sqrt(n)
    
    def _calc_min_int(self, trace, alpha):
        """Internal method to determine the minimum interval of
        a given width"""
        
        # Initialize interval
        min_int = [None,None]
        
        try:
            
            # Number of elements in trace
            n = len(trace)
            
            # Start at far left
            start, end = 0, int(n*(1-alpha))
            
            # Initialize minimum width to large value
            min_width = inf
            
            while end < n:
                
                # Endpoints of interval
                hi, lo = trace[end], trace[start]
                
                # Width of interval
                width = hi - lo
                
                # Check to see if width is narrower than minimum
                if width < min_width:
                    min_width = width
                    min_int = [lo, hi]
                
                # Increment endpoints
                start +=1
                end += 1
            
            return min_int
        
        except IndexError:
            print 'Too few elements for interval calculation'
            return [None,None]
    
    def hpd(self, alpha, burn=0, thin=1, chain=-1, composite=False):
        """Calculate HPD (minimum width BCI) for given alpha"""
        
        # Make a copy of trace
        trace = self.get_trace(burn, thin, chain, composite)
        
        # For multivariate node
        if ndim(trace)>1:
            
            # Transpose first, then sort
            traces = t(trace, range(ndim(trace))[1:]+[0])
            dims = shape(traces)
            
            # Container list for intervals
            intervals = resize(0.0, dims[:-1]+(2,))
            
            for index in make_indices(dims[:-1]):
                
                try:
                    index = tuple(index)
                except TypeError:
                    pass
                
                # Sort trace
                trace = sort(traces[index])
                
                # Append to list
                intervals[index] = self._calc_min_int(trace, alpha)
            
            # Transpose back before returning
            return array(intervals)
        
        else:
            # Sort univariate node
            trace = sort(trace)
            
            return array(self._calc_min_int(trace, alpha))
    
    def _calc_zscores(self, trace, a, b, intervals=20):
        """Internal support method to calculate z-scores for convergence
        diagnostics"""
        
        # Initialize list of z-scores
        zscores = []
        
        # Last index value
        end = len(trace) - 1
        
        # Calculate starting indices
        sindices = arange(0, end/2, step = int((end / 2) / intervals))
        
        # Loop over start indices
        for start in sindices:
            
            # Calculate slices
            slice_a = trace[start : start + int(a * (end - start))]
            slice_b = trace[int(end - b * (end - start)):]
            
            z = (slice_a.mean() - slice_b.mean())
            z /= sqrt(slice_a.std()**2 + slice_b.std()**2)
            
            zscores.append([start, z])
        
        return zscores
    
    def geweke(self, first=0.1, last=0.5, intervals=20, burn=0, thin=1, chain=-1, plotter=None, color='b'):
        """Test for convergence according to Geweke (1992)"""
        
        # Filter out invalid intervals
        if first + last >= 1:
            print "Invalid intervals for Geweke convergence analysis"
            return
        
        zscores = {}
        
        # Grab a copy of trace
        trace = self.get_trace(burn=burn, thin=thin, chain=chain)
        
        # For multivariate node
        if ndim(trace)>1:
            
            # Generate indices for node elements
            traces = t(trace, range(ndim(trace))[1:]+[0])
            dims = shape(traces)
            
            for index in make_indices(dims[:-1]):
                
                try:
                    name = "%s_%s" % (self.name, '_'.join([str(i) for i in index]))
                except TypeError:
                    name = "%s_%s" % (self.name, index)
                
                zscores[name] = self._calc_zscores(traces[index], first, last, intervals)
                
                # Plot if asked
                if plotter and self._plot:
                    plotter.geweke_plot(t(zscores[name]), name=name, color=color)
        
        else:
            
            zscores[self.name] = self._calc_zscores(trace, first, last, intervals)
            
            # Plot if asked
            if plotter and self._plot:
                plotter.geweke_plot(t(zscores[self.name]), name=self.name)
        
        return zscores
    
    def autocorrelation(self, max_lag=100, burn=0, thin=1, chain=-1, plotter=None, color='b'):
        """Calculate and plot autocorrelation"""
        
        autocorr = {}
        
        # Grab a copy of trace
        trace = self.get_trace(burn=burn, thin=thin, chain=chain)
        
        # For multivariate node
        if ndim(trace)>1:
            
            # Generate indices for node elements
            traces = t(trace, range(ndim(trace))[1:]+[0])
            dims = shape(traces)
            
            for index in make_indices(dims[:-1]):
                
                try:
                    # Separate index numbers by underscores
                    name = "%s_%s" % (self.name, '_'.join([str(i) for i in index]))
                except TypeError:
                    name = "%s_%s" % (self.name, index)
                
                # Call autocorrelation function across range of lags
                autocorr[name] = [acf(traces[index], k) for k in range(max_lag + 1)]
        
        else:
            
            # Call autocorrelation function across range of lags
            autocorr[self.name] = [acf(trace, k) for k in range(max_lag + 1)]
        
        # Plot if asked
        if plotter and self._plot:
            plotter.bar_series_plot(autocorr, ylab='Autocorrelation', color=color, suffix='-autocorr')
        
        return autocorr


class Parameter(Node):
    """
    Parameter class extends Node class, and represents a variable to be
    estimated using MCMC sampling. Generates candidate values using a
    random walk algorithm. The default proposal is a standard normal
    density, though any zero-centered distribution may be substituted. The
    proposal is usually adapted by the sampler to achieve an optimal
    acceptance rate (between 20 and 50 percent).
    """
    
    def __init__(self, name, init_val, sampler, dist='normal', scale=None, random=False, plot=True):
        # Class initialization
        
        # Initialize superclass
        Node.__init__(self, name, sampler, plot=plot)
        
        # Counter for rejected proposals; used for adaptation.
        self._rejected = 0
        
        # Initialize current value
        self.set_value(init_val)
        
        # Record dimension of parameter
        self.dim = shape(init_val)
        
        # Specify distribution of random walk deviate, and associated
        # scale parameters
        self._dist_name = dist
        if dist == 'exponential':
            self._dist = double_exponential_deviate
        elif dist == 'uniform':
            self._dist = uniform_deviate
        elif dist == 'normal':
            self._dist = normal_deviate
        elif dist == 'multivariate_normal':
            if self.get_value().ndim > 1:
                raise AttributeError, 'The multivariate_normal case is only intended for 1D arrays.'
            self._dist = lambda S, size : rmvnormal(zeros(self.dim), S)
        else:
            raise AttributeError, 'Proposal distribution for %s not recognized' % name
            
        
        # Vectorize proposal distribution if parameter is vector-valued
        # But not multivariate_normal, since it is already vectorized
        #if self.dim and dist != 'multivariate_normal':
            #self._dist = vectorize(self._dist)
        
        # Scale parameter for proposal distribution
        if scale is None:
            if dist == 'multivariate_normal':
                self._hyp = eye(*self.dim)
            else:
                self._hyp = 1.0
            """
            elif self.dim:  # Vector case
                self._hyp = ones(self.dim)
            else:           # Scalar case
                self._hyp = 1.0
            """
        elif dist == 'multivariate_normal':
            # Only the std variations are given. 
            if shape(scale) == shape(init_val):
                self._hyp = diag(scale)
            # The complete covariance matrix is given
            elif shape(scale) == shape(init_val)*2:
                self._hyp = asarray(scale)
        else:
            # Scalar or vector scale. Will raise an error if scale is not 
            # compatible with init_val.
            self._hyp = ones(self.dim) * asarray(scale)
        
        # Adaptative scaling factor
        self._asf = 1.
        
        # Random effect flag (for use in AIC calculation)
        self.random = random
    
    def sample_candidate(self):
        """Samples a candidate value based on proposal distribution"""
        
        try:
            return self.get_value() + self._dist(self._hyp*self._asf, shape(self.get_value()))
        
        except ValueError:
            print 'Hyperparameter approaching zero:', self._hyp
            raise DivergenceError
    
    def propose(self, debug=False):
        """Propose new values using a random walk algorithm, according to
        the proposal distribution specified:
        
        x(t+1) = x(t) + e(t)
        
        where e ~ proposal(hyperparameters)
        """
        
        # Current value of parameter
        current_value = self.get_value()
        
        # Sample candidate values using random walk
        new_value = self.sample_candidate()
        
        # Replace current value with new
        self.set_value(new_value)
        
        # If not accepted, replace old value
        if not self._sampler.test():
            self.set_value(current_value)
            self._rejected += 1
            if debug:
                print 'REJECTED proposed value %s for parameter %s' % (new_value, self.name)
                print
        else:
            if debug:
                print 'ACCEPTED proposed value %s for parameter %s' % (new_value, self.name)
                print
    
    def tune(self, int_length, divergence_threshold=1e10, verbose=False):
        """
        Tunes the scaling hyperparameter for the proposal distribution
        according to the acceptance rate of the last k proposals:
        
        Rate    Variance adaptation
        ----    -------------------
        <0.001        x 0.1
        <0.05         x 0.5
        <0.2          x 0.9
        >0.5          x 1.1
        >0.75         x 2
        >0.95         x 10
        
        This method is called exclusively during the burn-in period of the
        sampling algorithm.
        """
        
        if verbose:
            print
            print 'Tuning', self.name
            print '\tcurrent value:', self.get_value()
            print '\tcurrent proposal hyperparameter:', self._hyp*self._asf
        
        # Calculate recent acceptance rate
        acc_rate = 1.0 - self._rejected*1.0/int_length
        
        tuning = True
        
        # Switch statement
        if acc_rate<0.001:
            # reduce by 90 percent
            self._asf *= 0.1
        elif acc_rate<0.05:
            # reduce by 50 percent
            self._asf *= 0.5
        elif acc_rate<0.2:
            # reduce by ten percent
            self._asf *= 0.9
        elif acc_rate>0.95:
            # increase by factor of ten
            self._asf *= 10.0
        elif acc_rate>0.75:
            # increase by double
            self._asf *= 2.0
        elif acc_rate>0.5:
            # increase by ten percent
            self._asf *= 1.1
        else:
            tuning = False
        
        # Re-initialize rejection count
        self._rejected = 0
        
        # If the scaling factor is diverging, abort
        if self._asf > divergence_threshold:
            raise DivergenceError, 'Proposal distribution variance diverged'
        
        # Compute covariance matrix in the multivariate case and the standard
        # variation in all other cases.
        #self.compute_scale(acc_rate,  int_length)
        
        if verbose:
            print '\tacceptance rate:', acc_rate
            print '\tadaptive scaling factor:', self._asf
            print '\tnew proposal hyperparameter:', self._hyp*self._asf
        
        return tuning
    
    def compute_scale(self, acc_rate, int_length):
        # For multidimensional parameters, compute scaling factor for proposal
        # hyperparameter on last segment of computed trace.
        # Make sure that acceptance rate is high enough to ensure
        # nonzero covariance
        try :
            if (self._hyp.ndim) and (acc_rate > 0.05):
                
                # Length of trace containing non-zero elements (= current iteration)
                it = where(self._traces[-1]==0.)[0][0]
                
                # Computes the variance over the last 3 intervals.
                _var = cov(self._traces[-1][max(0, it-3 * int_length):it],axis=0)
                
                # Uncorrelated multivariate case
                if self._dist_name != 'multivariate_normal':
                    
                    # Ensure that there are no null values before commiting to self.
                    if (_var > 0).all():
                        self._hyp = sqrt(_var)
                
                # Correlated multivariate case
                # Compute correlation coefficients and clip correlation to .9 to
                # in order to avoid perfect correlations. 
                # Compute the covariance matrix and set it as _hyp.
                else:
                    d = diag(_var)
                    if (d > 0).all():
                        corr = _var / sqrt(outer(d,d))
                        corr = corr.clip(-.9, .9)
                        corr[range(self.ndim), range(self.ndim)] = 1.
                        covariance = corr * sqrt(outer(d,d)) 
                        self._hyp = covariance                       
        
        except AttributeError:
                pass


class BinaryParameter(Parameter):
    """
    Parameter subclass that only has 2 possible values: {x:0,1}
    """
    
    def __init__(self, name, init_val, sampler, dist='normal', scale=None, random=False, plot=True):
        # Class initialization
        
        # Initialize superclass
        Parameter.__init__(self, name, init_val, sampler, dist='normal', scale=None, random=False, plot=True)
        
        # Counter for rejected proposals; used for adaptation.
        self._rejected = 0
        
        # Initialize current value
        self.set_value(init_val)
        
        # Record dimension of parameter
        self.dim = shape(init_val)
        
        # Random effect flag (for use in AIC calculation)
        self.random = random
        
    def sample_candidate(self):
        """Samples a candidate value based on proposal distribution"""
        
        try:
            current = self.get_value()
            
            return abs(current - (random_number(shape(current)) < invlogit(self._hyp*self._asf)).astype('i'))
        
        except ValueError:
            print 'Hyperparameter approaching zero:', self._hyp
            raise DivergenceError
        
    

class DiscreteParameter(Parameter):
    
    def __init__(self, name, init_val, sampler, dist='normal', scale=None, random=False, plot=True):
        # Class initialization
        
        # Initialize superclass
        Parameter.__init__(self, name, init_val, sampler, scale=scale, random=random, plot=plot)
        
        # Specify distribution of random walk deviate, and associated
        # scale parameters
        if dist == 'exponential':
            self._dist = double_exponential_deviate
        elif dist == 'uniform':
            self._dist = discrete_uniform_deviate
        elif dist == 'normal':
            self._dist = normal_deviate
        else:
            print 'Proposal distribution for', name, 'not recognized'
            sys.exit()
    
    def set_value(self, value):
        """Stores new value for node in sampler dictionary"""
        
        try:
            self._sampler.__dict__[self.name] = around(value, 0).astype(int)
        except TypeError:
            self._sampler.__dict__[self.name] = int(round(value, 0))
    
    def compute_scale(self, acc_rate, int_length):
        """Returns 1 for discrete parameters.'"""
        # Since discrete parameters may not show much variability, they may
        # return standard variations equal to 0. It is hence more robust to let _asf
        # take care of the tuning.
        pass



class Sampler:
    """
    Superclass for Markov chain Monte Carlo samplers. Includes methods for
    calculating log-likelihoods, initializing parameters and nodes,
    and generating summary output.
    
    This general class is essentially a framework from which specific
    MCMC algorithms should be subclassed (e.g. Slice and MetropolisHastings).
    """
    
    def __init__(self, plot_format='png', plot_backend='TkAgg'):
        """Class initializer"""
        
        # Initialize parameter and node dictionaries
        self.parameters = {}
        self.nodes = {}
        
        # Create and initialize node for model deviance
        self.node('deviance')
        
        # Create plotter, if module was imported
        try:
            # Pass appropriate graphic format and backend
            self.plotter = PlotFactory(format=plot_format, backend=plot_backend)
        except NameError:
            self.plotter = None
        
        # Goodness of Fit flag
        self._gof = False
    
    def profile(self, iterations=2000, burn=1000, name='pymc'):
        """Profile sampler with hotshot"""
        
        from hotshot import Profile, stats
        
        # Create profile object
        prof = Profile("%s.prof" % name)
        
        # Run profile
        results = prof.runcall(self.sample, iterations, burn=burn, plot=False, verbose=False)
        prof.close()
        
        # Load stats
        s = stats.load("%s.prof" % name)
        s.strip_dirs()
        s.sort_stats('time','calls')
        
        # Print
        s.print_stats()
        
        # Clear traces of each parameter and node after profiling
        for node in self.parameters.values()+self.nodes.values():
            node.clear_trace()
    
    # Decorator function for compiling log-posterior
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
    
    # Log likelihood functions
    
    @_add_to_post
    def categorical_like(self, x, probs, minval=0, step=1, name='categorical', prior=False):
        """Categorical log-likelihood. Accepts an array of probabilities associated with the histogram, the minimum value of the histogram (defaults to zero), and a step size (defaults to 1)."""
        
        x = ravel(x)
        
        # Normalize, if not already
        if sum(probs) != 1.0: probs = probs/sum(probs)
        
        if self._gof and not prior:
            
            try:
                self._like_names.append(name)
            except AttributeError:
                pass
            
            expval = sum([p*(minval + i*step) for i, p in enumerate(probs)])
            
            self._gof_loss.append(array([self.loss(x, expval), self.loss(rcategorical(probs, minval, step), expval)], dtype=float))
            
            try:
                return sum(map(_categorical, x, probs, minval, step))
            except TypeError:
                return _categorical(x, probs, minval, step)
    
    def categorical_prior(self, parameters, probs, minval=0, step=1):
        """Categorical prior distribution"""
        
        return self.categorical_like(parameters, probs, minval=0, step=1, prior=True)
    
    @_add_to_post
    def uniform_like(self, x, lower, upper, name='uniform', prior=False):
        """Beta log-likelihood"""
        
        if not shape(lower) == shape(upper): raise ParameterError, 'Parameters must have same dimensions in uniform(like)'
        
        # Allow for multidimensional arguments
        if ndim(lower) > 1:
            
            return sum(self.uniform_like(y, l, u, name=name, prior=prior) for y, l, u in zip(x, lower, upper))
        
        else:
            
            # Ensure valid values of parameters
            self.constrain(x, lower=lower, upper=upper)
            
            # Equalize dimensions
            x = ravel(x)
            lower = resize(lower, len(x))
            upper = resize(upper, len(x))
            
            # Goodness-of-fit
            if self._gof and not prior:
                
                try:
                    self._like_names.append(name)
                except AttributeError:
                    pass
                
                expval = (upper - lower) / 2.
                
                # Simulated values
                y = array(map(runiform, lower, upper))
                
                # Generate GOF points
                gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]), 0)
                
                self._gof_loss.append(gof_points)
            
            return sum(log(1. / (array(upper) - array(lower))))
    
    def uniform_prior(self, parameter, lower, upper):
        """Uniform prior distribution"""
        
        return self.uniform_like(parameter, lower, upper, prior=True)
    
    @_add_to_post
    def uniform_mixture_like(self, x, lower, median, upper, name='uniform_mixture', prior=False):
        """Uniform mixture log-likelihood
        
        This distribution is specified by three parameters (upper bound,
        median, lower bound), defining a mixture of 2 uniform distributions,
        that share the median as an upper (lower) bound. Hence, half of the
        density is in the first distribution and the other half in the second."""
        
        if not shape(lower) == shape(median) == shape(upper): raise ParameterError, 'Parameters must have same dimensions in uniform_mixture_like()'
        
        # Allow for multidimensional arguments
        if ndim(lower) > 1:
            
            return sum(self.uniform_mixture_like(y, l, m, u, name=name, prior=prior) for y, l, m, u in zip(x, lower, median, upper))
        
        else:
            
            # Ensure valid values of parameters
            self.constrain(median, lower, upper)
            self.constrain(x, lower, upper)
            
            # Equalize dimensions
            x = ravel(x)
            lower = resize(lower, len(x))
            median = resize(median, len(x))
            upper = resize(upper, len(x))
            
            # Goodness-of-fit
            if self._gof and not prior:
                
                try:
                    self._like_names.append(name)
                except AttributeError:
                    pass
                
                expval = 0.5 * (median - lower) + 0.5 * (upper - median)
                
                # Simulated values
                y = array(map(rmixeduniform, lower, median, upper))
                
                # Generate GOF points
                gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]), 0)
                
                self._gof_loss.append(gof_points)
            
            return sum(log((x <= median) * 0.5 / (median - lower)  +  (x > median) * 0.5 / (upper - median)))
    
    def uniform_mixture_prior(self, parameter, lower, median, upper):
        """Uniform mixture prior distribution"""
        
        return self.uniform_mixture_like(parameter, lower, median, upper, prior=True)
    
    @_add_to_post
    def beta_like(self, x, alpha, beta, name='beta', prior=False):
        """Beta log-likelihood"""
        
        if not shape(alpha) == shape(beta): raise ParameterError, 'Parameters must have same dimensions in beta_like()'
        
        # Allow for multidimensional arguments
        if ndim(alpha) > 1:
            
            return sum(self.beta_like(y, a, b, name=name, prior=prior) for y, a, b in zip(x, alpha, beta))
        
        else:
            
            # Ensure valid values of parameters
            self.constrain(alpha, lower=0)
            self.constrain(beta, lower=0)
            self.constrain(x, 0, 1)
            
            # Equalize dimensions
            x = ravel(x)
            alpha = resize(alpha, len(x))
            beta = resize(beta, len(x))
            
            # Goodness-of-fit
            if self._gof and not prior:
                
                try:
                    self._like_names.append(name)
                except AttributeError:
                    pass
                
                expval = 1.0 * alpha / (alpha + beta)
                
                # Simulated values
                y = array(map(rbeta, alpha, beta))
                
                # Generate GOF points
                gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]), 0)
                
                self._gof_loss.append(gof_points)
            
            return sum(map(fbeta, x, alpha, beta))
    
    def beta_prior(self, parameter, alpha, beta):
        """Beta prior distribution"""
        
        return self.beta_like(parameter, alpha, beta, prior=True)
    
    @_add_to_post
    def dirichlet_like(self, x, theta, name='dirichlet', prior=False):
        """Dirichlet log-likelihood"""
        
        # Allow for multidimensional arguments
        if ndim(theta) > 1:
            
            return sum(self.dirichlet_like(y, t, name=name, prior=prior) for y, t in zip(x, theta))
        
        else:
            
            # Ensure valid values of parameters
            self.constrain(theta, lower=0)
            self.constrain(x, lower=0)
            self.constrain(sum(x), upper=1)
            
            # Ensure proper dimensionality of parameters
            if not len(x) == len(theta): raise ParameterError, 'Data and parameters must have same length in dirichlet_like()'
            
            # Goodness-of-fit
            if self._gof and not prior:
                
                try:
                    self._like_names.append(name)
                except AttributeError:
                    pass
                
                sumt = sum(theta)
                expval = theta/sumt
                
                if len(x) > 1:
                    
                    # Simulated values
                    y = rdirichlet(theta)
                    
                    # Generate GOF points
                    gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]), 0)
                    
                    self._gof_loss.append(gof_points)
            
            return fdirichlet(x, theta)
    
    def dirichlet_prior(self, parameter, theta):
        """Dirichlet prior distribution"""
        
        return self.dirichlet_like(parameter, theta, prior=True)
    
    @_add_to_post
    def dirichlet_multinomial_like(self, x, theta, name='dirichlet_multinomial', prior=False):
        """Dirichlet multinomial log-likelihood"""
        
        # Allow for multidimensional arguments
        if ndim(theta) > 1:
            
            return sum(self.dirichlet_multinomial_like(y, t, name=name, prior=prior) for y, t in zip(x, theta))
        
        else:
            
            # Ensure valid values of parameters
            self.constrain(theta, lower=0)
            self.constrain(x, lower=0)
            
            # Ensure proper dimensionality of parameters
            if not len(x) == len(theta): raise ParameterError, 'Data and parameters must have same length in dirichlet_multinomial_like()'
            
            # Goodness-of-fit
            if self._gof and not prior:
                
                try:
                    self._like_names.append(name)
                except AttributeError:
                    pass
                
                sumt = sum(theta)
                expval = theta/sumt
                
                if len(x) > 1:
                    
                    # Simulated values
                    y = rdirmultinom(theta, sum(x))
                    
                    # Generate GOF points
                    gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]), 0)
                    
                    self._gof_loss.append(gof_points)
            
            return fdirmultinom(x, theta)
    
    def dirichlet_multinomial_prior(self, parameter, theta):
        """Dirichlet multinomial prior distribution"""
        
        return self.dirichlet_multinomial_like(parameter, theta, prior=True)
    
    @_add_to_post
    def negative_binomial_like(self, x, mu, alpha, name='negative_binomial', prior=False):
        """Negative binomial log-likelihood"""
        
        if not shape(mu) == shape(alpha): raise ParameterError, 'Parameters must have same dimensions'
        
        # Allow for multidimensional arguments
        if ndim(mu) > 1:
            
            return sum(self.negative_binomial_like(y, _l, _p, name=name, prior=prior) for y, _l, _p in zip(x, mu, alpha))
        
        else:
            
            # Ensure valid values of parameters
            self.constrain(mu, lower=0)
            self.constrain(alpha, lower=0)
            self.constrain(x, lower=0, allow_equal=True)
            
            # Enforce array type
            x = ravel(x)
            mu = resize(mu, shape(x))
            alpha = resize(alpha, shape(x))
            
            # Goodness-of-fit
            if self._gof and not prior:
                
                try:
                    self._like_names.append(name)
                except AttributeError:
                    pass
                
                expval = mu
                
                # Simulated values
                y = array(map(rnegbin, alpha, alpha / (mu + alpha)))
                
                # Generate GOF points
                gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]), 0)
                
                self._gof_loss.append(gof_points)
            
            return fnegbin2(x, mu, alpha)
    
    def negative_binomial_prior(self, parameter, lamda, omega):
        """Negative binomial prior distribution"""
        
        return self.negative_binomial_like(parameter, lamda, omega, prior=True)
        
    @_add_to_post
    def geometric_like(self, x, p, name='geometric', prior=False):
        """Geometric log-likelihood
        
        Pr(X=k) = (1 - p)^(k-1) * p
        """
        
        # Allow for multidimensional arguments
        if ndim(p) > 1:
            
            return sum(self.geometric_like(y, q, name=name, prior=prior) for y, q in zip(x, p))
        
        else:
            
            # Ensure valid values of parameters
            self.constrain(p, 0, 1)
            self.constrain(x, lower=0)
            
            # Enforce array type
            x = ravel(x)
            p = resize(p, shape(x))
            
            # Goodness-of-fit
            if self._gof and not prior:
                
                try:
                    self._like_names.append(name)
                except AttributeError:
                    pass
                
                expval = p ** -1
                
                # Simulated values
                y = rgeometric(p)
                
                # Generate GOF points
                gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]), 0)
                
                self._gof_loss.append(gof_points)
            
            return fgeometric(x, p)
    
    def geometric_prior(self, parameter, p):
        """Geometric prior distribution"""
        
        return self.geometric_like(parameter, p, prior=True)
    
    @_add_to_post
    def hypergeometric_like(self, x, n, m, N, name='hypergeometric', prior=False):
        """
        Hypergeometric log-likelihood
        
        Distribution models the probability of drawing x successful draws in n
        draws from N total balls of which m are successes.
        """
        
        if not shape(n) == shape(m) == shape(N): raise ParameterError, 'Parameters must have same dimensions'
        
        # Allow for multidimensional arguments
        if ndim(n) > 1:
            
            return sum(self.hypergeometric_like(y, _n, _m, _N, name=name, prior=prior) for y, _n, _m, _N in zip(x, n, m, N))
        
        else:
            
            # Ensure valid values of parameters
            self.constrain(m, upper=N)
            self.constrain(n, upper=N)
            self.constrain(x, max(0, n - N + m), min(m, n))
            
            # Enforce array type
            x = ravel(x)
            n = resize(n, shape(x))
            m = resize(m, shape(x))
            N = resize(N, shape(x))
            
            # Goodness-of-fit
            if self._gof and not prior:
                
                try:
                    self._like_names.append(name)
                except AttributeError:
                    pass
                
                expval = n * (m / N)
                
                # Simulated values
                y = array(map(rhyperg, n, m, N))
                
                # Generate GOF points
                gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]), 0)
                
                self._gof_loss.append(gof_points)
            
            return sum(map(fhyperg, x, n, m, N))
    
    def hypergeometric_prior(self, parameter, n, m, N):
        """Hypergeometric prior distribution"""
        
        return self.hypergeometric_like(parameter, n, m, N, prior=True)
    
    @_add_to_post
    def multivariate_hypergeometric_like(self, x, m, name='multivariate_hypergeometric', prior=False):
        """Multivariate hypergeometric log-likelihood"""
        
        # Allow for multidimensional arguments
        if ndim(m) > 1:
            
            return sum(self.multivariate_hypergeometric_like(y, _m, name=name, prior=prior) for y, _m in zip(x, m))
        
        else:
            
            # Ensure valid parameter values
            self.constrain(x, upper=m)
            
            n = sum(x)
            N = sum(m)
            
            # Goodness-of-fit
            if self._gof and not prior:
                
                try:
                    self._like_names.append(name)
                except AttributeError:
                    pass
                
                expval = n * (array(m) / N)
                
                if ndim(x) > 1:
                    
                    # Simulated values
                    y = rmvhyperg(n, m)
                    
                    # Generate GOF points
                    gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]), 0)
                    
                    self._gof_loss.append(gof_points)
            
            return fmvhyperg(x, m)
    
    def multivariate_hypergeometric_prior(self, parameter, m):
        """Multivariate hypergeometric prior distribution"""
        
        return self.multivariate_hypergeometric_like(parameter, m, prior=True)
    
    @_add_to_post
    def binomial_like(self, x, n, p, name='binomial', prior=False):
        """Binomial log-likelihood"""
        
        if not shape(n) == shape(p): raise ParameterError, 'Parameters must have same dimensions'
        
        if ndim(n) > 1:
            
            return sum(self.binomial_like(y, _n, _p, name=name, prior=prior) for y, _n, _p in zip(x, n, p))
        
        else:
            
            # Ensure valid values of parameters
            self.constrain(p, 0, 1)
            self.constrain(n, lower=x, allow_equal=True)
            self.constrain(x, 0, allow_equal=True)
            
            # Enforce array type
            x = ravel(x)
            p = resize(p, shape(x))
            n = resize(n, shape(x))
            
            # Goodness-of-fit
            if self._gof and not prior:
                
                try:
                    self._like_names.append(name)
                except AttributeError:
                    pass
                
                expval = p * n

                # Simulated values
                y = array(map(rbinomial, n, p))
                
                # Generate GOF points
                gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]), 0)
                
                self._gof_loss.append(gof_points)
            
            return sum(map(fbinomial, x, n, p))
    
    def binomial_prior(self, parameter, n, p):
        """Binomial prior distribution"""
        
        return self.binomial_like(parameter, n, p, prior=True)
    
    @_add_to_post
    def bernoulli_like(self, x, p, name='bernoulli', prior=False):
        """Bernoulli log-likelihood"""
        
        if ndim(p) > 1:
            
            return sum(self.bernoulli_like(y, _p, name=name, prior=prior) for y, _p in zip(x, p))
        
        else:
            
            # Ensure valid values of parameters
            self.constrain(p, 0, 1)
            self.constrain(x, 0, 1, allow_equal=True)
            
            # Enforce array type
            x = ravel(x)
            p = resize(p, shape(x))
            
            # Goodness-of-fit
            if self._gof and not prior:
                
                try:
                    self._like_names.append(name)
                except AttributeError:
                    pass
                
                expval = p
                
                # Simulated values
                y = array([rbinomial(1, _p) for _p in p])
                
                # Generate GOF points
                gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]), 0)
                
                self._gof_loss.append(gof_points)
            
            return sum(map(fbernoulli, x, p))
    
    def bernoulli_prior(self, parameter, p):
        """Bernoulli prior distribution"""
        
        return self.bernoulli_like(parameter, p, prior=True)
    
    @_add_to_post
    def multinomial_like(self, x, n, p, name='multinomial', prior=False):
        """Multinomial log-likelihood with k-1 bins"""
        
        if ndim(n) > 1:
            
            return sum(self.multinomial_like(y, _n, _p, name=name, prior=prior) for y, _n, _p in zip(x, n, p))
        
        else:
            
            # Ensure valid parameter values
            self.constrain(p, lower=0)
            self.constrain(x, lower=0)
            self.constrain(sum(p), upper=1)
            self.constrain(sum(x), upper=n)
            
            # Goodness-of-fit
            if self._gof and not prior:
                
                try:
                    self._like_names.append(name)
                except AttributeError:
                    pass
                
                expval = array([pr * n for pr in p])
                
                # Simulated values
                y = rmultinomial(n, p)
                
                # Generate GOF points
                gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]), 0)
                
                self._gof_loss.append(gof_points)
            
            return fmultinomial(x, n, p)
    
    def multinomial_prior(self, parameter, n, p):
        """Beta prior distribution"""
        
        return self.multinomial_like(parameter, n, p, prior=True)
    
    @_add_to_post
    def poisson_like(self, x, mu, name='poisson', prior=False):
        """Poisson log-likelihood"""
        
        if ndim(mu) > 1:
            
            return sum(self.poisson_like(y, m, name=name, prior=prior) for y, m in zip(x, mu))
        
        else:
            
            # Ensure valid values of parameters
            self.constrain(x, lower=0)
            self.constrain(mu, lower=0)
            
            # Enforce array type
            x = ravel(x)
            mu = resize(mu, shape(x))
            
            # Goodness-of-fit
            if self._gof and not prior:
                
                try:
                    self._like_names.append(name)
                except AttributeError:
                    pass
                
                expval = mu
                
                # Simulated values
                y = array([rpoisson(a) for a in mu])
                
                # Generate GOF points
                gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]), 0)
                
                self._gof_loss.append(gof_points)
            
            like = sum(map(fpoisson, x, mu))
            
            return like
    
    def poisson_prior(self, parameter, mu):
        """Poisson prior distribution"""
        
        return self.poisson_like(parameter, mu, prior=True)
    
    @_add_to_post
    def gamma_like(self, x, alpha, beta, name='gamma', prior=False):
        """Gamma log-likelihood"""
        
        if not shape(alpha) == shape(beta): raise ParameterError, 'Parameters must have same dimensions in gamma_like()'
        
        # Allow for multidimensional arguments
        if ndim(alpha) > 1:
            
            return sum(self.gamma_like(y, a, b, name=name, prior=prior) for y, a, b in zip(x, alpha, beta))
        
        # Ensure valid values of parameters
        self.constrain(x, lower=0)
        self.constrain(alpha, lower=0)
        self.constrain(beta, lower=0)
        
        # Ensure proper dimensionality of parameters
        x = ravel(x)
        alpha = resize(alpha, shape(x))
        beta = resize(beta, shape(x))
        
        # Goodness-of-fit
        if self._gof and not prior:
            
            try:
                self._like_names.append(name)
            except AttributeError:
                pass
            
            # This is the EV for the RandomArray parameterization
            # in which beta is inverse
            expval = array(alpha) / beta
            
            ibeta = 1. / array(beta)
            
            # Simulated values
            y = array(map(rgamma, ibeta, alpha))
            
            # Generate GOF points
            gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]), 0)
            
            self._gof_loss.append(gof_points)
        
        return sum(map(fgamma, x, alpha, beta))
    
    def gamma_prior(self, parameter, alpha, beta):
        """Gamma prior distribution"""
        
        return self.gamma_like(parameter, alpha, beta, prior=True)
    
    @_add_to_post
    def chi2_like(self, x, df, name='chi_squared', prior=False):
        """Chi-squared log-likelihood"""
        
        if ndim(df) > 1:
            
            return sum(self.chi2_like(y, d, name=name, prior=prior) for y, d in zip(x, df))
        
        else:
            
            # Ensure valid values of parameters
            self.constrain(x, lower=0)
            self.constrain(df, lower=0)
            
            # Ensure array type
            x = ravel(x)
            df = resize(df, shape(x))
            
            # Goodness-of-fit
            if self._gof and not prior:
                
                try:
                    self._like_names.append(name)
                except AttributeError:
                    pass
                
                expval = df
                
                # Simulated values
                y = array([rchi2(d) for d in df])
                
                # Generate GOF points
                gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]), 0)
                
                self._gof_loss.append(gof_points)
            
            return sum(map(fgamma, x, 0.5*df, [2]*len(x)))
    
    def chi2_prior(self, parameter, df):
        """Chi-squared prior distribution"""
        
        return self.chi2_like(parameter, df, prior=True)
    
    @_add_to_post
    def inverse_gamma_like(self, x, alpha, beta, name='inverse_gamma', prior=False):
        """Inverse gamma log-likelihood"""
        
        if not shape(alpha) == shape(beta): raise ParameterError, 'Parameters must have same dimensions in gamma_like()'
        
        # Allow for multidimensional arguments
        if ndim(alpha) > 1:
            
            return sum(self.inverse_gamma_like(y, a, b, name=name, prior=prior) for y, a, b in zip(x, alpha, beta))
        else:
            
            # Ensure valid values of parameters
            self.constrain(x, lower=0)
            self.constrain(alpha, lower=0)
            self.constrain(beta, lower=0)
            
            # Ensure proper dimensionality of parameters
            x = ravel(x)
            alpha = resize(alpha, shape(x))
            beta = resize(beta, shape(x))
            
            # Goodness-of-fit
            if self._gof and not prior:
                
                try:
                    self._like_names.append(name)
                except AttributeError:
                    pass
                
                expval = array(alpha) / beta
                
                ibeta = 1. / array(beta)
                
                # Simulated values
                y = array(map(rgamma, ibeta, alpha))
                
                # Generate GOF points
                gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]), 0)
                
                self._gof_loss.append(gof_points)
            
            return sum(map(figamma, x, alpha, beta))
    
    def inverse_gamma_prior(self, parameter, alpha, beta):
        """Inverse gamma prior distribution"""
        
        return self.inverse_gamma_like(parameter, alpha, beta, prior=True)
    
    @_add_to_post
    def exponential_like(self, x, beta, name='exponential', prior=False):
        """Exponential log-likelihood"""
        
        # Allow for multidimensional arguments
        if ndim(beta) > 1:
            
            return sum(self.exponential_like(y, b, name=name, prior=prior) for y, b in zip(x, beta))
        
        else:
            
            # Ensure valid values of parameters
            self.constrain(x, lower=0)
            self.constrain(beta, lower=0)
            
            # Ensure proper dimensionality of parameters
            x = ravel(x)
            beta = resize(beta, shape(x))
            
            # Goodness-of-fit
            if self._gof and not prior:
                
                try:
                    self._like_names.append(name)
                except AttributeError:
                    pass
                
                expval = beta
                
                ibeta = 1./array(beta)
                
                # Simulated values
                y = array([rexponential(b) for b in ibeta])
                
                # Generate GOF points
                gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]), 0)
                
                self._gof_loss.append(gof_points)
            
            return sum(map(fgamma, x, [1]*len(x), beta))
    
    def exponential_prior(self, parameter, beta):
        """Exponential prior distribution"""
        
        return self.exponential_like(parameter, beta, prior=True)
    
    @_add_to_post
    def normal_like(self, x, mu, tau, name='normal', prior=False):
        """Normal log-likelihood"""
        
        if not shape(mu) == shape(tau): raise ParameterError, 'Parameters must have same dimensions in normal_like()'
        
        if ndim(mu) > 1:
            
            return sum(self.normal_like(y, m, t, name=name, prior=prior) for y, m, t in zip(x, mu, tau))
        
        else:
            
            # Ensure valid values of parameters
            self.constrain(tau, lower=0)
            
            # Ensure array type
            x = ravel(x)
            mu = resize(mu, shape(x))
            tau = resize(tau, shape(x))
            
            # Goodness-of-fit
            if self._gof and not prior:
                
                try:
                    self._like_names.append(name)
                except AttributeError:
                    pass
                
                expval = mu
                
                sigma = sqrt(1. / array(tau))
                
                # Simulated values
                y = array(map(rnormal, mu, sigma))
                
                # Generate GOF points
                gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]), 0)
                
                self._gof_loss.append(gof_points)
            
            return sum(map(fnormal, x, mu, tau))
    
    def normal_prior(self, parameter, mu, tau):
        """Normal prior distribution"""
        
        return self.normal_like(parameter, mu, tau, prior=True)
        
    @_add_to_post
    def half_normal_like(self, x, tau, name='halfnormal', prior=False):
        """Half-normal log-likelihood"""
        
        if ndim(tau) > 1:
            
            return sum(self.half_normal_like(y, t, name=name, prior=prior) for y, t in zip(x, tau))
        
        else:
            
            # Ensure valid values of parameters
            self.constrain(tau, lower=0)
            self.constrain(x, lower=0)
            
            # Ensure array type
            x = ravel(x)
            tau = resize(tau, shape(x))
            
            # Goodness-of-fit
            if self._gof and not prior:
                
                try:
                    self._like_names.append(name)
                except AttributeError:
                    pass
                
                expval = sqrt(0.5 * pi / array(tau))
                
                sigma = sqrt(1. / tau)
                
                # Simulated values
                y = absolute([rnormal(0, sig) for sig in sigma])
                
                # Generate GOF points
                gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]), 0)
                
                self._gof_loss.append(gof_points)
            
            return sum(map(fhalfnormal, x, tau))
    
    def half_normal_prior(self, parameter, tau):
        """Half-normal prior distribution"""
        
        return self.half_normal_like(parameter, tau, prior=True)
    
    @_add_to_post
    def lognormal_like(self, x, mu, tau, name='lognormal', prior=False):
        """Log-normal log-likelihood"""
        
        if not shape(mu) == shape(tau): raise ParameterError, 'Parameters must have same dimensions in lognormal_like()'
        
        if ndim(mu) > 1:
            
            return sum(self.lognormal_like(y, m, t, name=name, prior=prior) for y, m, t in zip(x, mu, tau))
        
        else:
            
            # Ensure valid values of parameters
            self.constrain(tau, lower=0)
            self.constrain(x, lower=0)
            
            # Ensure array type
            x = ravel(x)
            mu = resize(mu, shape(x))
            tau = resize(tau, shape(x))
            
            # Goodness-of-fit
            if self._gof and not prior:
                
                try:
                    self._like_names.append(name)
                except AttributeError:
                    pass
                
                expval = exp(mu + (0.5 / tau))
                
                sigma = sqrt(1. / array(tau))
                
                # Simulated values
                y = exp(map(rnormal, mu, sigma))
                
                # Generate GOF points
                gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]), 0)
                
                self._gof_loss.append(gof_points)
            
            return sum(map(flognormal, x, mu, tau))
    
    def lognormal_prior(self, parameter, mu, tau):
        """Log-normal prior distribution"""
        
        return self.lognormal_like(parameter, mu, tau, prior=True)
    
    @_add_to_post
    def multivariate_normal_like(self, x, mu, tau, name='multivariate_normal', prior=False):
        """Multivariate normal"""
        
        if ndim(tau) > 2:
            
            return sum(self.multivariate_normal_like(y, m, t, name=name, prior=prior) for y, m, t in zip(x, mu, tau))
        
        else:
            
            # Ensure valid values of parameters
            self.constrain(diagonal(tau), lower=0)
            
            # Goodness-of-fit
            if self._gof and not prior:
                
                try:
                    self._like_names.append(name)
                except AttributeError:
                    pass
                
                expval = array(mu)
                
                if ndim(x) > 1:
                    
                    # Simulated values
                    y = rmvnormal(mu, inverse(tau))
                    
                    # Generate GOF points
                    gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]), 0)
                    
                    self._gof_loss.append(gof_points)
            
            return fmvnorm(x, mu, tau)
    
    # Deprecated name
    mvnormal_like = multivariate_normal_like
    
    def multivariate_normal_prior(self, parameter, mu, tau):
        """Multivariate normal prior distribution"""
        
        return self.multivariate_normal_like(parameter, mu, tau, prior=True)
    
    @_add_to_post
    def wishart_like(self, X, n, Tau, name='wishart', prior=False):
        """Wishart log-likelihood"""
        
        if ndim(Tau) > 2:
            
            return sum(self.wishart_like(x, m, t, name=name, prior=prior) for x, m, t in zip(X, n, Tau))
        
        else:
            
            # Ensure valid values of parameters
            self.constrain(diagonal(Tau), lower=0)
            self.constrain(n, lower=0)
            
            # Goodness-of-fit
            if self._gof and not prior:
                
                try:
                    self._like_names.append(name)
                except AttributeError:
                    pass
                
                expval = n * array(Tau)
                
                if ndim(x) > 1:
                    
                    # Simulated values
                    y = rwishart(n, inverse(Tau))
                    
                    # Generate GOF points
                    gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]), 0)
                    
                    self._gof_loss.append(gof_points)
            
            return fwishart(X, n, Tau)
    
    def wishart_prior(self, parameter, n, Tau):
        """Wishart prior distribution"""
        
        return self.wishart_like(parameter, n, Tau, prior=True)
    
    @_add_to_post
    def weibull_like(self, x, alpha, beta, name='weibull', prior=False):
        """Weibull log-likelihood"""
        
        if not shape(alpha) == shape(beta): raise ParameterError, 'Parameters must have same dimensions in gamma_like()'
        
        # Allow for multidimensional arguments
        if ndim(alpha) > 1:
            
            return sum(self.weibull_like(y, a, b, name=name, prior=prior) for y, a, b in zip(x, alpha, beta))
        
        else:
            
            # Ensure valid values of parameters
            self.constrain(alpha, lower=0)
            self.constrain(beta, lower=0)
            self.constrain(x, lower=0)
            
            # Ensure proper dimensionality of parameters
            x = ravel(x)
            alpha = resize(alpha, shape(x))
            beta = resize(beta, shape(x))
            
            # Goodness-of-fit
            if self._gof and not prior:
                
                try:
                    self._like_names.append(name)
                except AttributeError:
                    pass
                
                expval = beta * [gamfun((a + 1) / a) for a in alpha]
                
                # Simulated values
                y = beta * (-log(runiform(0, 1, len(x))) ** (1. / alpha))
                
                # Generate GOF points
                gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]), 0)
                
                self._gof_loss.append(gof_points)
            
            return sum(map(fweibull, x, alpha, beta))
    
    def weibull_prior(self, parameter, alpha, beta):
        """Weibull prior distribution"""
        
        return self.weibull_like(parameter, alpha, beta, prior=True)
    
    @_add_to_post
    def cauchy_like(self, x, alpha, beta, name='cauchy', prior=False):
        """Cauchy log-likelhood"""
        
        if not shape(alpha) == shape(beta): raise ParameterError, 'Parameters must have same dimensions in gamma_like()'
        
        # Allow for multidimensional arguments
        if ndim(alpha) > 1:
            
            return sum(self.cauchy_like(y, a, b, name=name, prior=prior) for y, a, b in zip(x, alpha, beta))
        
        else:
            
            # Ensure valid values of parameters
            self.constrain(beta, lower=0)
            
            # Ensure proper dimensionality of parameters
            x = ravel(x)
            alpha = resize(alpha, shape(x))
            beta = resize(beta, shape(x))
            
            # Goodness-of-fit
            if self._gof and not prior:
                
                try:
                    self._like_names.append(name)
                except AttributeError:
                    pass
                
                expval = alpha
                
                # Simulated values
                y = array(map(rcauchy, alpha, beta))
                
                # Generate GOF points
                gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]), 0)
                
                self._gof_loss.append(gof_points)
            
            return sum(map(fcauchy, x, alpha, beta))
    
    def cauchy_prior(self, parameter, alpha, beta):
        """Cauchy prior distribution"""
        
        return self.cauchy_like(parameter, alpha, beta, prior=True)
    
    def constrain(self, value, lower=-inf, upper=inf, allow_equal=True):
        """Apply interval constraint on parameter value"""
        
        value = ravel(value)
        
        if allow_equal:
            if any(lower > value) or any(value > upper):
            
                raise LikelihoodError
            
        elif any(lower >= value) or any(value >= upper):
            
            raise LikelihoodError
    
    def parameter(self, name, init_val, discrete=False, binary=False, dist='normal', scale=None, random=False, plot=True):
        """Create a new parameter"""
        
        if binary:
            self.parameters[name] = BinaryParameter(name, init_val, sampler=self, plot=plot, random=random)
        elif discrete:
            self.parameters[name] = DiscreteParameter(name, init_val, sampler=self, dist=dist, scale=scale, plot=plot, random=random)
        else:
            self.parameters[name] = Parameter(name, init_val, sampler=self, dist=dist, scale=scale, plot=plot, random=random)
    
    def node(self, name, shape=None, plot=True):
        """Create a new node"""
        
        self.nodes[name] = Node(name, self, shape=shape, plot=plot)
    
    def summary(self, burn=0, thin=1, alpha=0.05, composite=False):
        """Generates summary statistics"""
        
        # Initialize dictionary
        results = {}
        
        # Retrieve deviance node
        deviance = self.nodes['deviance']
        
        # Count parameters
        k = 0
        for parameter in self.parameters:
            # Only count non-random-effect parameters
            if not self.parameters[parameter].random:
                try:
                    k += len(ravel(self.__dict__[parameter]))
                except TypeError:
                    k += 1
        
        # Calculate AIC quantiles
        aic_quantiles = deviance.quantiles(burn=burn, thin=thin, composite=composite)
        for q in aic_quantiles:
            aic_quantiles[q] += 2*k
        
        try:
            results['AIC'] = {
                'n': len(deviance.get_trace(burn=burn, thin=thin, composite=composite)),
                'standard deviation': deviance.std(burn=burn, thin=thin, composite=composite),
                'mean': deviance.mean(burn=burn, thin=thin, composite=composite) + 2*k,
                '%s%s HPD interval' % (int(100*(1-alpha)),'%'): [d + 2*k for d in deviance.hpd(alpha=alpha, burn=burn, thin=thin, composite=composite)],
                'mc error': deviance.mcerror(burn=burn, thin=thin, composite=composite),
                'quantiles': aic_quantiles
            }
        except TypeError:
            print "Could not generate summary of AIC"
        
        # Loop over parameters and nodes
        for p in self.parameters.values()+self.nodes.values():
            try:
                results[p.name] = {
                    'n': len(p.get_trace(burn=burn, thin=thin, composite=composite)),
                    'standard deviation': p.std(burn=burn, thin=thin, composite=composite),
                    'mean': p.mean(burn=burn, thin=thin, composite=composite),
                    '%s%s HPD interval' % (int(100*(1-alpha)),'%'): p.hpd(alpha=alpha, burn=burn, thin=thin, composite=composite),
                    'mc error': p.mcerror(burn=burn, thin=thin, composite=composite),
                    'quantiles': p.quantiles(burn=burn, thin=thin, composite=composite)
                }
            except AttributeError:
                print "Could not find", p.name
            except TypeError:
                print "Could not generate summary of", p.name
        
        results['DIC'] = self.calculate_dic(burn, thin, composite=composite)
        
        return results
    
    def export(self, summary, filename='pymc', outfile=None):
        """Exports summary statistics to CSV file"""
        
        # Exit if nothing to export
        if not summary: return
        
        # Open output file
        output_file = outfile or open(filename+'.csv','w')
        
        # Build list of column names
        columns = ['n','mean','standard deviation','mc error']
        
        # Pull sample node
        sample_item = summary[summary.keys()[0]]
        if not shape(sample_item):
            # Incase DIC was selected
            sample_item = summary[summary.keys()[-1]]
        
        # Build header
        header = 'name'
        for i in columns:
            header = '%s,%s' % (header,i)
        
        # Find HPD interval header
        for label in sample_item:
            if label.endswith('interval'):
                header += ',lower %s,upper %s' % (label[:3],label[:3])
                columns.append(label)
        
        # Quantile headers
        quantiles = sample_item['quantiles']
        qkeys = quantiles.keys()
        qkeys.sort()
        
        for i in qkeys:
            header = '%s,%s%s' % (header,i,'%')
        
        # Write header
        output_file.write(header+'\n')
        
        columns.append('quantiles')
        
        # Iterate over summary output
        for name in summary:
            
            # Skip DIC
            if name=='DIC': continue
            
            # Retrieve node
            node = summary[name]
            
            # Account for array nodes
            dims = shape(node['mean'])
            
            if dims:
                
                for index in make_indices(dims):
                    
                    row = name
                    
                    try:
                        for i in index:
                            row += '_%s' % i
                    except TypeError:
                        row += '_%s' % index
                    
                    # Loop over columns
                    for c in columns:
                        
                        if c=='n':
                            row = '%s,%s' % (row,node[c])
                        elif c=='quantiles':
                            quantiles = node[c]
                            for k in qkeys:
                                row = '%s,%s' % (row,quantiles[k][index])
                        elif c.endswith('interval'):
                            
                            row = '%s,%s,%s' % (row,node[c][index][0],node[c][index][1])
                        else:
                            row = '%s,%s' % (row,node[c][index])
                    
                    # Write row to file
                    output_file.write(row+'\n')
            
            # Univariate nodes
            else:
                
                row = name
                
                # Loop over columns
                for c in columns:
                    
                    if c=='quantiles':
                        quantiles = node[c]
                        for k in qkeys:
                            row = '%s,%s' % (row,quantiles[k])
                    elif c.endswith('interval'):
                        row = '%s,%s,%s' % (row,node[c][0],node[c][1])
                    else:
                        row = '%s,%s' % (row,node[c])
                
                # Write row to file
                output_file.write(row+'\n')
        
        # Write DIC to its own cell
        output_file.write('\nDIC\n%s' % summary['DIC'])
        
        # Close output file
        if not outfile: output_file.close()
    
    def coda_output(self, filename='coda', burn=0, thin=1):
        """Generate output files that are compatible with CODA"""
        
        print
        print "Generating CODA output"
        print '='*50
        
        # Open trace file
        trace_file = open(filename+'.out', 'w')
        
        # Open index file
        index_file = open(filename+'.ind', 'w')
        
        # Initialize index
        index = 1
        
        # Loop over all parameters
        for p in self.parameters.values()+self.nodes.values():
            
            print "Processing", p.name
            
            for c in range(p.trace_count()):
                
                index = self._process_trace(trace_file, index_file, p.get_trace(burn, thin, chain=c), '%s%i' % (p.name, c), index)
        
        # Close files
        trace_file.close()
        index_file.close()
        
    def output(self, filename='coda', burn=0, thin=1):
        """Deprecated method name"""
        
        print "output() is deprecated; please use coda_output()"
        
        self.coda_output(filename, burn, thin)
    
    def _process_trace(self, trace_file, index_file, trace, name, index):
        """Support function for output(); writes output to files"""
        
        if ndim(trace)>1:
            trace = swapaxes(trace, 0, 1)
            for i, seq in enumerate(trace):
                _name = '%s_%s' % (name, i)
                index = self._process_trace(trace_file, index_file, seq, _name, index)
        else:
            index_buffer = '%s\t%s\t' % (name, index)
            for i, val in enumerate(trace):
                trace_file.write('%s\t%s\r\n' % (i+1, val))
                index += 1
            index_file.write('%s%s\r\n' % (index_buffer, index-1))
        
        return index
    
    def print_summary(self, summary):
        """Print simulation results"""
        
        # Exit if nothing to export
        if not summary: return
        
        print
        print "Marginal Posterior Statistics"
        print '='*50
        
        keys = summary.keys()
        keys.sort()
        
        # Loop over nodes
        for name in keys:
            
            if name=='DIC': continue
            
            try:
                print 'Node:', name
                print
                parameter = summary[name]
                
                # Obtain parameter statistic names
                pkeys = parameter.keys()
                pkeys.sort()
                
                # Loop over statistics
                for label in pkeys:
                    # Special formatting for quantiles
                    if label=='quantiles':
                        quantiles = parameter[label]
                        print 'quantiles ='
                        qkeys = quantiles.keys()
                        qkeys.sort()
                        for q in qkeys:
                            print '\t%s:\t' % q,quantiles[q]
                    else:
                        print '%s = %s' % (label,parameter[label])
                
                print
                print '='*50
            except AttributeError:
                pass
        
        print 'DIC =', summary['DIC']
        print
        
    def get_value(self):
        # Returns last-calculated posterior (log scale) 
        
        try:
            return self._post
        except AttributeError:
            print 'Posterior has not yet been calculated'
        
    def __call__(self):
        # Initializes posterior return value, then calls model
        
        self._post = 0.0
        
        return self.model()

    def model(self):
        # To be specified in subclass
        
        # calculate_likelihood is deprecated (from version 1.0)
        return self.calculate_likelihood()
    
    def convergence(self, first=0.1, last=0.5, intervals=20, burn=0, thin=1, chain=-1, plot=True, color='b'):
        """Run convergence diagnostics"""
        
        print
        print "Running convergence diagnostics"
        print '='*50
        
        # Initialize dictionary scores
        zscores = {}
        
        # Plotting flag
        plotter = None
        if plot:
            plotter = self.plotter
        
        # Loop over all parameters and nodes
        for node in self.parameters.values():
            
            # Geweke diagnostic
            zscores.update(node.geweke(first=first, last=last, intervals=intervals, burn=burn, thin=thin, chain=chain, plotter=plotter, color=color))
        
        return zscores
    
    def goodness(self, iterations, plot=True, loss='squared', burn=0, thin=1, chain=-1, composite=False, color='b', filename='gof'):
        """
        Calculates Goodness-Of-Fit according to Brooks et al. 1998
        """
        
        print
        print "Goodness-of-fit"
        print '='*50
        print 'Generating %s goodness-of-fit simulations' % iterations
        
        
        # Specify loss function
        if loss=='squared':
            self.loss = squared_loss
        elif loss=='absolute':
            self.loss = absolute_loss
        elif loss=='chi-square':
            self.loss = chi_square_loss
        else:
            print 'Invalid loss function specified.'
            return
            
        # Open file for GOF output
        outfile = open(filename + '.csv', 'w')
        outfile.write('Goodness of Fit based on %s iterations\n' % iterations)
        
        # Empty list of GOF plot points
        D_points = []
        
        # Set GOF flag
        self._gof = True
        
        # List of names for conditional likelihoods
        self._like_names = []
        
        # Local variable for the same
        like_names = None
        
        # Generate specified number of points
        for i in range(iterations):
            
            valid_gof_points = False
            
            # Sometimes the likelihood bombs out and doesnt produce
            # GOF points
            while not valid_gof_points:
                
                # Initializealize list of GOF error loss values
                self._gof_loss = []
                
                # Loop over parameters
                for name in self.parameters:
                    
                    # Look up parameter
                    parameter = self.parameters[name]
                    
                    # Retrieve copy of trace
                    trace = parameter.get_trace(burn=burn, thin=thin, chain=chain, composite=composite)
                    
                    # Sample value from trace
                    sample = trace[random_integers(len(trace)) - 1]
                    
                    # Set current value to sampled value
                    parameter.set_value(sample)
                
                # Run calculate likelihood with sampled parameters
                try:
                    self()
                except (LikelihoodError, OverflowError, ZeroDivisionError):
                    # Posterior dies for some reason
                    pass
                
                try:
                    like_names = self._like_names
                    del(self._like_names)
                except AttributeError:
                    pass
                
                # Conform that length of GOF points is valid
                if len(self._gof_loss) == len(like_names):
                    valid_gof_points = True
            
            # Append points to list
            D_points.append(self._gof_loss)
        
        # Transpose and plot GOF points
        
        D_points = t([[y for y in x if shape(y)] for x in D_points], (1,2,0))
        
        # Keep track of number of simulation deviances that are
        # larger than the corresponding observed deviance
        sim_greater_obs = 0
        n = 0
        
        # Dictionary to hold GOF statistics
        stats = {}
        
        # Dictionary to hold GOF plot data
        plots = {}
        
        # Loop over the sets of points for plotting
        for name,points in zip(like_names,D_points):
            
            if plots.has_key(name):
                # Append points, if already exists
                plots[name] = concatenate((plots[name], points), 1)
            
            else:
                plots[name] = points
            
            count = sum(s>o for o,s in t(points))
            
            try:
                stats[name] += array([count,iterations])
            except KeyError:
                stats[name] = array([1.*count,iterations])
            
            sim_greater_obs += count
            n += iterations
        
        # Generate plots
        if plot:
            for name in plots:
                self.plotter.gof_plot(plots[name], name, color=color)
        
        # Report p(D(sim)>D(obs))
        for name in stats:
            num,denom = stats[name]
            print 'p( D(sim) > D(obs) ) for %s = %s' % (name,num/denom)
            outfile.write('%s,%f\n' % (name,num/denom))
        
        p = 1.*sim_greater_obs/n
        print 'Overall p( D(sim) > D(obs) ) =', p
        print
        outfile.write('overall,%f\n' % p)
        
        stats['overall'] = array([1.*sim_greater_obs,n])
        
        # Unset flag
        self._gof = False
        
        # Close output file
        outfile.close()
        
        return stats
    
    def autocorrelation(self, max_lag=100, burn=0, thin=1, chain=-1, color='b'):
        """Generates autocorrelation plots for all parameters and nodes"""
        
        print
        print "Plotting parameter autocorrelation"
        print '='*50
        
        # Loop over parameters
        for parameter in self.parameters.values():
            
            parameter.autocorrelation(max_lag=max_lag, burn=burn, thin=thin, chain=chain, plotter=self.plotter, color=color)
    
    def calculate_aic(self, deviance):
        """Calculates Akaikes Information Criterion (AIC)"""
        
        k = 0
        for parameter in self.parameters:
            if not self.parameters[parameter].random:
                try:
                    k += len(ravel(self.__dict__[parameter]))
                except TypeError:
                    k += 1
        
        try:
            return array([d + 2.*k for d in deviance])
        except TypeError:
            return deviance + 2*k
    
    def calculate_dic(self, burn=0, thin=1, composite=False):
        """Calculates deviance information Criterion"""
        
        # Set values of all parameters to their mean
        for name in self.parameters:
            
            # Calculate mean of paramter
            parameter = self.parameters[name]
            meanval = parameter.mean(burn=burn, thin=thin, composite=composite)
            
            # Set current value to mean
            parameter.set_value(meanval)
        
        mean_deviance = self.nodes['deviance'].mean(burn, thin, composite=composite)
        
        # Return twice deviance minus deviance at means
        try:
            self()
            return 2*mean_deviance + 2*self._post
        except LikelihoodError:
            return -inf



class Slice(Sampler):
    """
    Slice sampling MCMC algorithm (see R.M. Neal, Annals of Statistics,
    31, no. 3 (2003), 705-767).
    """
    
    def __init__(self, plot_format='png', plot_backend='TkAgg'):
        # Class initialization
        
        # Initialize superclass
        Sampler.__init__(self, plot_format, plot_backend)
    
    def sample(self, iterations=1000, burn=0, thin=1, verbose=True, plot=True, debug=False):
        """Sampling algorithm"""
        
        pass



class MetropolisHastings(Sampler):
    """
    This is the base class for an adaptive random walk Metropolis-Hastings
    MCMC sampling algorithm. Proposed values of each parameter in the model
    are generated by a random walk algorithm; these values are accepted or
    rejected according to Hastings' ratio:
    
    a(x, y) = p(y)q(y, x)/p(x)q(x, y)
    
    where p(y) is the conditional posterior probability of Y=y. In the case
    of a symmetric proposal distribution, this probability reduces to:
    
    a(x, y) = p(y)/p(x)
    
    Subclasses of MetropolisHastings need only specify relevant Parameters and
    Nodes in the __init__() method, as well as the model specification 
    method, model(). Parameter and Node objects created by the
    parameter() and node() methods, respectively, are automatically associated
    with the sampler, and therefore sampled automatically during the
    simulation.
    
    See the DisasterSampler subclass in this module for a simple
    example.
    
    """
    
    def __init__(self, plot_format='png', plot_backend='TkAgg'):
        """Class initializer"""
        
        # Initialize superclass
        Sampler.__init__(self, plot_format, plot_backend)
    
    def test(self):
        """Accept or reject proposed parameter values"""
        
        try:
            self()
        except (LikelihoodError, OverflowError, ZeroDivisionError):
            return False
        
        # Reject bogus results
        if str(self._post) == 'nan' or self._post == -inf:
            return False
        
        # Difference of log likelihoods
        alpha = self._post - self._last_post
        
        # Accept
        try:
            if alpha >= 0 or random_number() <= exp(alpha):
                self._last_post = self._post
                return True
        except (ValueError, OverflowError):
            pass
        # Reject
        return False
    
    def sample(self, iterations, burn=0, thin=1, tune=True, tune_interval=100, divergence_threshold=1e10, verbose=True, plot=True, color='b', debug=False):
        """Sampling algorithm"""
        
        if iterations <= burn :
            raise TypeError, 'iterations must be greater than burn.'
        
        # Tuning flag
        tuning = tune
        
        # Initialize rejection counter
        self._rejected = 0
        
        # Initialize divergence counter
        infinite_deviance_count = 0
        
        if debug: pdb.set_trace()
        
        # Initialize model
        try:
            self()
            self._last_post = self._post
        except (LikelihoodError, OverflowError, ZeroDivisionError):
            self._last_post = -inf
        
        # Initialize traces of each parameter and node
        for node in self.parameters.values() + self.nodes.values():
            try:
                node.init_trace(concatenate(([iterations], shape(node.get_value()))))
            except KeyError:
                node.init_trace(iterations)
        
        try:
            
            start = time.time()
            
            clean = 0
            
            for iteration in range(iterations):
                
                # Feedback every 100 iterations
                if not iteration%100:
                    print '\nIteration', iteration, 'at', time.time() - start
                    
                    # Check for divergent model
                    if iteration and abs(self.deviance) is inf:
                        
                        # If this is the third consecutive check
                        # with infinite deviance, abort
                        if infinite_deviance_count == 2:
                            raise DivergenceError
                        
                        # Increment deviance counter
                        infinite_deviance_count += 1
                    
                    else: infinite_deviance_count = 0
                
                # Halt tuning
                if iteration == burn:
                    print "*** Burn-in period expired ***"
                    tuning = False
                
                # Adapt sampling distribution at appropriate intervals
                if tuning and iteration % tune_interval == 0 and iteration:
                    
                    # Negate tuning flag
                    still_tuning = 0
                    
                    # Adapt each parameter
                    for parameter in self.parameters.values():
                        # Check to see if this parameter is still adapting
                        still_tuning += parameter.tune(tune_interval, divergence_threshold, verbose)
                    
                    if verbose:
                        for node in self.nodes.values():
                            print '\nNode %s current value: %s' % (node.name,node.get_value())
                    
                    # Halt tuning if all parameters are tuned
                    if not still_tuning:
                        # Increment the number of clean intervals
                        clean += 1
                        # If there has been 5 consecutive clean intervals, halt tuning
                        if clean == 5:
                            tuning = False
                            print "*** Finished tuning proposals ***"
                    else:
                        clean = 0
                
                # Sample each parameter in turn
                for parameter in self.parameters.values():
                    
                    # New value of parameter
                    parameter.propose(debug)
                    
                    # Tally current value of parameter
                    parameter.tally(iteration)
                
                # Calculate deviance, given new parameters and node values
                # Likelihood must be recalculated to guarantee valid node
                # values.
                try:
                    self()
                    self.deviance = -2 * self._post
                except (LikelihoodError, OverflowError, ZeroDivisionError):
                    self.deviance = -inf
                
                # Tally current value of nodes
                for node in self.nodes.values():
                    node.tally(iteration)
        
        
        
            # Generate summary
            results = self.summary(burn=burn, thin=thin)
        
            # Generate output
            if verbose:
                self.print_summary(results)
        
            # Plot if requested
            if plot:
                # Loop over parameters and nodes
                for p in self.parameters.values()+self.nodes.values():
                    p.plot(self.plotter, burn=burn, thin=thin, color=color)
        
            return results

        except KeyboardInterrupt:
            # Stopped by hand
            pass
        
        except DivergenceError:
            # If model has diverged, stop the model
            
            print "Divergent model. Aborting run."
            
            # Remove remnant trace from each node
            for node in self.parameters.values()+self.nodes.values():
                node.clear_trace()
            
            return

"""Unit testing code"""


class DisasterSampler(MetropolisHastings):
    """
    Test example based on annual coal mining disasters in the UK. Occurrences
    of disasters in the time series is thought to be derived from a
    Poisson process with a large rate parameter in the early part of
    the time series, and from one with a smaller rate in the later part.
    We are interested in locating the switchpoint in the series using
    MCMC.
    """
    
    def __init__(self):
        """Class initialization"""
        
        MetropolisHastings.__init__(self)
        
        # Sample changepoint data (Coal mining disasters per year)
        self.data = (4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
            3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
            2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
            1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
            0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
            3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1)
        
        # Switchpoint is specified as a parameter to be estimated
        self.parameter(name='k', init_val=50, discrete=True)
        
        # Rate parameters of poisson distributions
        self.parameter(name='theta', init_val=array([1.0, 1.0]))
    
    def model(self):
        """Joint log-posterior"""
        
        # Obtain current values of parameters as local variables
        theta0, theta1 = self.theta
        k = self.k
        
        # Constrain k with prior
        self.uniform_prior(k, 1, len(self.data)-2)

        # Joint likelihood of parameters based on 2 assumed Poisson densities
        self.poisson_like(self.data[:k], theta0, name='early_mean')

        self.poisson_like(self.data[k:], theta1, name='late_mean')


class MCMCTest(unittest.TestCase):
    
    def testCoalMiningDisasters(self):
        """Run coal mining disasters example sampler"""
        
        print 'Running coal mining disasters test case ...'
        
        # Create an instance of the sampler
        self.sampler = DisasterSampler()
        
        # Specify the nimber of iterations to execute
        iterations = 10000
        thin = 2
        burn = 5000
        chains = 2
        
        # Run MCMC simulation
        for i in range(chains):
            
            self.failUnless(self.sampler.sample(iterations, burn=burn, thin=thin, plot=True, color='r'))
            
            # Run convergence diagnostics
            self.sampler.convergence(burn=burn, thin=thin)
            
            # Plot autocorrelation
            self.sampler.autocorrelation(burn=burn, thin=thin)
        
        # Goodness of fit
        x, n = self.sampler.goodness(iterations/10, burn=burn, thin=thin)['overall']
        self.failIf(x/n < 0.05 or x/n > 0.95)


if __name__=='__main__':
    # Run unit tests
    unittest.main()
