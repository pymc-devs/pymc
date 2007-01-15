"""

 To Try:

 - Store self.state_counter, self.parent_state_counter, and self.prior n computations deep. A depth of 2 
   guarantees only one spate of prior computations per Metropolis-Hastings step.

 - In individual parameters, compute or don't compute individual terms in log-likelihood based on whether parent has 
   changed. In Normal, for instance, only compute -.5 sum(log(tau)) if the tau parent has changed, but definitely compute 
   .5 * sum(tau * (x-mu) ** 2). This is probably best implemented as an overhaul of get_prior and compute_prior.

 - Build RJMCMC functionality right into the base Parameter class: if a Parameter is a prior - iid array, 
   it shouldn't be too hard. Give the user some options for how to make the transdimensional jumps.
 
 - Nodes and SubSamplers, which belong to only one process, should project avatar objects bearing their same name into all 
   processes. Avatar objects know how to communicate with the original via MPI. Avatars could be created by Sampler upon
   receipt of an add_parameter or add_subsampler command, since only Sampler should really _have_ to have global
   knowledge of what processes are available.
   
 - Parameters and SubSamplers need a propose_from method that takes another Parameter or SubSampler as an argument.
   Parameters that have some element of resolution to them, like Gaussian process convolutions and discretized
   stochastic differential equations, should know how to propose from other parameters like themselves on 
   coarser or finer grids. This'll help with multiscale parallel-chain MCMC.
   
 - Fully implement the skeletal samplers and parameters in the libraries. Especially the nonlinear_joint one, that'd
   be sweet!   

"""

"""
Markov chain Monte Carlo (MCMC) simulation, implementing an adaptive form of random walk Metropolis-Hastings sampling.
"""

# Import system functions
import sys, time, unittest, pdb

# Import numpy functions
from numpy import random, linalg
# Generalized inverse
inverse = linalg.pinv
from numpy import absolute, any, arange, around, array, atleast_1d
from numpy import concatenate
from numpy import diagonal
from numpy import exp
from numpy import log
from numpy import mean, cov, std
from numpy import ndim, ones,eye
from numpy import pi
from numpy import ravel, resize, reshape
from numpy import searchsorted, shape, sqrt, sort, sum, swapaxes, where
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
rexponential = random.exponential
from flib import binomial as fbinomial
rbinomial = random.binomial
from flib import bernoulli as fbernoulli
from flib import hyperg as fhyperg
from flib import mvhyperg as fmvhyperg
from flib import negbin as fnegbin
rnegbin = random.negative_binomial
from flib import normal as fnormal
vfnormal = vectorize(fnormal)
rnormal = random.normal
from flib import mvnorm as fmvnorm
rmvnormal = random.multivariate_normal
from flib import poisson as fpoisson
rpoisson = random.poisson
from flib import wishart as fwishart
from flib import gamma as fgamma
vfgamma = vectorize(fgamma)
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
from flib import gumbel as fgumbel
from flib import wshrt
from flib import gamfun
from flib import chol

# Shorthand of some verbose local variables

t = transpose

# Not sure why this doesnt work on Windoze!
try:
	inf = float('inf')
except ValueError:
	inf = 1.e100

class DivergenceError(ValueError):
	# Exception class for catching divergent samplers
	pass

class ParameterError(ValueError):
	# Exception class for catching parameters with invalid values
	pass
	
class LikelihoodError(ValueError):
	# Exception class for catching infinite log-likelihood
	pass
	
class ParametrizationError(ValueError):
	#Exception class for catching parameters with over- or under-parametrized distributions
	pass	
	
class TimestampExpired(ValueError):
	#Exception class for flagging when a prior needs to be recomputed
	pass	
	

""" Transformations """
def logit(p):
	"""Logit transformation"""
	
	try:
		return log(p/(1.0-p))
	except ZeroDivisionError:
		return inf

def invlogit(x):
	"""Inverse-logit transformation"""
	
	try:
		return 1.0 / (1.0 + exp(-1.0*x))
	except OverflowError:
		return 0.0

def normalize(x):
	"""Normalizes an array of elements"""
	
	y = ravel(x)
	
	ynorm = array((y-mean(y))/std(y))
	
	ynorm.reshape(shape(x))
	
	return ynorm

def histogram(x, nbins=None, normalize=False):
	"""
	Generates compnents of histogram for a given array of data.
	If the number of bins is not given, the optimal number is
	calculated according to:
	
	4 + 1.5*log(len(x))
	
	Returns (1) number of elements in each bin, (2) the minimum bin
	value and (3) the step size. Optional 'normalize' argument returns
	array that sums to unity.
	"""
	
	# Calculate optimal number of bins if none given
	nbins = nbins or int(4 + 1.5*log(len(x)))
	
	# Generate bins
	mn, mx = min(x), max(x)
	by = float(mx-mn)/nbins
	bins = arange(mn, mx, by)
	
	n = searchsorted(sort(x), bins)
	n = concatenate([n, [len(x)]])
	
	values = array(n[1:] - n[:-1])
	
	return normalize*(values/float(sum(values))) or values, mn, by

def ListOfCombinations(dims):
	# Returns a list of combinations for an array of passed dimensions
	if not dims: return []
	sets = [range(i) for i in dims]
	F = MakeListComprehensionFunction ('F', len(sets))
	return F(*sets)

def MakeListComprehensionFunction (name, nsets):
	"""Returns a function applicable to exactly <nsets> sets.
	  The returned function has the signature
		 F(set0, set1, ..., set<nsets>)
	  and returns a list of all element combinations as tuples.
	  A set may be any iterable object.
	"""
	if nsets <= 0:
		source = 'def %s(): return []\n' % name
	else:
		constructs = [ ('set%d'%i, 'e%d'%i, 'for e%d in set%d'%(i,i))
						for i in range(nsets) ]
		a, e, f = map(None, *constructs)
		##e.reverse() # <- reverse ordering of tuple elements if needed
		source = 'def %s%s:\n   return [%s %s]\n' % \
				(name, _tuplestr(a), _tuplestr(e), ' '.join(f))
	scope = {}
	exec source in scope
	return scope[name]

def _tuplestr(t):
	# Internal function that makes a tuple oc characters from a string
	if not t: return '()'
	return '(' + ','.join(t) + ',)'



""" Random number generation """

def randint(upper, lower):
	"""Returns a random integer. Accepts float arguments."""
	
	return randint(int(upper), int(lower))

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
		return [sum([urn[i] for i in permutation(total)[:draws]]) for j in range(n)]
	else:
		return sum([urn[i] for i in permutation(total)[:draws]])

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
normal_deviate = lambda var : rnormal(0,var)

# Centered uniform random deviate
uniform_deviate = lambda half_width: uniform(-half_width, half_width)

# Centered discrete uniform random deviate
discrete_uniform_deviate = lambda half_width: randint(-half_width, half_width)

def double_exponential_deviate(beta):
	"""Centered double-exponential random deviate"""
	
	u = random_number()
	
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
	
	def __init__(self, name, sampler=None, init_val=None, observed=False, shape=None, plot=True):
		"""Class initialization"""
		
		self.name = name
		
		# Timestamps
		self.state_counter = -1	
		self.state_counter_of_prior = None
		
		# Specify sampler
		self._sampler = sampler
		if self._sampler is not None:
			self._sampler.add_node(self)
		
		# Flag for plotting
		self._plot = plot
		
		#Initialize value
		self.current_value = None
		if shape:
			self.set_value(zeros(shape, 'd'))
		if init_val is not None:
			self.set_value(init_val)			
		
		# Empty list of traces
		self._traces = []
		
		# Parents is a dictionary. A parent is accessed by its parameter name. The elements of the dictionary
		# are 2-tuples, in which the first element is the parent node and the second element is the arguments
		# which will be passed to a parent node whenever self queries its state.
		self.parents = {}
		self.parent_eval_args = {}
		self.parent_state_counters_of_prior = {}
		
		# Children is a set; each child should be counted only once.
		self.children = set([])		

		
		# Is the value of this node known with no uncertainty?
		self.observed = observed
		

	
	def add_parent(self, new_parent, param_name, eval_argument = None):
		"""
		eval_argument is used if new_parent is an array or unobserved function 
		if new_parent is a scalar, eval_argument can be left as None.
		"""
		
		# Do I already have a parent by this name? If so, kick them out.
		if self.parents.has_key(param_name):
			if self.parents[param_name] is not None:
				self.remove_parent(param_name)
		self.parents[param_name] = new_parent
		self.parent_eval_args[param_name] = eval_argument
		new_parent.children.add(self)
		
	def remove_parent(self, bad_parent_key):
		
		if self.parents[bad_parent_key] is None:
			print "Warning, attempted to remove nonexistent parent ",bad_parent_key," from node ",self.name
		
		else:
			if self.parents.values().count(parents[bad_parent_key]) == 1:
				self.parents[bad_parent_key].children.remove(self)	
			self.parents[bad_parent_key] = None
			self.parent_eval_args[bad_parent_key] = None
				
	
	def init_trace(self, size):
		"""Initialize trace"""
		
		self._traces.append(zeros(size, 'd'))
		
		
	def get_value(self,eval_args = None):
		"""
		If no argument is passed in, return entire state.
		If an argument is passed in, this version of get_value assumes the argument is an index tuple
		and returns that element of state.
		This function may be overridden for particular parameters.
		"""
		if eval_args is None:
			return self.current_value
		else:
			return self.current_value[eval_args]
			
	def get_state_counter(self):
		return self.state_counter			

	def constrain(self, lower=-inf, upper=inf):
		"""Apply interval constraint on parameter value"""
		if any(lower > self.current_value) or any(self.current_value > upper):
			raise ParameterError

	def set_value(self,value):
		# Increment the timestamp
		self.last_value = self.current_value
		self.state_counter += 1
		
		if shape(value):
			self.current_value = array(value)
		else:
			self.current_value = value				

	
	def clear_trace(self):
		"""Removes last trace"""
		
		self._traces.pop()
		
	
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
	
	def plot(self, plotter, burn=0, thin=1, chain=-1, composite=False):
		"""Plot trace and histogram using Matplotlib"""
		
		if self._plot:
			# Call plotting support function
			try:
				trace = self.get_trace(burn, thin, chain, composite)
				plotter.plot(trace, self.name)
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
	
	def geweke(self, first=0.1, last=0.5, intervals=20, burn=0, thin=1, chain=-1, plotter=None):
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
					plotter.geweke_plot(t(zscores[name]), name=name)
		
		else:
			
			zscores[self.name] = self._calc_zscores(trace, first, last, intervals)
			
			# Plot if asked
			if plotter and self._plot:
				plotter.geweke_plot(t(zscores[self.name]), name=self.name)
		
		return zscores
	
	def autocorrelation(self, max_lag=100, burn=0, thin=1, chain=-1, plotter=None):
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
			plotter.bar_series_plot(autocorr, ylab='Autocorrelation', suffix='-autocorr')
		
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
	
	def __init__(self, name, init_val, sampler=None, dist='normal', scale=None, observed = False, random=False, plot=True):
		# Class initialization
		
		# Initialize superclass
		Node.__init__(self, name, sampler, init_val, observed, shape(init_val), plot)
		
		# Counter for rejected proposals; used for adaptation.
		self._rejected = 0
		
		# Counter for number of prior computations so far
		self.prior_computations = 0
		self.prior_computation_skips = 0
		
		# Augmented children are nearest non-deterministic descendants
		self.augmented_children = None		
		
		#Set current value of prior
		self.current_prior = None
		
		# Which sampler am I a member of?
		self._sampler = sampler
		if self._sampler is not None: 
			self._sampler.add_parameter(self)
		
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
			if self.get_value().ndim == 2:
				raise AttributeError, 'The multivariate_normal case is only intended for 1D arrays.'
			self._dist = lambda S : rmvnormal(zeros(self.dim), S)
		elif dist == 'prior':
			self._dist = None
		else:
			print 'Proposal distribution for', name, 'not recognized,' 'sampling from prior'
			self._dist = None
		
		# Vectorize proposal distribution if parameter is vector-valued
		# But not multivariate_normal, since it is already vectorized
		if self.dim and dist != 'multivariate_normal':
			self._dist = vectorize(self._dist)
		
		# Scale parameter for proposal distribution
		if scale is None:
			if dist == 'multivariate_normal':
				self._hyp = eye(*self.dim)
			elif self.dim:  # Vector case
				self._hyp = ones(self.dim)
			else:   		# Scalar case
				self._hyp = 1.0
		elif dist == 'multivariate_normal':
				if shape(scale) == shape(init_val):
					self._hyp = diagonal(scale)
				elif array((shape(scale)) == array(shape(init_val))).all():
					self._hyp = array(scale)
		elif shape(scale) != self.dim:
			raise AttributeError, 'The scale for parameter %s must have a shape coherent with init_val.' % self.name
		else:
			 self._hyp = scale
		
		# Adaptative scaling factor
		self._asf = 1.
		
		# Random effect flag (for use in AIC calculation)
		self.random = random

	def compute_prior(self):
		"""Distribution-specific"""
		pass
		
								
	def get_prior(self):
	
		"""
		Are last recorded values for parents not up to date?
		"""
		try:
		
			# Has my state changed?
			if not self.state_counter_of_prior == self.state_counter: 
				raise TimestampExpired

			for key in self.parents.keys():
				# Is the parent defined?
				if not self.parents[key] is None:
					# Is my snapshot of the parent's state up to date?
					if not self.parents[key].get_state_counter() == self.parent_state_counters_of_prior[key]: raise TimestampExpired
			
			self.prior_computation_skips += 1
			"""
			If necessary, recompute prior, and increment timestamps
			"""

		except TimestampExpired:
			
			# Count up number of prior computations for profiling
			self.prior_computations += 1
			self.current_prior = self.compute_prior()
			self.state_counter_of_prior = self.state_counter

			for key in self.parents.keys():
				if self.parents[key] is not None:
					self.parent_state_counters_of_prior[key] = self.parents[key].get_state_counter()
		
		
		return self.current_prior
						
	def get_likelihood(self):
		"""
		Return sum of log priors of children, conditional on state of self
		
		If a child is deterministic, add the sum of the log priors of its augmented children
		conditional on state of self. 
		"""
		if self.augmented_children is None:
			self.find_augmented_children()
		if self.augmented_children == set([]):
			return 0
		else:
			return sum([child.get_prior() for child in self.augmented_children])

	def find_augmented_children(self):
		"""
		The union of nondeterministic children and the earliest nondeterministic descendants of deterministic
		children is the augmented children.
		"""
		
		if self.augmented_children is None:
			self.augmented_children = self.children
			
		if not any([isinstance(child,DeterministicFunction) for child in self.augmented_children]): 
			return
		else:
			for child in self.augmented_children:
				if isinstance(child,DeterministicFunction):
					self.augmented_children.update(child.children)
					self.augmented_children.discard(child)
			# Recur in case deterministic nodes have deterministic descendants
			self.find_augmented_children()
		
		return

	
	def sample_from_prior(self):
		pass
	
	def sample_candidate(self):
		"""Samples a candidate value based on proposal distribution"""
		
		if self._dist is None:
			self.sample_from_prior()
			return
		
		try:
			self.set_value(self.current_value + self._dist(self._hyp*self._asf))
		
		except ValueError:
			print self.name, ': Hyperparameter approaching zero:', self._hyp
			raise DivergenceError

	def metropolis_step(self, debug=False):
				
		# If this is an observed parameter, do nothing
		if self.observed:
			return
		
		try:
			old_prior = self.get_prior()
			old_like = self.get_likelihood()			
		except ParametrizationError, msg:
			print msg
			sys.exit(1)
		
		"""
		Propose new values using a random walk algorithm, according to
		the proposal distribution specified:

		x(t+1) = x(t) + e(t)

		where e ~ proposal(hyperparameters)
		"""
		self.sample_candidate()
		
		# New log-likelihood and log-probability
		try:
			new_prior = self.get_prior()					
			new_like = self.get_likelihood()
		except ParametrizationError, msg:
			print msg
			sys.exit(1)
			
		"""Accept or reject proposed parameter values"""
						
		# Reject bogus results
		if str(new_like) == 'nan' or new_like == -inf or str(new_prior) == 'nan' or new_prior == -inf:
			self.revert()
			self._rejected += 1		
			return		
		
		# Compute likelihood ratio
		logp_difference = new_like - old_like
		# If proposal isn't prior, multiply by prior probability ratio
		if self._dist is not None:
			logp_difference += new_prior - old_prior
		
		# Test
		try:
			if log(random_number()) <= logp_difference:
				pass
			else:
				self.revert()
				self._rejected += 1	
		except ParameterError:
			print self.name, ': ', msg
			sys.exit(1)

	def revert(self):
		# Revert current_value to last_value and decrement state counter
		if self.last_value is not None:
			self.current_value = self.last_value
			self.last_value = None
			self.state_counter -= 1
		else:
			raise ParameterError, "can't revert; last_value not defined"
			
	def tune(self, int_length, divergence_threshold=1e10, verbose=False):
		"""
		Tunes the scaling hyperparameter for the proposal distribution
		according to the acceptance rate of the last k proposals:

		Rate	Variance adaptation
		----	-------------------
		<0.001		x 0.1
		<0.05 		x 0.5
		<0.2  		x 0.9
		>0.5  		x 1.1
		>0.75 		x 2
		>0.95 		x 10

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
				
				# Uncorrelated multivariate case
				if self._dist_name != 'multivariate_normal':
					
					# Computes the standard variation over the last 3 intervals.
					hyp = std(self._traces[-1][max(0, it-3 * int_length):it])

					# Ensure that there are no null values before commiting to self.
					if (hyp > 0).all():
						self._hyp = hyp
				
				# Correlated multivariate case
				else:
					
					hyp = cov(self._traces[-1][max(0, it-3 * int_length):it], rowvar=0)

					if (hyp.diagonal() > 0).all(): self._hyp = hyp

					# Reset the correlation coefficients to 0 once in a while to
					# avoid staying trapped on a N-D line, due to previous high correlations.
					# I need to find a better solution to this problem, this one is unpredictable.

					if random_number() < 0.1: self._hyp = self._hyp.diagonal()*eye(*self.dim)
					
		except AttributeError:
			pass

class Constant(Node):
	"""
	Constant parameters are never updated.
	
	Use this class for eg. prior parameters.
	
	DO NOT use this class for data that depends on other parameters.
	"""
	
	def __init__(self, name, init_val=None, sampler=None):
		Node.__init__(self, name, sampler, init_val, observed=True, shape=shape(init_val), plot=False)
			
class DeterministicFunction(Node):
	"""
	DeterministicFunction is a deterministic function of its parents.
	Pass this function in as eval_fun.
	Pass the argument keys as parent_keys.
	
	For example, to implement parameter z in z = x^2 + y^2,
	
	z = DeterministicFunction("z",  lambda x,y : x ** 2 + y ** 2,  ("x","y") )
	
	z.add_parent(x,"x"), z.add_parent(y,"y")
	
	and you're ready to go!
	"""
	
	def __init__(self, name, eval_fun, parent_keys, sampler=None, observed=False, random=False, plot=False, output_shape = ()):
		Node.__init__(self, name, sampler, None, observed, output_shape, plot)
		self.eval_fun = vectorize(eval_fun)
		self.parent_keys = parent_keys
		self.args = None
		self.current_value = None
		
		self.value_computations = 0
		self.value_computation_skips = 0

		
	def get_value(self,eval_args = None):
	
		if any([not self.parents.has_key(key) for key in self.parent_keys]):
			raise ParametrizationError

		# Are last recorded values for parents not up to date, or has value never been calculated?
		try:
			if self.current_value is None: raise TimestampExpired
			for key in self.parent_keys:
				if not self.parents[key].get_state_counter() == self.parent_state_counters_of_prior[key]: raise TimestampExpired
			self.value_computation_skips += 1
		# If not, recompute value and increment timestamps		
		except TimestampExpired:
			self.set_value(self.compute_value())
			self.state_counter_of_prior = self.state_counter
			for key in self.parents.keys():
				self.parent_state_counters_of_prior[key] = self.parents[key].get_state_counter()
			self.value_computations += 1
		
		"""
		If no argument is passed in, return entire state.
		If an argument is passed in, this version of get_value assumes the argument is an index tuple
		and returns that element of state.
		"""
		if eval_args is None:
			return self.current_value
		else:
			return self.current_value[eval_args]

	def get_state_counter(self):
		# Test for up-to-date-ness of current value, and if it's behind the times recompute it
		self.get_value()
		# Return current state counter
		return self.state_counter
					
	def compute_value(self):
		"""
		Evaluates the function that was passed to the constructor based on parents' values
		"""
	
		self.args = ([self.parents[key].get_value(self.parent_eval_args[key]) for key in self.parent_keys])
		return self.eval_fun(*self.args)		

class Index(DeterministicFunction):
	"""
	Used in model averaging; points to one of a set of candidate parent nodes or subsamplers, and directs
	inquiries about its state to that node/subsampler.
	"""
	pass

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
			self.current_value = around(value, 0).astype(int)
		except TypeError:
			self.current_value = int(round(value, 0))
	
	def compute_scale(self, acc_rate, int_length):
		"""Returns 1 for discrete parameters.'"""
		# Since discrete parameters may not show much variability, they may
		# return standard variations equal to 0. It is hence more robust to let _asf
		# take care of the tuning.
		pass



class SubSampler:
	"""
	I haven't tested this yet...
    """

	
	def __init__(self, sampler = None, parameters = None, debug = False):
		"""Class initializer"""
		
		# Initialize parameters and children
		self.children = set()
		self.augmented_children = set()
		self._sampler = sampler		
		for parameter in parameters:
			self.add_parameter(parameter)

		
		if self._sampler is not None:
			self._sampler.add_sampler(self)
		
		if any(self.parameters) is DeterministicFunction:
			print "Error, deterministic parameters cannot be added to a sampler" 
			raise ParametrizationError
				
	def add_parameter(self,new_parameter):
		if new_parameter is DeterministicFunction:
			print "Error, deterministic parameter ",new_parameter.name," cannot be added to a sampler"
			raise ParametrizationError
		else:
			self.parameters.update([new_parameter])
			self.children.update(new_parameter.children).difference(parameters)

	def get_prior(self):
		"""
		No complicated monkeyshines with timestamps are necessary here, each node makes its own decision about 
		whether to recompute its prior.
		"""
		return sum([parameter.get_prior() for parameter in self.parameters])

	def find_augmented_children(self):
		"""
		The union of nondeterministic children and the earliest nondeterministic descendants of deterministic
		children is the augmented children.
		"""	
		if self.augmented_children is None:
			self.augmented_children = self.children
		if not any([isinstance(child,DeterministicFunction) for child in self.augmented_children]):
			return

		else:
			for child in self.augmented_children:
				if isinstance(child,DeterministicFunction):
					self.augmented_children.update(child.children)
				else:
					self.augmented_children.add(child)
			self.find_augmented_children


	def get_likelihood(self):
		"""
		Return sum of log priors of children, conditional on state of self
		
		If a child is deterministic, add the sum of the log priors of its augmented children
		conditional on state of self. 
		"""
		
		if self.augmented_children is None:
			self.find_augmented_children()
		if self.agumented_children == set([]):
			return 0
		else:
			return sum([child.get_prior() for child in self.augmented_children])

	def step(self):
		"""
		Overrideable for individual samplers
		"""
		for parameter in self.parameters: 
			parameter.metropolis_step()

	def tally(self):
		for parameter in self.parameters: 
			parameter.tally()

	def tune(self):
		for parameter in self.parameters: 
			parameter.tune()


class Sampler:
	"""
	Should be able to retain the functionality of the original Sampler without much trouble, but I haven't tried yet.
	"""

	def __init__(self, plot_format='png', plot_backend='TkAgg'):

		self.nodes = {}
		self.parameters = set()
		self.subsamplers = set()
	
		# Create and initialize node for subsampler deviance
		self.node('deviance')
		
		self.plotter = None
		# Create plotter, if module was imported
		try:
			# Pass appropriate graphic format and backend
			self.plotter = PlotFactory(format=plot_format, backend=plot_backend)
		except NameError:
			pass
        
		# Goodness of Fit flag
		self._gof = False
		

	def add_parameter(self,new_parameter):
		self.parameters.add(new_parameter)
		
	def add_subsampler(self,new_subsampler):
		self.subsamplers.add(new_subsampler)		
    
	def node(self, name, shape=None, plot=True):
		"""Create a new node"""
        
		self.nodes[name] = Node(name, self, shape=shape, plot=plot)

	def sample(self, iterations, burn=0, thin=1, tune=True, tune_interval=100, divergence_threshold=1e10, verbose=True, plot=True, debug=False):
		"""
		Skeletal version given here
		"""
				
		#Find all parameters that aren't members of a subsampler
		self.lone_parameters = self.parameters
		for subsampler in self.subsamplers:
			self.lone_parameters.difference(subsampler.parameters)
		
		#Gather the lone parameters into a one-at-a-time subsampler
		if len(self.lone_parameters) > 0:
			self.add_subsampler(SubSampler(self.lone_parameters))
		
		#Tell all the subsamplers to step, tune, etc.		
		for iteration in range(iterations):
			for subsampler in subsamplers:
				subsampler.sample()
				subsampler.tune()
				
	"""
	And all the model statistics and goodness-of-fit functionality, management of tuning and tallying,
	plotting, etc.
	"""