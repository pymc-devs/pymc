"""
NOTE: In Parameters that might conceivably benefit from some kind of multiscale scheme, self.value()
should be overriden to handle values that are on coarser or finer scales/meshes than self.
"""

from base_classes import *
from scipy.linalg.fblas import *

class Normal(Parameter):
	def __init__(self, name, init_val=None, model=None, dist='normal', scale=None, observed = False, random=False, plot=True):

		Parameter.__init__(self, name, init_val, model, dist, scale, observed, random, plot)

		self.parents["mu"] = None
		self.parents["V"] = None
		self.parents["sigma"] = None
		self.parents["tau"] = None
		self.mu = None
		self.tau = None
		self.x = None		

		# vectorize fnormal
		self.vfnormal = vectorize(fnormal)


	def compute_prior(self):		

		"""Normal log-likelihood"""
		if self.parents["mu"] is None: 
			raise ParametrizationError, self.name + ': my mu parameter is missing.'
		elif ( self.parents["V"] is None and self.parents["sigma"] is None and self.parents["tau"] is None ):
			raise ParametrizationError, self.name + ": I don't have a V, sigma, or tau parameter."
		elif sum([ self.parents["V"] is None, self.parents["sigma"] is None, self.parents["tau"] is None ]) < 2:
			raise ParametrizationError, self.name + ": I only want a V, sigma, OR tau parameter, but have more than one of these."
			
		self.get_value()
		
		# Retrieve mean
		shape_now = shape(self.parents["mu"].get_value(self.parent_eval_args["mu"]))
		if shape_now == ():
			self.mu = self.parents["mu"].get_value(self.parent_eval_args["mu"])
		elif not shape_now == shape(self.current_value):
			raise ParametrizationError, self.name + ": my mu parameter " + self.parents["mu"].name + "'s shape is " + str(shape_now) + " but mine is " + str(shape(self.current_value))
		else:
			self.mu = reshape(self.parents["mu"].get_value(self.parent_eval_args["mu"]),-1)
		
		# Retrieve sigma, tau, or V, depending on which parent is defined
		if self.parents["V"]:
			try:
				self.parents["V"].constrain(lower=0)
			except ParameterError:
				return -inf
			shape_now = shape(self.parents["V"].get_value(self.parent_eval_args["V"]))
			if shape_now == ():
				self.tau = 1. /self.parents["V"].get_value(self.parent_eval_args["V"])
			elif not shape_now == shape(self.current_value):
				raise ParametrizationError, self.name + ": my V parameter " + self.parents["V"].name + "'s shape is " + str(shape_now) + " but mine is " + str(shape(self.current_value))			
			else:
				self.tau = 1. / reshape(self.parents["V"].get_value(self.parent_eval_args["V"]),-1)

		if self.parents["sigma"]:
			shape_now = shape(self.parents["sigma"].get_value(self.parent_eval_args["sigma"]))
			if shape_now == ():			
				self.tau = 1. / self.parents["sigma"].get_value(self.parent_eval_args["sigma"]) ** 2.
			elif not shape_now == shape(self.current_value):
				raise ParametrizationError, self.name + ": my sigma parameter " + self.parents["sigma"].name + "'s shape is " + str(shape_now) + " but mine is " + str(shape(self.current_value))
			else: 
				self.tau = 1. / reshape(self.parents["sigma"].get_value(self.parent_eval_args["sigma"]),-1)
				self.tau *= self.tau
			

		if self.parents["tau"]:
			try:
				self.parents["tau"].constrain(lower=0)
			except ParameterError:
				return -inf
			shape_now = shape(self.parents["tau"].get_value(self.parent_eval_args["tau"]))				
			if shape_now == ():			
				self.tau = self.parents["tau"].get_value(self.parent_eval_args["tau"])
			elif not shape_now == shape(self.current_value):
				raise ParametrizationError, self.name + ": my tau parameter " + self.parents["tau"].name + "'s shape is " + str(shape_now) + " but mine is " + str(shape(self.current_value))
			else:
				self.tau = reshape(self.parents["tau"].get_value(self.parent_eval_args["tau"]),-1)

			
		
		# Flatten current value
		if shape(self.current_value):			
			self.x = self.current_value.flat[:]
		else:
			self.x = self.current_value

		self.x -= self.mu
		self.x *= self.x
		self.x *= self.tau
		logp = .5 * (sum(log(self.tau)) - sum(self.x))
		return logp
			
		
		
class Gamma(Parameter):
	def __init__(self, name, init_val=None, model=None, dist='normal', scale=None, observed = False, random=False, plot=True):

		Parameter.__init__(self, name, init_val, model, dist, scale, observed, random, plot)

		self.parents["alpha"] = None
		self.parents["beta"] = None
		self.x = None
		self.alpha = None
		self.beta = None

		
		self.vfgamma = vectorize(fgamma)
		
	def compute_prior(self):
		
		if self.parents["alpha"] is None:
			raise ParametrizationError, self.name + ': my shape (alpha) parameter is missing'
		elif self.parents["beta"] is None:
			raise ParametrizationError, self.name + ': my scale (beta) parameter is missing'
			
		# Make sure current value and values of alpha and beta are positive
		try:
			self.constrain(lower = 0)
			self.parents["alpha"].constrain(lower = 0)
			self.parents["beta"].constrain(lower = 0)
		except ParameterError:
			return -inf
		
		
		# Flatten current value
		if shape(self.current_value):			
			self.x = self.current_value.flat[:]
		else:
			self.x = self.current_value
		
		# Retrieve alpha and beta						
		try:
			self.parents["alpha"].constrain(lower=0)
		except ParameterError:
			return -inf
		shape_now = shape(self.parents["alpha"].get_value(self.parent_eval_args["alpha"]))				
		if shape_now == ():			
			self.alpha = resize(self.parents["alpha"].get_value(self.parent_eval_args["alpha"]),shape(self.x))
		elif not shape_now == shape(self.current_value):
			raise ParametrizationError, self.name + ": my shape (alpha) parameter " + self.parents["alpha"].name + "'s shape is " + str(shape_now) + " but mine is " + str(shape(self.current_value))
		else:
			self.alpha = self.parents["alpha"].get_value(self.parent_eval_args["alpha"])
			
		try:
			self.parents["beta"].constrain(lower=0)
		except ParameterError:
			return -inf
		shape_now = shape(self.parents["beta"].get_value(self.parent_eval_args["beta"]))				
		if shape_now == ():			
			self.beta = resize(self.parents["beta"].get_value(self.parent_eval_args["beta"]),shape(self.x))
		elif not shape_now == shape(self.current_value):
			raise ParametrizationError, self.name + ": my scale (beta) parameter " + self.parents["beta"].name + "'s shape is " + str(shape_now) + " but mine is " + str(shape(self.current_value))
		else:
			self.beta = self.parents["beta"].get_value(self.parent_eval_args["beta"])
		
		# Call to vectorized fgamma
		return sum(self.vfgamma( self.x, self.alpha, self.beta ))

		
class Uniform(Parameter):
	pass
	
class UniformMixture(Parameter):
	pass
	
class Beta(Parameter):
	pass
	
class Dirichlet(Parameter):
	pass
	
class NegativeBinomial(Parameter):
	pass
	
class Geometric(Parameter):
	pass
	
class Hypergeometric(Parameter):
	pass
	
class MVHypergeometric(Parameter):
	pass
	
class Bernoulli(Parameter):
	pass
	
class Multinomial(Parameter):
	pass
	
class Poisson(Parameter):
	pass
	
class Chi2(Parameter):
	pass
	
class InverseGamma(Parameter):
	pass
	
class Exponential(Parameter):
	pass
	
class HalfNormal(Parameter):
	pass
	
class LogNormal(Parameter):
	pass
	
class MVNormal(Parameter):
	pass
	
class Wishart(Parameter):
	pass
	
class Gumbel(Parameter):
	pass
	
class Cauchy(Parameter):
	pass
	
class Weibull(Parameter):
	pass
	
class OrdinaryDifferentialEquation(DeterministicFunction):
	"""
	Differential geometry package PyDX has an ODE solver, but it's not documented
	as yet and may not be optimized for speed. Surely SciPy has some kind of ODE
	solver, though; otherwise just use an RK4.
	"""
	pass
	
def uniform_density(self, x, lower, upper, name='uniform', prior=False):
	"""Beta log-likelihood"""
	
	if not shape(lower) == shape(upper): raise ParameterError, 'Parameters must have same dimensions in uniform(like)'
	
	# Allow for multidimensional arguments
	if ndim(lower) > 1:
		
		return sum([self.uniform_like(y, l, u, name, prior) for y, l, u in zip(x, lower, upper)])
	
	else:
		
		# Ensure valid values of parameters
		self.constrain(x, lower=lower, upper=upper)
		
		# Equalize dimensions
		x = atleast_1d(x)
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
			y = array([runiform(a, b) for a, b in zip(lower, upper)])
			
			# Generate GOF points
			gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]))
			
			self._gof_loss.append(gof_points)
		
		return sum(log(1. / (array(upper) - array(lower))))

def uniform_prior(self, parameter, lower, upper):
	"""Uniform prior distribution"""
	
	return self.uniform_like(parameter, lower, upper, prior=True)

def uniform_mixture_like(self, x, lower, median, upper, name='uniform_mixture', prior=False):
	"""Uniform mixture log-likelihood
	
	This distribution is specified by three parameters (upper bound,
	median, lower bound), defining a mixture of 2 uniform distributions,
	that share the median as an upper (lower) bound. Hence, half of the
	density is in the first distribution and the other half in the second."""
	
	if not shape(lower) == shape(median) == shape(upper): raise ParameterError, 'Parameters must have same dimensions in uniform_mixture_like()'
	
	# Allow for multidimensional arguments
	if ndim(lower) > 1:
		
		return sum([self.uniform_mixture_like(y, l, m, u, name, prior) for y, l, m, u in zip(x, lower, median, upper)])
	
	else:
		
		# Ensure valid values of parameters
		self.constrain(median, lower, upper)
		self.constrain(x, lower, upper)
		
		# Equalize dimensions
		x = atleast_1d(x)
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
			y = array([rmixeduniform(a, m, b) for a, m, b in zip(lower, median, upper)])
			
			# Generate GOF points
			gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]))
			
			self._gof_loss.append(gof_points)
		
		return sum(log((x <= median) * 0.5 / (median - lower)  +  (x > median) * 0.5 / (upper - median)))

def uniform_mixture_prior(self, parameter, lower, median, upper):
	"""Uniform mixture prior distribution"""
	
	return self.uniform_mixture_like(parameter, lower, median, upper, prior=True)

def beta_like(self, x, alpha, beta, name='beta', prior=False):
	"""Beta log-likelihood"""
	
	if not shape(alpha) == shape(beta): raise ParameterError, 'Parameters must have same dimensions in beta_like()'
	
	# Allow for multidimensional arguments
	if ndim(alpha) > 1:
		
		return sum([self.beta_like(y, a, b, name, prior) for y, a, b in zip(x, alpha, beta)])
	
	else:
		
		# Ensure valid values of parameters
		self.constrain(alpha, lower=0)
		self.constrain(beta, lower=0)
		self.constrain(x, 0, 1)
		
		# Equalize dimensions
		x = atleast_1d(x)
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
			y = array([rbeta(a, b) for a, b in zip(alpha, beta)])
			
			# Generate GOF points
			gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]))
			
			self._gof_loss.append(gof_points)
		
		return sum([fbeta(y, a, b) for y, a, b in zip(x, alpha, beta)])

def beta_prior(self, parameter, alpha, beta):
	"""Beta prior distribution"""
	
	return self.beta_like(parameter, alpha, beta, prior=True)

def dirichlet_like(self, x, theta, name='dirichlet', prior=False):
	"""Dirichlet log-likelihood"""
	
	# Allow for multidimensional arguments
	if ndim(theta) > 1:
		
		return sum([self.dirichlet_like(y, t, name, prior) for y, t in zip(x, theta)])
	
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
				gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]))
				
				self._gof_loss.append(gof_points)
		
		return fdirichlet(x, theta)

def dirichlet_prior(self, parameter, theta):
	"""Dirichlet prior distribution"""
	
	return self.dirichlet_like(parameter, theta, prior=True)

def dirichlet_multinomial_like(self, x, theta, name='dirichlet_multinomial', prior=False):
	"""Dirichlet multinomial log-likelihood"""
	
	# Allow for multidimensional arguments
	if ndim(theta) > 1:
		
		return sum([self.dirichlet_multinomial_like(y, t, name, prior) for y, t in zip(x, theta)])
	
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
				gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]))
				
				self._gof_loss.append(gof_points)
		
		return fdirmultinom(x, theta)

def dirichlet_multinomial_prior(self, parameter, theta):
	"""Dirichlet multinomial prior distribution"""
	
	return self.dirichlet_multinomial_like(parameter, theta, prior=True)

def negative_binomial_like(self, x, r, p, name='negative_binomial', prior=False):
	"""Negative binomial log-likelihood"""
	
	if not shape(r) == shape(p): raise ParameterError, 'Parameters must have same dimensions'
	
	# Allow for multidimensional arguments
	if ndim(r) > 1:
		
		return sum([self.negative_binomial_like(y, _r, _p, name, prior) for y, _r, _p in zip(x, r, p)])
	
	else:
		
		# Ensure valid values of parameters
		self.constrain(p, 0, 1)
		self.constrain(r, lower=0)
		self.constrain(x, lower=0)
		
		# Enforce array type
		x = atleast_1d(x)
		r = resize(r, shape(x))
		p = resize(p, shape(x))
		
		# Goodness-of-fit
		if self._gof and not prior:
			
			try:
				self._like_names.append(name)
			except AttributeError:
				pass
			
			expval = r * (1. - p) / p
			
			# Simulated values
			y = array([rnegbin(_r, _p) for _r, _p in zip(r, p)])
			
			# Generate GOF points
			gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]))
			
			self._gof_loss.append(gof_points)
		
		return sum([fnegbin(_x, _r, _p) for _x, _r, _p in zip(x, r, p)])

def negative_binomial_prior(self, parameter, r, p):
	"""Negative binomial prior distribution"""
	
	return self.negative_binomial_like(parameter, r, p, prior=True)

def geometric_like(self, x, p, name='geometric', prior=False):
	"""Geometric log-likelihood"""
	
	# Allow for multidimensional arguments
	if ndim(p) > 1:
		
		return sum([self.geometric_like(y, q, name, prior) for y, q in zip(x, p)])
	
	else:
		
		# Ensure valid values of parameters
		self.constrain(p, 0, 1)
		self.constrain(x, lower=0)
		
		# Enforce array type
		x = atleast_1d(x)
		p = resize(p, shape(x))
		
		# Goodness-of-fit
		if self._gof and not prior:
			
			try:
				self._like_names.append(name)
			except AttributeError:
				pass
			
			expval = (1. - p) / p
			
			# Simulated values
			y = array([rnegbin(1, q) for q in p])
			
			# Generate GOF points
			gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]))
			
			self._gof_loss.append(gof_points)
		
		return sum([fnegbin(y, 1, q) for y, q in zip(x, p)])

def geometric_prior(self, parameter, p):
	"""Geometric prior distribution"""
	
	return self.geometric_like(parameter, p, prior=True)

def hypergeometric_like(self, x, n, m, N, name='hypergeometric', prior=False):
	"""
	Hypergeometric log-likelihood
	
	Distribution models the probability of drawing x successful draws in n
	draws from N total balls of which m are successes.
	"""
	
	if not shape(n) == shape(m) == shape(N): raise ParameterError, 'Parameters must have same dimensions'
	
	# Allow for multidimensional arguments
	if ndim(n) > 1:
		
		return sum([self.hypergeometric_like(y, _n, _m, _N, name, prior) for y, _n, _m, _N in zip(x, n, m, N)])
	
	else:
		
		# Ensure valid values of parameters
		self.constrain(m, upper=N)
		self.constrain(n, upper=N)
		self.constrain(x, max(0, n - N + m), min(m, n))
		
		# Enforce array type
		x = atleast_1d(x)
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
			y = array([rhyperg(_n, _m, _N) for _n, _m, _N in zip(n, m, N)])
			
			# Generate GOF points
			gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]))
			
			self._gof_loss.append(gof_points)
		
		return sum([fhyperg(_x, _n, _m, _N) for _x, _n, _m, _N in zip(x, n, m, N)])

def hypergeometric_prior(self, parameter, n, m, N):
	"""Hypergeometric prior distribution"""
	
	return self.hypergeometric_like(parameter, n, m, N, prior=True)

def multivariate_hypergeometric_like(self, x, m, name='multivariate_hypergeometric', prior=False):
	"""Multivariate hypergeometric log-likelihood"""
	
	# Allow for multidimensional arguments
	if ndim(m) > 1:
		
		return sum([self.multivariate_hypergeometric_like(y, _m, name, prior) for y, _m in zip(x, m)])
	
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
				gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]))
				
				self._gof_loss.append(gof_points)
		
		return fmvhyperg(x, m)

def multivariate_hypergeometric_prior(self, parameter, m):
	"""Multivariate hypergeometric prior distribution"""
	
	return self.multivariate_hypergeometric_like(parameter, m, prior=True)

def binomial_like(self, x, n, p, name='binomial', prior=False):
	"""Binomial log-likelihood"""
	
	if not shape(n) == shape(p): raise ParameterError, 'Parameters must have same dimensions'
	
	if ndim(n) > 1:
		
		return sum([self.binomial_like(y, _n, _p, name, prior) for y, _n, _p in zip(x, n, p)])
	
	else:
		
		# Ensure valid values of parameters
		self.constrain(p, 0, 1)
		self.constrain(n, lower=x)
		self.constrain(x, 0)
		
		# Enforce array type
		x = atleast_1d(x)
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
			y = array([rbinomial(_n, _p) for _n, _p in zip(n, p)])
			
			# Generate GOF points
			gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]))
			
			self._gof_loss.append(gof_points)
		
		return sum([fbinomial(xx, nn, pp) for xx, nn, pp in zip(x, n, p)])

def binomial_prior(self, parameter, n, p):
	"""Binomial prior distribution"""
	
	return self.binomial_like(parameter, n, p, prior=True)

def bernoulli_like(self, x, p, name='bernoulli', prior=False):
	"""Bernoulli log-likelihood"""
	
	if ndim(p) > 1:
		
		return sum([self.bernoulli_like(y, _p, name, prior) for y, _p in zip(x, p)])
	
	else:
		
		# Ensure valid values of parameters
		self.constrain(p, 0, 1)
		self.constrain(x, 0, 1)
		
		# Enforce array type
		x = atleast_1d(x)
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
			gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]))
			
			self._gof_loss.append(gof_points)
		
		return sum([fbernoulli(y, _p) for y, _p in zip(x, p)])

def bernoulli_prior(self, parameter, p):
	"""Bernoulli prior distribution"""
	
	return self.bernoulli_like(parameter, p, prior=True)

def multinomial_like(self, x, n, p, name='multinomial', prior=False):
	"""Multinomial log-likelihood with k-1 bins"""
	
	if not shape(n) == shape(p): raise ParameterError, 'Parameters must have same dimensions'
	
	if ndim(n) > 1:
		
		return sum([self.multinomial_like(y, _n, _p, name, prior) for y, _n, _p in zip(x, n, p)])
	
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
			gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]))
			
			self._gof_loss.append(gof_points)
		
		return fmultinomial(x, n, p)

def multinomial_prior(self, parameter, n, p):
	"""Beta prior distribution"""
	
	return self.multinomial_like(parameter, n, p, prior=True)

def poisson_like(self, x, mu, name='poisson', prior=False):
	"""Poisson log-likelihood"""
	
	if ndim(mu) > 1:
		
		return sum([self.poisson_like(y, m) for y, m in zip(x, mu)])
	
	else:
		
		# Ensure valid values of parameters
		self.constrain(x, lower=0)
		self.constrain(mu, lower=0)
		
		# Enforce array type
		x = atleast_1d(x)
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
			gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]))
			
			self._gof_loss.append(gof_points)
		
		return sum([fpoisson(y, m) for y, m in zip(x, mu)])

def poisson_prior(self, parameter, mu):
	"""Poisson prior distribution"""
	
	return self.poisson_like(parameter, mu, prior=True)

def gamma_like(self, x, alpha, beta, name='gamma', prior=False):
	"""Gamma log-likelihood"""
	
	if not shape(alpha) == shape(beta): raise ParameterError, 'Parameters must have same dimensions in gamma_like()'
	
	# Allow for multidimensional arguments
	if ndim(alpha) > 1:
		
		return sum([self.gamma_like(y, a, b, name, prior) for y, a, b in zip(x, alpha, beta)])
	
	# Ensure valid values of parameters
	self.constrain(x, lower=0)
	self.constrain(alpha, lower=0)
	self.constrain(beta, lower=0)
	
	# Ensure proper dimensionality of parameters
	x = atleast_1d(x)
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
		y = array([rgamma(b, a) for b, a in zip(ibeta, alpha)])
		
		# Generate GOF points
		gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]))
		
		self._gof_loss.append(gof_points)
	
	return sum([fgamma(y, a, b) for y, a, b in zip(x, alpha, beta)])

def gamma_prior(self, parameter, alpha, beta):
	"""Gamma prior distribution"""
	
	return self.gamma_like(parameter, alpha, beta, prior=True)

def chi2_like(self, x, df, name='chi_squared', prior=False):
	"""Chi-squared log-likelihood"""
	
	if ndim(df) > 1:
		
		return sum([self.chi2_like(y, d, name, prior) for y, d in zip(x, df)])
	
	else:
		
		# Ensure valid values of parameters
		self.constrain(x, lower=0)
		self.constrain(df, lower=0)
		
		# Ensure array type
		x = atleast_1d(x)
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
			gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]))
			
			self._gof_loss.append(gof_points)
		
		return sum([fgamma(y, 0.5*d, 2) for y, d in zip(x, df)])

def chi2_prior(self, parameter, df):
	"""Chi-squared prior distribution"""
	
	return self.chi2_like(parameter, df, prior=True)

def inverse_gamma_like(self, x, alpha, beta, name='inverse_gamma', prior=False):
	"""Inverse gamma log-likelihood"""
	
	if not shape(alpha) == shape(beta): raise ParameterError, 'Parameters must have same dimensions in gamma_like()'
	
	# Allow for multidimensional arguments
	if ndim(alpha) > 1:
		
		return sum([self.inverse_gamma_like(y, a, b, name, prior) for y, a, b in zip(x, alpha, beta)])
	else:
		
		# Ensure valid values of parameters
		self.constrain(x, lower=0)
		self.constrain(alpha, lower=0)
		self.constrain(beta, lower=0)
		
		# Ensure proper dimensionality of parameters
		x = atleast_1d(x)
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
			y = array([rgamma(b, a) for b, a in zip(ibeta, alpha)])
			
			# Generate GOF points
			gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]))
			
			self._gof_loss.append(gof_points)
		
		return sum([figamma(xx, a, b) for xx, a, b in zip(x, alpha, beta)])

def inverse_gamma_prior(self, parameter, alpha, beta):
	"""Inverse gamma prior distribution"""
	
	return self.inverse_gamma_like(parameter, alpha, beta, prior=True)

def exponential_like(self, x, beta, name='exponential', prior=False):
	"""Exponential log-likelihood"""
	
	# Allow for multidimensional arguments
	if ndim(beta) > 1:
		
		return sum([self.exponential_like(y, b, name, prior) for y, b in zip(x, beta)])
	
	else:
		
		# Ensure valid values of parameters
		self.constrain(x, lower=0)
		self.constrain(beta, lower=0)
		
		# Ensure proper dimensionality of parameters
		x = atleast_1d(x)
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
			gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]))
			
			self._gof_loss.append(gof_points)
		
		return sum([fgamma(xx, 1, b) for xx, b in zip(x, beta)])

def exponential_prior(self, parameter, beta):
	"""Exponential prior distribution"""
	
	return self.exponential_like(parameter, beta, prior=True)

def normal_like(self, x, mu, tau, name='normal', prior=False):
	"""Normal log-likelihood"""
	
	if not shape(mu) == shape(tau): raise ParameterError, 'Parameters must have same dimensions in normal_like()'
	
	if ndim(mu) > 1:
		
		return sum([self.normal_like(y, m, t, name, prior) for y, m, t in zip(x, mu, tau)])
	
	else:
		
		# Ensure valid values of parameters
		self.constrain(tau, lower=0)
		
		# Ensure array type
		x = atleast_1d(x)
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
			y = array([rnormal(_mu, _sig) for _mu, _sig in zip(mu, sigma)])
			
			# Generate GOF points
			gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]))
			
			self._gof_loss.append(gof_points)
		
		return sum([fnormal(y, m, t) for y, m, t in zip(x, mu, tau)])

def normal_prior(self, parameter, mu, tau):
	"""Normal prior distribution"""
	
	return self.normal_like(parameter, mu, tau, prior=True)

def half_normal_like(self, x, tau, name='halfnormal', prior=False):
	"""Half-normal log-likelihood"""
	
	if ndim(tau) > 1:
		
		return sum([self.half_normal_like(y, t, name, prior) for y, t in zip(x, tau)])
	
	else:
		
		# Ensure valid values of parameters
		self.constrain(tau, lower=0)
		self.constrain(x, lower=0)
		
		# Ensure array type
		x = atleast_1d(x)
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
			gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]))
			
			self._gof_loss.append(gof_points)
		
		return sum([fhalfnormal(_x, _tau) for _x, _tau in zip(x, tau)])

def half_normal_prior(self, parameter, tau):
	"""Half-normal prior distribution"""
	
	return self.half_normal_like(parameter, tau, prior=True)

def lognormal_like(self, x, mu, tau, name='lognormal', prior=False):
	"""Log-normal log-likelihood"""
	
	if not shape(mu) == shape(tau): raise ParameterError, 'Parameters must have same dimensions in lognormal_like()'
	
	if ndim(mu) > 1:
		
		return sum([self.lognormal_like(y, m, t, name, prior) for y, m, t in zip(x, mu, tau)])
	
	else:
		
		# Ensure valid values of parameters
		self.constrain(tau, lower=0)
		self.constrain(x, lower=0)
		
		# Ensure array type
		x = atleast_1d(x)
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
			y = exp([rnormal(m, s) for m, s in zip(mu, sigma)])
			
			# Generate GOF points
			gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]))
			
			self._gof_loss.append(gof_points)
		
		return sum([flognormal(y, m, t) for y, m, t in zip(x, mu, tau)])

def lognormal_prior(self, parameter, mu, tau):
	"""Log-normal prior distribution"""
	
	return self.lognormal_like(parameter, mu, tau, prior=True)

def multivariate_normal_like(self, x, mu, tau, name='multivariate_normal', prior=False):
	"""Multivariate normal"""
	
	if ndim(tau) > 2:
		
		return sum([self.multivariate_normal_like(y, m, t, name, prior) for y, m, t in zip(x, mu, tau)])
	
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
				gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]))
				
				self._gof_loss.append(gof_points)
		
		return fmvnorm(x, mu, tau)

# Deprecated name
mvnormal_like = multivariate_normal_like

def multivariate_normal_prior(self, parameter, mu, tau):
	"""Multivariate normal prior distribution"""
	
	return self.multivariate_normal_like(parameter, mu, tau, prior=True)

def wishart_like(self, X, n, Tau, name='wishart', prior=False):
	"""Wishart log-likelihood"""
	
	if ndim(Tau) > 2:
		
		return sum([self.wishart_like(x, m, t, name, prior) for x, m, t in zip(X, n, Tau)])
	
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
				gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]))
				
				self._gof_loss.append(gof_points)
		
		return fwishart(X, n, Tau)

def wishart_prior(self, parameter, n, Tau):
	"""Wishart prior distribution"""
	
	return self.wishart_like(parameter, n, Tau, prior=True)

def gumbel_like(self, x, mu, sigma, name='gumbel', prior=False):
	"""Gumbel log-likelihood"""
	
	if not shape(mu) == shape(sigma): raise ParameterError, 'Parameters must have same dimensions'
	
	if ndim(mu) > 1:
		
		return sum([self.gumbel_like(y, m, s, name, prior) for y, m, s in zip(x, mu, sigma)])
	
	else:
		
		# Ensure valid values of parameters
		self.constrain(sigma, lower=0)
		
		# Ensure proper dimensionality of parameters
		x = atleast_1d(x)
		mu = resize(mu, shape(x))
		sigma = resize(sigma, shape(x))
		
		# Goodness-of-fit
		if self._gof and not prior:
			
			try:
				self._like_names.append(name)
			except AttributeError:
				pass
			
			expval = mu + 0.57722 * sigma
			
			# Simulated values
			y = mu - sigma * log(-log(runiform(0, 1, len(x))))
			
			# Generate GOF points
			gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]))
			
			self._gof_loss.append(gof_points)
		
		return fgumbel(x, mu, sigma)

def gumbel_prior(self, x, mu, sigma):
	"""Gumbel prior distribution"""
	
	return self.gumbel_like(self, x, mu, sigma, prior=True)

def weibull_like(self, x, alpha, beta, name='weibull', prior=False):
	"""Weibull log-likelihood"""
	
	if not shape(alpha) == shape(beta): raise ParameterError, 'Parameters must have same dimensions in gamma_like()'
	
	# Allow for multidimensional arguments
	if ndim(alpha) > 1:
		
		return sum([self.weibull_like(y, a, b, name, prior) for y, a, b in zip(x, alpha, beta)])
	
	else:
		
		# Ensure valid values of parameters
		self.constrain(alpha, lower=0)
		self.constrain(beta, lower=0)
		self.constrain(x, lower=0)
		
		# Ensure proper dimensionality of parameters
		x = atleast_1d(x)
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
			gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]))
			
			self._gof_loss.append(gof_points)
		
		return sum([fweibull(y, a, b) for y, a, b in zip(x, alpha, beta)])

def weibull_prior(self, parameter, alpha, beta):
	"""Weibull prior distribution"""
	
	return self.weibull_like(parameter, alpha, beta, prior=True)

def cauchy_like(self, x, alpha, beta, name='cauchy', prior=False):
	"""Cauchy log-likelhood"""
	
	if not shape(alpha) == shape(beta): raise ParameterError, 'Parameters must have same dimensions in gamma_like()'
	
	# Allow for multidimensional arguments
	if ndim(alpha) > 1:
		
		return sum([self.cauchy_like(y, a, b, name, prior) for y, a, b in zip(x, alpha, beta)])
	
	else:
		
		# Ensure valid values of parameters
		self.constrain(beta, lower=0)
		
		# Ensure proper dimensionality of parameters
		x = atleast_1d(x)
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
			y = array([rcauchy(a, b) for a, b in zip(alpha, beta)])
			
			# Generate GOF points
			gof_points = sum(transpose([self.loss(x, expval), self.loss(y, expval)]))
			
			self._gof_loss.append(gof_points)
		
		return sum([fcauchy(y, a, b) for y, a, b in zip(x, alpha, beta)])

def cauchy_prior(self, parameter, alpha, beta):
	"""Cauchy prior distribution"""
	
	return self.cauchy_like(parameter, alpha, beta, prior=True)

