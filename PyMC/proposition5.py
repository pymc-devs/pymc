"""
proposition5.py

Classes:

	PyMCBase:					Abstract base class from which Parameter and PyMCBase inherit.

	Parameter:					Variable whose value is unknown given the value of its parents
								(a random variable under the Bayesian interpretation of probability).
				  
	Node:						Variable whose value is known given the value of its parents.

	SamplingMethod:				Object that knows how to make member variables take one MCMC step.

	OneAtATimeMetropolis:		Default SamplingMethod, instantiated by Model to handle Parameters
								that aren't handled by any existing SamplingMethod.

	Model:					Object that manages an MCMC loop.


Functions:

	weight:			Get posterior probabilities for a list of models.

	parameter:		Decorator used to instantiate a Parameter.

	data:			Decorator used to instantiate a Node.

Basically all the underscored functions should eventually be written in C.

DANGEROUS THINGS:

	-	If a parameter's value is an ndarray, NEVER use parameter.value += new_array. Use
		parameter.value = parameter.value + new_array instead, otherwise last_value will
		get set to value + new_array. I don't know why this happens.

"""
from copy import deepcopy
from numpy import *
from numpy.linalg import cholesky, eigh
import sys, inspect
from numpy.random import randint, random
from numpy.random import normal as rnormal
import database
from decorators import magic_set

def _push(seq,new_value):
	"""
	Usage:
	_push(seq,new_value)

	Put a deep copy of new_value at the beginning of seq, and kick out the last value.
	"""
	length = len(seq)
	for i in range(length-1):
		seq[i+1] = seq[i]
	if isinstance(seq,ndarray):
		# ndarrays will automatically make a copy
		seq[0] = new_value
	else:
		seq[0] = deepcopy(new_value)

def _extract(__func__, kwds, keys):	
	"""
	Used by decorators parameter and node to inspect declarations
	"""
	kwds.update({'doc':__func__.__doc__, 'name':__func__.__name__})

	def probeFunc(frame, event, arg):
		if event == 'return':
			locals = frame.f_locals
			kwds.update(dict((k,locals.get(k)) for k in keys))
			sys.settrace(None)
		return probeFunc

	# Get the __func__tions logp and random (complete interface).
	sys.settrace(probeFunc)
	try:
		__func__()
	except:
		if 'logp' in keys:	
			kwds['logp']=__func__
		else:
			kwds['eval_fun'] = __func__
	if 'logp' in keys:
		if kwds['logp'] is None:
			kwds['logp'] = __func__

	# Build parents dictionary by parsing the __func__tion's arguments.
	(args, varargs, varkw, defaults) = inspect.getargspec(__func__)
	try:
		kwds.update(dict(zip(args[-len(defaults):], defaults)))
	# No parents at all		
	except TypeError: 
		pass	


def parameter(__func__=None, **kwds):
	"""
	Decorator function for instantiating parameters. Usages:
	
	Medium:
	
		@parameter
		def A(value = ., parent_name = .,  ...):
			return foo(value, parent_name, ...)
		
		@parameter(caching=True, tracing=False)
		def A(value = ., parent_name = .,  ...):
			return foo(value, parent_name, ...)
			
	Long:

		@parameter
		def A(value = ., parent_name = .,  ...):
			
			def logp(value, parent_name, ...):
				return foo(value, parent_name, ...)
				
			def random(parent_name, ...):
				return bar(parent_name, ...)
				
	
		@parameter(caching=True, tracing=False)
		def A(value = ., parent_name = .,  ...):
			
			def logp(value, parent_name, ...):
				return foo(value, parent_name, ...)
				
			def random(parent_name, ...):
				return bar(parent_name, ...)
				
	where foo() computes the log-probability of the parameter A
	conditional on its value and its parents' values, and bar()
	generates a random value from A's distribution conditional on
	its parents' values.
	"""

	def instantiate_p(__func__):
		_extract(__func__, kwds, keys)
		return Parameter(**kwds)		
	keys = ['logp','random']

	if __func__ is None:
		return instantiate_p
	else:
		instantiate_p.kwds = kwds
		return instantiate_p(__func__)

	return instantiate_p


def node(__func__ = None, **kwds):
	"""
	Decorator function instantiating nodes. Usage:
	
	@node
	def B(parent_name = ., ...)
		return baz(parent_name, ...)
		
	where baz returns the node B's value conditional
	on its parents.
	"""
	def instantiate_n(__func__):
		_extract(__func__, kwds, keys=[])
		return Node(**kwds)		

	if __func__ is None:
		return instantiate_n
	else:
		instantiate_n.kwds = kwds
		return instantiate_n(__func__)

	return instantiate_n


def data(__func__=None, **kwds):
	"""
	Decorator instantiating data objects. Usage is just like
	parameter.
	"""
	return parameter(__func__, isdata=True, tracing=False, **kwds)

class PyMCBase(object):
	"""
	The base PyMC object. Parameter and Node inherit from this class.

	Externally-accessible attributes:

		parents :		A dictionary containing parents of self with parameter names.
						Parents can be any type.

		parent_values:	A dictionary containing the values of self's parents.
						This descriptor should eventually be written in C.

		children :		A set containing children of self.
						Children must be PyMC objects.

		timestamp :		A counter indicating how many times self's value has been updated.

	PyMCBase should not usually be instantiated directly.

	See also Parameter and Node,
	as well as parameter(), node(), and data().
	"""
	def __init__(self, doc, name, cache_depth = 2, **parents):

		self.parents = parents
		self.children = set()
		self.__doc__ = doc
		self.__name__ = name
		self.timestamp = 0

		self._cache_depth = cache_depth

		# Find self's parents that are nodes, to speed up cache checking,
		# and add self to node parents' children sets

		self._parent_timestamp_caches = {}
		self._pymc_object_parents = {}
		self._parent_values = {}

		# Make sure no parents are None.
		for key in self.parents.iterkeys():
			assert self.parents[key] is not None, self.__name__ + ': Error, parent ' + key + ' is None.'

		# Sync up parents and children, figure out which parents are PyMC
		# objects and which are just objects.
		for key in self.parents.iterkeys():

			if isinstance(self.parents[key],PyMCBase):

				# Add self to this parent's children set
				self.parents[key].children.add(self)

				# Remember that this parent is a PyMCBase
				self._pymc_object_parents[key] = self.parents[key]

				# Initialize a timestamp cache for this parent
				self._parent_timestamp_caches[key] = -1 * ones(self._cache_depth,dtype='int')

				# Record a reference to this parent's value
				self._parent_values[key] = self.parents[key].value

			else:

				# Make a private copy of this parent's value
				self._parent_values[key] = deepcopy(self.parents[key])

		self._recompute = True
		self._value = None
		self._match_indices = zeros(self._cache_depth,dtype='int')
		self._cache_index = 1

	#
	# Define the attribute parent_values.
	#
	# Extract the values of parents that are PyMCBases
	def _get_parent_values(self):
		for item in self._pymc_object_parents.iteritems():
			self._parent_values[item[0]] = item[1].value
		return self._parent_values

	parent_values = property(fget=_get_parent_values)


class Node(PyMCBase):
	"""
	A PyMCBase that is deterministic conditional on its parents.

	Externally-accessible attributes:

	value : Value conditional on parents. Retrieved from cache when possible,
			recomputed only when necessary. This descriptor should eventually
			be written in C.

	Externally-accessible attributes inherited from PyMCBase:

		parents
		children
		timestamp
		parent_values

	To instantiate: see node()

	See also Parameter and PyMCBase,
	as well as parameter(), and data().
	"""

	def __init__(self, eval_fun,  doc, name, tracing=True, caching=False, **parents):

		PyMCBase.__init__(self, doc=doc, name=name, **parents)

		self._eval_fun = eval_fun
		self._value = None
		self._tracing = tracing
		self._caching = caching


		# Caches, if necessary
		if self._caching:
			self._cached_value = []
			for i in range(self._cache_depth): self._cached_value.append(None)

	#
	# Define the attribute value. This should eventually be written in C.
	#
	# See if a recompute is necessary.
	def _check_for_recompute(self):

		# Loop over cache positions
		for index in range(self._cache_depth):
			match = True

			# Loop over parents and try to catch mismatches
			for item in self._pymc_object_parents.iteritems():
				if not self._parent_timestamp_caches[item[0]][index] == item[1].timestamp:
					match = False
					break

			# If no mismatches, load value from current cache position
			if match:
				self._recompute = False
				self._cache_index = index
				return

		# If there are mismatches at every cache position, recompute
		self._recompute = True

	def _get_value(self):

		if self._caching:
			self._check_for_recompute()

			if self._recompute:

				#Recompute
				self._value = self._eval_fun(**self.parent_values)
				self.timestamp += 1

				# Cache
				_push(self._cached_value, self._value)
				for item in self._pymc_object_parents.iteritems():
					_push(self._parent_timestamp_caches[item[0]], item[1].timestamp)

			else: self._value = self._cached_value[self._cache_index]

		else:
			self._value = self._eval_fun(**self.parent_values)
			return self._value

		return self._value

	value = property(fget = _get_value)

class Parameter(PyMCBase):
	"""
	A PyMCBase that is random conditional on its parents.

	Externally-accessible attributes:

		value :		Current value. When changed, timestamp is incremented. Cannot be
					changed if isdata = True. This descriptor should eventually be
					written in C.

		logp :		Current probability of self conditional on parents. Retrieved
					from cache when possible, recomputed only when necessary.
					This descriptor should eventually be written in C.

		isdata :	A flag indicating whether self is data.

	Externally-accessible attributes inherited from PyMCBase:

		parents
		children
		timestamp
		parent_values

	Externally-accessible methods:

		revert():	Return value to last value, decrement timestamp.

		random():	If random_fun is defined, this draws a value for self.value from
					self's distribution conditional on self.parents. Used for
					model averaging.

	To instantiate with isdata = False: see parameter().
	To instantiate with isdata = True: see data().

	See also PyMCBase and Node,
	as well as node().
	"""

	def __init__(self, logp, doc, name, random = None, tracing=True, caching=False, value=None, rseed=False, isdata=False, **parents):

		PyMCBase.__init__(self, doc=doc, name=name, **parents)

		self.isdata = isdata
		self._logp_fun = logp
		self._logp = None
		self.last_value = None
		self._tracing = tracing
		self._caching = caching
		self._random = random
		self._rseed = rseed
		if value is None:
			self._rseed = True

		# Caches, if necessary
		if self._caching:
			self._cached_logp = zeros(self._cache_depth,dtype='float')
			self._self_timestamp_caches = -1 * ones(self._cache_depth,dtype='int')

		if rseed is True:
			self._value = self.random()
		else:
			self._value = value

	#
	# Define the attribute value.
	#
	# NOTE: relative timings:
	#
	# A.value = .2:		22.1s  (16.7s in deepcopy, but a shallow copy just doesn't seem like enough...)
	# A._set_value(.2): 21.2s
	# A._value = .2:	.9s
	#
	# A.value:			1.9s
	# A._get_value():	1.8s
	# A._value:			.9s
	#
	# There's a lot to be gained by writing these in C, but not so much
	# by using direct getters and setters.

	def _get_value(self):
		return self._value

	# Record new value and increment timestamp
	def _set_value(self, value):
		if self.isdata: print 'Warning, data value updated'
		self.timestamp += 1
		# Save a deep copy of current value
		self.last_value = deepcopy(self._value)
		self._value = value

	value = property(fget=_get_value, fset=_set_value)

	#
	# Define attribute logp.
	#
	# NOTE: relative timings (with one-parent, trivial logp function):
	#
	# caching=False:
	# 
	# A.logp:			9.5s
	# A._get_logp():	9.3s
	# A._logp:			.9s
	#
	# caching=True:
	# 
	# A.logp:			16.3s
	# A._get_logp():	15.4s
	# A._logp:			.9s 
	#
	# Again, there's a lot to be gained by writing these in C.

	# _check_for_recompute should eventually be written in Weave, it's pretty
	# time-consuming.
	def _check_for_recompute(self):

		# Loop over indices
		for index in range(self._cache_depth):
			match = True

			# Look for mismatch of self's timestamp
			if not self._self_timestamp_caches[index] == self.timestamp:
				match = False

			if match:
				# Loop over parents and try to catch mismatches
				for item in self._pymc_object_parents.iteritems():
					if not self._parent_timestamp_caches[item[0]][index] == item[1].timestamp:
						match = False
						break

			# If no mismatches, load value from current cache position
			if match:
				self._recompute = False
				self._cache_index = index
				return

		# If there are mismatches at any cache position, recompute
		self._recompute = True

	def _get_logp(self):
		if self._caching:
			self._check_for_recompute()
			if self._recompute:

				#Recompute
				self.last_logp = self._logp
				self._logp = self._logp_fun(self._value, **self.parent_values)

				#Cache
				_push(self._self_timestamp_caches, self.timestamp)
				_push(self._cached_logp, self._logp)
				for item in self._pymc_object_parents.iteritems():
					_push(self._parent_timestamp_caches[item[0]], item[1].timestamp)

			else: self._logp = self._cached_logp[self._cache_index]

		else:
			self.last_logp = self._logp
			self._logp = self._logp_fun(self._value, **self.parent_values)
			return self._logp

		return self._logp

	logp = property(fget = _get_logp)



	#
	# Call this when rejecting a jump.
	#
	def revert(self):
		"""
		Call this when rejecting a jump.
		"""
		self._logp = self.last_logp
		self._value = self.last_value
		self.timestamp -= 1

	#
	# Sample self's value conditional on parents.
	#
	def random(self):
		"""
		Sample self conditional on parents.
		"""
		if self._random:
			self.value = self._random(**self.parent_values)
		else:
			raise AttributeError, self.__name__+' does not know how to draw its value, see documentation'


class SamplingMethod(object):
	"""
	This object knows how to make Parameters take single MCMC steps.
	It's sample() method will be called by Model at every MCMC iteration.

	Externally-accessible attributes:

		nodes:	The Nodes over which self has jurisdiction.

		parameters: The Parameters over which self has jurisdiction which have isdata = False.

		data:		The Parameters over which self has jurisdiction which have isdata = True.

		pymc_objects:		The Nodes and Parameters over which self has jurisdiction.

		children:	The combined children of all PyMCBases over which self has jurisdiction.

		loglike:	The summed log-probability of self's children conditional on all of
					self's PyMCBases' current values. These will be recomputed only as necessary.
					This descriptor should eventually be written in C.

	Externally accesible methods:

		sample():	A single MCMC step for all the Parameters over which self has jurisdiction.
					Must be overridden in subclasses.

		tune():		Tunes proposal distribution widths for all self's Parameters.

	To instantiate a SamplingMethod called S with jurisdiction over a sequence/set N of PyMCBases:

		S = SamplingMethod(N)

	See also OneAtATimeMetropolis and Model.
	"""

	def __init__(self, pymc_objects):

		self.pymc_objects = set(pymc_objects)
		self.nodes = set()
		self.parameters = set()
		self.data = set()
		self.children = set()
		self._asf = .1
		self._accepted = 0
		self._rejected = 0

		# File away the pymc_objects
		for pymc_object in self.pymc_objects:

			# Sort.
			if isinstance(pymc_object,Node):
				self.nodes.add(pymc_object)
			elif isinstance(pymc_object,Parameter):
				if pymc_object.isdata:
					self.data.add(pymc_object)
				else:
					self.parameters.add(pymc_object)

		# Find children, no need to find parents; each pymc_object takes care of those.
		for pymc_object in self.pymc_objects:
			self.children |= pymc_object.children

		self._extend_children()

		self.children -= self.nodes
		self.children -= self.parameters
		self.children -= self.data

	#
	# Must be overridden in subclasses
	#
	def step(self):
		pass

	#
	# Must be overridden in subclasses
	#
	def tune(self):
		pass

	#
	# Find nearest random descendants
	#
	def _extend_children(self):
		need_recursion = False
		node_children = set()
		for child in self.children:
			if isinstance(child,Node):
				self.children |= child.children
				node_children.add(child)
				need_recursion = True
		self.children -= node_children
		if need_recursion:
			self._extend_children()
		return

	#
	# Define attribute loglike.
	#
	def _get_loglike(self):
		sum = 0.
		for child in self.children: sum += child.logp
		return sum

	loglike = property(fget = _get_loglike)

# The default SamplingMethod, which Model uses to handle singleton parameters.
class OneAtATimeMetropolis(SamplingMethod):
	"""
	The default SamplingMethod, which Model uses to handle singleton parameters.

	Applies the one-at-a-time Metropolis-Hastings algorithm to the Parameter over which
	self has jurisdiction.

	To instantiate a OneAtATimeMetropolis called M with jurisdiction over a Parameter P:

		M = OneAtATimeMetropolis(P)

	But you never really need to instantiate OneAtATimeMetropolis, Model does it
	automatically.

	See also SamplingMethod and Model.
	"""
	def __init__(self, parameter, scale=1, dist='Normal'):
		SamplingMethod.__init__(self,[parameter])
		self.parameter = parameter
		self.proposal_sig = ones(shape(self.parameter.value)) * abs(self.parameter.value) * scale
		self._dist = dist

	#
	# Do a one-at-a-time Metropolis-Hastings step self's Parameter.
	#
	def step(self):

		# Probability and likelihood for parameter's current value:
		logp = self.parameter.logp
		loglike = self.loglike

		# Sample a candidate value
		self.propose()

		# Probability and likelihood for parameter's proposed value:
		logp_p = self.parameter.logp

		# Skip the rest if a bad value is proposed
		if logp_p == -Inf:
			self.parameter.revert()
			return

		loglike_p = self.loglike

		# Test
		if log(random()) > logp_p + loglike_p - logp - loglike:
			# Revert parameter if fail
			self.parameter.revert()
			
			self._rejected+=1
		else:
			self._accepted += 1


	def propose(self):
		if self._dist == 'RoundedNormal':
			self.parameter.value = round(rnormal(self.parameter.value, self.proposal_sig))
		# Default to normal random-walk proposal
		else:
			self.parameter.value = rnormal(self.parameter.value, self.proposal_sig)

	#
	# Tune the proposal width.
	#
	def tune(self):
		#
		# Adjust _asf according to some heuristic
		#
		pass

class Joint(SamplingMethod):
	"""
	S = Joint(pymc_objects, epoch=1000, memory=10, interval=1, delay=1000)

	Applies the Metropolis-Hastings algorithm to several parameters
	together. Jumping density is a multivariate normal distribution
	with mean zero and covariance equal to the empirical covariance
	of the parameters, times _asf ** 2.

	Externally-accessible attributes:

		pymc_objects:	A sequence of pymc objects to handle using
						this SamplingMethod.

		interval:		The interval at which S's parameters' values
						should be written to S's internal traces
						(NOTE: If the traces are moved back into the
						PyMC objects, it should be possible to avoid this
						double-tallying. As it stands, though, the traces
						are stored in Model, and SamplingMethods have no
						way to know which Model they're going to be a
						member of.)

		epoch:			After epoch values are stored in the internal
						traces, the covariance is recomputed.

		memory:			The maximum number of epochs to consider when
						computing the covariance.

		delay:			Number of one-at-a-time iterations to do before
						starting to record values for computing the joint
						covariance.

		_asf:			Adaptive scale factor.

	Externally-accessible methods:

		step():			Make a Metropolis step. Applies the one-at-a-time
						Metropolis algorithm until the first time the
						covariance is computed, then applies the joint
						Metropolis algorithm.

		tune():			sets _asf according to a heuristic.

	"""
	def __init__(self, pymc_objects, epoch=1000, memory=10, interval = 1, delay = 0):

		SamplingMethod.__init__(self,pymc_objects)

		self.epoch = epoch
		self.memory = memory
		self.interval = interval
		self.delay = delay

		# Flag indicating whether covariance has been computed
		self._ready = False

		# Use OneAtATimeMetropolis instances to handle independent jumps
		# before first epoch is complete
		self._single_param_handlers = set()
		for parameter in self.parameters:
			self._single_param_handlers.add(OneAtATimeMetropolis(parameter))

		# Allocate memory for internal traces and get parameter slices
		self._slices = {}
		self._len = 0
		for parameter in self.parameters:
			if isinstance(parameter.value, ndarray):
				param_len = len(parameter.value.ravel())
			else:
				param_len = 1
			self._slices[parameter] = slice(self._len, self._len + param_len)
			self._len += param_len
			
		self._trace = zeros((self._len, self.memory * self.epoch),dtype='float')			

		# __init__ should also check that each parameter's value is an ndarray or
		# a numerical type.

	#
	# Compute and store matrix square root of covariance every epoch
	#
	def compute_sig(self):
		
		print 'Joint SamplingMethod ' + self.__name__ + ' computing covariance.'
		
		# Figure out which slice of the traces to use
		if (self._model._cur_trace_index - self.delay) / self.epoch / self.interval > self.memory:
			trace_slice = slice(self._model._cur_trace_index-self.epoch * self.memory,\
								self._model._cur_trace_index, \
								self.interval)
			trace_len = self.memory * self.epoch
		else:
			trace_slice = slice(self.delay, self._model._cur_trace_index, \
								self.interval)
			trace_len = (self._model._cur_trace_index - self.delay) / self.interval
			
		
		# Store all the parameters' traces in self._trace
		for parameter in self.parameters:
			param_trace = parameter.trace(slicing=trace_slice)
			
			# If parameter is an array, ravel each tallied value
			if isinstance(parameter.value, ndarray):
				for i in range(trace_len):
					self._trace[self._slices[parameter], i] = param_trace[i,:].ravel()
			
			# If parameter is a scalar, there's no need.
			else:
				self._trace[self._slices[parameter], :trace_len] = param_trace

		# Compute matrix square root of covariance of self._trace
		self._cov = cov(self._trace[: , :trace_len])
		
		# Try Cholesky factorization
		try:
			self._sig = cholesky(self._cov)
		
		# If there's a small eigenvalue, diagonalize
		except linalg.linalg.LinAlgError:
			val, vec = eigh(self._cov)
			self._sig = vec * sqrt(val)

		self._ready = True



	def tune(self):
		if self._ready:
			for handler in self._single_param_handlers:
				handler.tune()
		else:
			#
			# Adjust _asf according to some heuristic
			#
			pass

	def propose(self):
		# Eventually, round the proposed values for discrete parameters.
		proposed_vals = self._asf * inner(rnormal(size=self._len) , self._sig)
		for parameter in self.parameters:
			parameter.value = parameter.value + reshape(proposed_vals[self._slices[parameter]],shape(parameter.value))

	#
	# Make a step
	#
	def step(self):
		# Step
		if not self._ready:
			for handler in self._single_param_handlers:
				handler.step()
		else:
			# Probability and likelihood for parameter's current value:
			logp = sum([parameter.logp for parameter in self.parameters])
			loglike = self.loglike

			# Sample a candidate value
			self.propose()

			# Probability and likelihood for parameter's proposed value:
			logp_p = sum([parameter.logp for parameter in self.parameters])

			# Skip the rest if a bad value is proposed
			if logp_p == -Inf:
				for parameter in self.parameters:
					parameter.revert()
				return

			loglike_p = self.loglike

			# Test
			if log(random()) > logp_p + loglike_p - logp - loglike:
				# Revert parameter if fail
				self._rejected += 1
				for parameter in self.parameters:
					parameter.revert()
			else:
				self._accepted += 1

		# If an epoch has passed, recompute covariance.
		if	(float(self._model._cur_trace_index - self.delay) / float(self.interval)) % self.epoch == 0 \
			and self._model._cur_trace_index > self.delay:
			self.compute_sig()

class Model(object):
	"""
	Model manages MCMC loops. It is initialized with:

	A = Model(prob_def, dbase=None)

	Arguments
	
		prob_def: class, module or dictionary containing PyMC objects and 
		SamplingMethods)
		
		dbase: Database backend used to tally the samples. 
		Implemented backends: None, hdf5.

	Externally-accessible attributes:

		nodes:			All extant Nodes.

		parameters:			All extant Parameters with isdata = False.

		data:				All extant Parameters with isdata = True.

		pymc_objects:				All extant Parameters and Nodes.

		sampling_methods:	All extant SamplingMethods.

	Externally-accessible methods:

		sample(iter,burn,thin): At each MCMC iteration, calls each sampling_method's step() method.
								Tallies Parameters and Nodes as appropriate.

		trace(parameter, burn, thin, slice): Return the trace of parameter, 
		sliced according to slice or burn and thin arguments.


		remember(trace_index): Return the entire model to the tallied state indexed by trace_index.

		DAG: Draw the model as a directed acyclic graph.

		All the plotting functions can probably go on the base namespace and take Parameters as
		arguments.

	See also SamplingMethod, OneAtATimeMetropolis, PyMCBase, Parameter, Node, and weight.
	"""
	def __init__(self, input, dbase=None):

		self.nodes = set()
		self.parameters = set()
		self.data = set()
		self.sampling_methods = set()
		self._generations = []
		self._prepared = False
		self.__name__ = None

		if hasattr(input,'__name__'):
			self.__name__ = input.__name__

		#Change input into a dictionary
		if isinstance(input, dict):
			input_dict = input
		else:
			try:
				# If input is a module, reload it to make a fresh copy.
				reload(input)
			except TypeError:
				pass

			input_dict = input.__dict__

		for item in input_dict.iteritems():
			self._fileitem(item)
		
		self._assign_trace_methods(dbase)

	def _fileitem(self, item):

		# If a dictionary is passed in, open it up.
		if isinstance(item[1],dict):
			for subitem in item[1].iteritems():
				self._fileitem(subitem)

		# If another iterable object is passed in, open it up.
		# Broadcast the same name over all the elements.
		"""
		This doesn't work so hot, anyone have a better idea?
		I was trying to allow sets/tuples/lists
		of PyMC objects and SamplingMethods to be passed in.

		elif iterable(item[1]) == 1:
			for subitem in item[1]:
				self._fileitem((item[0],subitem))
		"""
		# File away the SamplingMethods
		if isinstance(item[1],SamplingMethod):
			# Teach the SamplingMethod its name
			item[1].__name__ = item[0]
			#File it away
			self.__dict__[item[0]] = item[1]
			self.sampling_methods.add(item[1])
			setattr(self.__dict__[item[0]], '_model', self)

		# File away the PyMC objects
		elif isinstance(item[1],PyMCBase):
			self.__dict__[item[0]] = item[1]
			# Add an attribute to the object referencing the model instance.
			setattr(self.__dict__[item[0]], '_model', self)
			
			if isinstance(item[1],Node):
				self.nodes.add(item[1])

			elif isinstance(item[1],Parameter):
				if item[1].isdata:
					self.data.add(item[1])
				else:  self.parameters.add(item[1])

	#
	# Override __setattr__ so that PyMC objects are read-only once instantiated
	#
	def __setattr__(self, name, value):

		# Don't allow changes to PyMC object attributes
		if self.__dict__.has_key(name):
			if isinstance(self.__dict__[name],PyMCBase):
				raise AttributeError, 'Attempt to write read-only attribute of Model.'

			# Do allow other objects to be changed
			else:
				self.__dict__[name] = value

		# Allow new attributes to be created.
		else:
			self.__dict__[name] = value

	def _assign_trace_methods(self, dbase):
		"""Assign trace method to parameters and nodes. 
		Assign database initialization methods to the Model class.
		
		Defined databases: 
		  - None: Traces stored in memory.
		  - Txt: Traces stored in memory and saved in txt files at end of 
				sampling. Not implemented.
		  - SQLlite: Traces stored in sqllite database. Not implemented. 
		  - HDF5: Traces stored in HDF5 database. Partially implemented.
		"""
		# Load the trace backend.
		if dbase is None:
			dbase = 'memory_trace'
		db = getattr(database, dbase)
		reload(db)
		
		# Assign trace methods to parameters and nodes. 
		for object in self.parameters | self.nodes :
			try:
				for name, method in db.parameter_methods().iteritems():
					magic_set(object, method)
			except TypeError:
				for name, method in db.parameter_methods.iteritems():
					magic_set(object, method)
		
		# Assign database methods to Model.
		try:
			for name, method in db.model_methods().iteritems():
				magic_set(self, method)
		except TypeError:
			for name, method in db.model_methods.iteritems():
				magic_set(self, method)
	#
	# Prepare for sampling
	#
	def _prepare(self):

		# Initialize database
		self._init_dbase()
		
		# Seed new initial values for the parameters.
		for parameters in self.parameters:
			if parameters._rseed:
				parameters.value = parameters.random(**parameters.parent_values)

		if self._prepared:
			return

		self._prepared = True

		# Tell all pymc_objects to get ready for sampling
		self.pymc_objects = self.nodes | self.parameters | self.data
		for pymc_object in self.pymc_objects:
			self._extend_children(pymc_object)

		# Take care of singleton parameters
		for parameter in self.parameters:

			# Is it a member of any SamplingMethod?
			homeless = True
			for sampling_method in self.sampling_methods:
				if parameter in sampling_method.parameters:
					homeless = False
					break

			# If not, make it a new one-at-a-time Metropolis-Hastings SamplingMethod
			if homeless:
				self.sampling_methods.add(OneAtATimeMetropolis(parameter))
				
	#
	# Find PyMC object's random children.
	#
	def _extend_children(self,pymc_object):
		need_recursion = False
		node_children = set()
		for child in pymc_object.children:
			if isinstance(child,Node):
				pymc_object.children |= child.children
				node_children.add(child)
				need_recursion = True
		pymc_object.children -= node_children
		if need_recursion:
			self._extend_children(pymc_object)
		return
				

	#
	# Initialize traces
	#
	def _init_traces(self, length):
		"""
		init_traces(length)

		Enumerates the pymc_objects that are to be tallied and initializes traces
		for them.

		To be tracing, a pymc_object has to pass the following criteria:

			-	It is not included in the argument pymc_objects_not_to_tally.

			-	Its value has a shape.

			-	Its value can be made into a numpy array with a numerical
				dtype.
		"""
		self._traces = {}
		self._pymc_objects_to_tally = set()
		self._cur_trace_index = 0
		self.max_trace_length = length

		for pymc_object in self.pymc_objects:
			if pymc_object._tracing:
				pymc_object._init_trace(length)
				self._pymc_objects_to_tally.add(pymc_object)

	#
	# Tally
	#
	def tally(self):
		"""
		tally()

		Records the value of all tracing pymc_objects.
		"""
		if self._cur_trace_index < self.max_trace_length:
			for pymc_object in self._pymc_objects_to_tally:
				pymc_object.tally(self._cur_trace_index)

		self._cur_trace_index += 1

	#
	# Return to a sampled state
	#
	def remember(self, trace_index = None):
		"""
		remember(trace_index = randint(trace length to date))

		Sets the value of all tracing pymc_objects to a value recorded in
		their traces.
		"""
		if trace_index:
			trace_index = randint(self.cur_trace_index)

		for pymc_object in self._pymc_objects_to_tally:
			pymc_object.value = pymc_object.trace()[trace_index]

	#
	# Run the MCMC loop!
	#
	def sample(self,iter,burn,thin):
		"""
		sample(iter,burn,thin)

		Prepare pymc_objects, initialize traces, run MCMC loop.
		"""

		# Do various preparations for sampling
		self._prepare()

		# Initialize traces
		#self._init_traces((iter-burn)/thin)
		self._init_traces(iter)


		for i in range(iter):

			# Tell all the sampling methods to take a step
			for sampling_method in self.sampling_methods:
				sampling_method.step()

			# # Tally as appropriate.
			# if i > burn and (i - burn) % thin == 0:
			#	self.tally()
			
			self.tally()

			if i % 1000 == 0:
				print 'Iteration ', i, ' of ', iter

		# Tuning, etc.

		# Finalize
		self._finalize_dbase(burn, thin)
		
	def tune(self):
		"""
		Tell all samplingmethods to tune themselves.
		"""
		for sampling_method in self.sampling_methods:
			sampling_method.tune()

	def _parse_generations(self):
		"""
		Parse up the _generations for model averaging.
		"""
		self._prepare()


		# Find root generation
		self._generations.append(set())
		all_children = set()
		for parameter in self.parameters:
			all_children.update(parameter.children & self.parameters)
		self._generations[0] = self.parameters - all_children

		# Find subsequent _generations
		children_remaining = True
		gen_num = 0
		while children_remaining:
			gen_num += 1

			# Find children of last generation
			self._generations.append(set())
			for parameter in self._generations[gen_num-1]:
				self._generations[gen_num].update(parameter.children & self.parameters)

			# Take away parameters that have parents in the current generation.
			thisgen_children = set()
			for parameter in self._generations[gen_num]:
				thisgen_children.update(parameter.children & self.parameters)
			self._generations[gen_num] -= thisgen_children

			# Stop when no subsequent _generations remain
			if len(thisgen_children) == 0:
				children_remaining = False

	def sample_model_likelihood(self,iter):
		"""
		Returns iter samples of (log p(data|this_model_params, this_model) | data, this_model)
		"""
		loglikes = zeros(iter)

		if len(self._generations) == 0:
			self._parse_generations()
		for i in range(iter):
			if i % 10000 == 0:
				print 'Sample ',i,' of ',iter

			for generation in self._generations:
				for parameter in generation:
					parameter.random()

			for datum in self.data:
				loglikes[i] += datum.logp

		return loglikes

	def DAG(self,format='raw',path=None):
		"""
		DAG(format='raw', path=None)

		Draw the directed acyclic graph for this model and writes it to path.
		If self.__name__ is defined and path is None, the output file is
		./'name'.'format'. If self.__name__ is undefined and path is None,
		the output file is ./model.'format'.

		Format is a string. Options are:
		'ps', 'ps2', 'hpgl', 'pcl', 'mif', 'pic', 'gd', 'gd2', 'gif', 'jpg',
		'jpeg', 'png', 'wbmp', 'ismap', 'imap', 'cmap', 'cmapx', 'vrml', 'vtx', 'mp',
		'fig', 'svg', 'svgz', 'dia', 'dot', 'canon', 'plain', 'plain-ext', 'xdot'

		format='raw' outputs a GraphViz dot file.
		"""

		if not self._prepared:
			self._prepare()

		if self.__name__ == None:
			self.__name__ = model

		import pydot

		self.dot_object = pydot.Dot()

		pydot_nodes = {}
		pydot_subgraphs = {}

		# Create the pydot nodes from pymc objects
		for datum in self.data:
			pydot_nodes[datum] = pydot.Node(name=datum.__name__,shape='box')
			self.dot_object.add_node(pydot_nodes[datum])

		for parameter in self.parameters:
			pydot_nodes[parameter] = pydot.Node(name=parameter.__name__)
			self.dot_object.add_node(pydot_nodes[parameter])

		for node in self.nodes:
			pydot_nodes[node] = pydot.Node(name=node.__name__,shape='invtriangle')
			self.dot_object.add_node(pydot_nodes[node])

		# Create subgraphs from pymc sampling methods
		for sampling_method in self.sampling_methods:
			if not isinstance(sampling_method,OneAtATimeMetropolis):
				pydot_subgraphs[sampling_method] = subgraph(graph_name = sampling_method.__class__.__name__)
				for pymc_object in sampling_method.pymc_objects:
					pydot_subgraphs[sampling_method].add_node(pydot_nodes[pymc_object])
				self.dot_object.add_subgraph(pydot_subgraphs[sampling_method])


		# Create edges from parent-child relationships
		counter = 0
		for pymc_object in self.pymc_objects:
			for key in pymc_object.parents.iterkeys():
				if not isinstance(pymc_object.parents[key],PyMCBase):

					parent_name = pymc_object.parents[key].__class__.__name__ + ' const ' + str(counter)
					self.dot_object.add_node(pydot.Node(name = parent_name, shape = 'trapezium'))
					counter += 1
				else:
					parent_name = pymc_object.parents[key].__name__

				new_edge = pydot.Edge(	src = parent_name,
										dst = pymc_object.__name__,
										label = key)


				self.dot_object.add_edge(new_edge)

		# Draw the graph
		if not path == None:
			self.dot_object.write(path=path,format=format)
		else:
			ext=format
			if format=='raw':
				ext='dot'
			self.dot_object.write(path='./' + self.__name__ + '.' + ext,format=format)

#
# Get posterior probabilities for a list of models
#
def weight(models, iter, priors = None):
	"""
	weight(models, iter, priors = None)

	models is a list of models, iter is the number of samples to use, and
	priors is a dictionary of prior weights keyed by model.

	Example:

	M1 = Model(model_1)
	M2 = Model(model_2)
	weight(models = [M1,M2], iter = 100000, priors = {M1: .8, M2: .2})

	Returns a dictionary keyed by model of the model posterior probabilities.

	Need to attach an MCSE value to the return values!
	"""
	loglikes = {}
	i=0
	for model in models:
		print 'Model ', i
		loglikes[model] = model.sample_model_likelihood(iter)
		i+=1

	# Find max log-likelihood for regularization purposes
	max_loglike = 0
	for model in models:
		max_loglike = max((max_loglike,loglikes[model].max()))

	posteriors = {}
	sumpost = 0
	for model in models:
		# Regularize
		loglikes[model] -= max_loglike
		# Exponentiate and average
		posteriors[model] = mean(exp(loglikes[model]))
		# Multiply in priors
		if priors is not None:
			posteriors[model] *= priors[model]
		# Count up normalizing constant
		sumpost += posteriors[model]

	# Normalize
	for model in models:
		posteriors[model] /= sumpost

	return posteriors




