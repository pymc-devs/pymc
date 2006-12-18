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
	
	add_draw_fun:	Decorator used to teach Parameters to sample their value conditional 
					on their parents.

"""
from copy import deepcopy
from numpy import *
from numpy.random import randint, random
from numpy.random import normal as rnormal
from scipy import weave
from scipy.weave import converters
from weakref import proxy

def _push(seq,new_value):
	"""
	Usage:
	_push(seq,new_value)
	
	Put a deep copy of new_value at the beginning of seq, and kick out the last value.
	"""
	length = len(seq)
	seq[1:length] = seq[0:length-1]
	if isinstance(seq,ndarray):
		# ndarrays will automatically make a copy
		seq[0] = new_value
	else:
		seq[0] = deepcopy(new_value)

def add_draw_fun(draw_fun):
	"""
	Teach a Parameter how to sample its value conditional on its parents.

	Usage:
	
	@add_draw_fun(draw_fun)
	@parameter(	init_val, traceable = True, caching = True,
				parent_1_name = some object, parent_2_name = some object, ...):
	def P(value, parent_1_name, parent_2_name, ...):
		return foo(value, parent_1_name, parent_2_name, ...):
		
	Example:
	
	@add_draw_fun(lambda mu,tau: rnormal(mu,1/tau)):
	@parameter(init_val = value, mu = A, tau = 6.0):
	def P(12.3, mu, tau):
		return normal_like(value, mu[10,63], tau)
		
	where A is an Array or an array-valued PyMC object.
		
	
	"""
	def _add_draw_fun(obj):
		if isinstance(obj,Parameter):
			obj.draw_fun = draw_fun
		else:
			raise TypeError, 'Decorator add_draw_fun can only be applied to Parameters.'
		return obj
			
	return _add_draw_fun

def parameter(**kwargs):
	"""
	Decorator function instantiating the Parameter class. Usage:
	
	@parameter(	init_val, traceable = True, caching = True,
				parent_1_name = some object, parent_2_name = some object, ...):
	def P(value, parent_1_name, parent_2_name, ...):

		return foo(value, parent_1_name, parent_2_name, ...):
		
	will create a Parameter named P whose log-probability given its parents
	is computed by foo. If draw is defined, it will be used when this Parameter
	is asked to sample itself from its prior.
	
		init_val:		The initial value of the parameter. Required.
		
		traceable:		Whether Model should make a trace for this Parameter.
		
		caching:		Whether this Parameter should avoid recomputing its probability
						by caching previous results.
					
		parent_i_name:	The label of parent i. See example.
	
	Example:
	
	@parameter(init_val = value, mu = A, tau = 6.0):
	def P(12.3, mu, tau):
		
		return normal_like(value, mu[10,63], tau)
		
	creates a parameter called P with two parents, A and 6.0. P.value will be set to
	12.3. When P.prob is computed, it will be set to 
	
	normal_like(P.value, A[10,63], 6.0)				if A is a numpy ndarray
	
	OR
	
	normal_like(P.value, A.value[10,63], 6.0)		if A is a PyMC object.
	
	See also data() and node(),
	as well as Parameter, Node and PyMCBase.	
	"""			
	
	def __instantiate_parameter(f):
		P = Parameter(prob_fun=f,**kwargs)
		P.__doc__ = f.__doc__
		P.__name__ = f.__name__
		return P
		
	return __instantiate_parameter


def node(**kwargs):
	"""
	Decorator function instantiating the Node class. Usage:
	
	@Node(	traceable=True, caching=True, 
				parent_1_name = some object, parent_2_name = some object, ...):
	def N(parent_1_name, parent_2_name, ...):
		return bar(parent_1_name, parent_2_name, ...):
		
	will create a Node named N whose value is computed by bar based on the 
	parent objects that were passed into the decorator. 
	
		traceable:		Whether Model should make a trace for this Node.
	
		caching:		Whether this Node should avoid recomputing its value
						by caching previous results.
				
		parent_i_name:	The label of parent i. See example.
	
	
	Example:
	
	@node(p = .176, q = B):
	def N(p, q):
		return p * q[132]
		
	creates a Node called N with two parents, .176 and B. 
	When N.value is computed, it will be set to 
	
	B[132] * .176			if B is a numpy ndarray
	
	OR
	
	B.value[132] * .176		if B is a PyMC object.
	
	See also parameter() and data(),
	as well as Parameter, Node and PyMCBase.	
	"""

	def __instantiate_node(f):
		N =Node(eval_fun=f,**kwargs)
		N.__doc__ = f.__doc__
		N.__name__ = f.__name__
		return N
		
	return __instantiate_node
	

def data(**kwargs):
	"""
	Decorator function instantiating the Parameter class with the 'isdata' flag set to True. 
	That means that the attribute value cannot be changed after instantiation. Usage:

	@data(init_val = some value, parent_1_name = some object, parent_2_name = some object, ...):
	def D(value, parent_1_name, parent_2_name, ...):
		return foo(value, parent_1_name, parent_2_name, ...):

	will create a Parameter named D whose log-probability is computed by foo, with the property
	that P.value cannot be changed from init_val. Example:

	@data(init_val = value, mu = A, tau = 6.0):
	def D(12.3, mu, tau):
		return normal_like(value, mu[10,63], tau)

	creates a parameter called D with two parents, A and 6.0. D.value will be set to
	12.3 forever. When D.prob is computed, it will be set to 

	normal_like(D.value, A[10,63], 6.0)				if A is a numpy ndarray

	OR

	normal_like(D.value, A.value[10,63], 6.0)		if A is a Node.

	See also parameter() and node(),
	as well as Parameter, Node and PyMCBase.
	"""

	def __instantiate_data(f):
		D = Parameter(prob_fun=f,isdata=True,**kwargs)
		D.__doc__ = f.__doc__
		D.__name__ = f.__name__
		return D

	return __instantiate_data


	
class PyMCBase(object):
	"""
	The base PyMC object. Paramete and Node inherit from this class.
	
	Externally-accessible attributes:
	
		parents :		A dictionary containing parents of self with parameter names.
						Parents can be any type.
					
		parent_values:	A dictionary containing the values of self's parents.
						This descriptor should eventually be written in C.
					
		children :		A set containing children of self.
						Children must be PyMC objects.
					
		timestamp :		A counter indicating how many times self's value has been updated.
		
	Externally-accessible methods:
	
		init_trace(length):	Initializes trace of given length.
	
		tally():			Writes current value of self to trace.
		
	PyMCBase should not usually be instantiated directly.
		
	See also Parameter and Node,
	as well as parameter(), node(), and data().
	"""
	def __init__(self, cache_depth = 2, **parents):

		self.parents = parents
		self.children = set()
		self.timestamp = 0
						
		self._cache_depth = cache_depth
		
		# Find self's parents that are nodes, to speed up cache checking,
		# and add self to node parents' children sets		

		self._parent_timestamp_caches = {}
		self._pymc_object_parent_keys = set()		
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
				self._pymc_object_parent_keys.add(key)
				
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
		for key in self._pymc_object_parent_keys:
			self._parent_values[key] = self.parents[key].value
		return self._parent_values
			
	parent_values = property(fget=_get_parent_values)

	#		
	# Overrideable, if anything needs to be done after Model is initialized.
	#
	def _prepare(self):
		pass

	#		
	# Find self's random children. Will be called by Model.__init__().
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


class Node(PyMCBase):
	"""
	A PyMCBase that is deterministic conditional on its parents.

	Externally-accessible attributes:

	value :	Value conditional on parents. Retrieved from cache when possible,
			recomputed only when necessary. This descriptor should eventually
			be written in C.

	Externally-accessible attributes inherited from PyMCBase:

		parents
		children
		timestamp
		parent_values

	Externally-accessible methods inherited from PyMCBase:

		init_trace(length)
		tally()

	To instantiate: see node()

	See also Parameter and PyMCBase,
	as well as parameter(), and data().
	"""

	def __init__(self, eval_fun, traceable=True, caching=False, **parents):

		PyMCBase.__init__(self, **parents)

		self._eval_fun = eval_fun
		self._value = None
		self._traceable = traceable
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
		indices = range(self._cache_depth)
		for key in self._pymc_object_parent_keys:
			indices = where(self._parent_timestamp_caches[key][indices] == self.parents[key].timestamp)[0]

		if len(indices)==0:
			self._recompute = True				
		else:
			self._recompute = False
			self._cache_index = indices[0]
		return

	def _get_value(self):

		if self._caching:
			self._check_for_recompute()

			if self._recompute:

				#Recompute
				self._value = self._eval_fun(**self.parent_values)
				self.timestamp += 1

				# Cache
				_push(self._cached_value, self._value)			
				for key in self._pymc_object_parent_keys:
					_push(self._parent_timestamp_caches[key], self.parents[key].timestamp)

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

		prob :		Current probability of self conditional on parents. Retrieved
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
		
		draw():		If draw_fun is defined, this draws a value for self.value from
					self's distribution conditional on self.parents. Used for
					model averaging.
		
	Externally-accessible methods inherited from PyMCBase:
	
		init_trace(length)
		tally()		
		
	To instantiate with isdata = False: see parameter().
	To instantiate with isdata = True: see data().
	
	See also PyMCBase and Node,
	as well as node().
	"""

	def __init__(self, prob_fun, traceable=True, caching=False, init_val=0, isdata=False, **parents):

		PyMCBase.__init__(self, **parents)

		self.isdata = isdata
		self._prob_fun = prob_fun
		self._prob = None
		self._last_value = None
		self._traceable = traceable
		self._caching = caching
		
		# Caches, if necessary
		if self._caching:
			self._cached_prob = zeros(self._cache_depth,dtype='float')
			self._self_timestamp_caches = -1 * ones(self._cache_depth,dtype='int')

		self._value = init_val

	#	
	# Define the attribute value.
	#
	def _get_value(self, *args, **kwargs):
		return self._value
	
	# Record new value and increment timestamp
	def _set_value(self, value):
		if self.isdata: print 'Warning, data value updated'
		self.timestamp += 1
		# Save a deep copy of current value
		self._last_value = self._value
		self._value = value
		
	value = property(fget=_get_value, fset=_set_value)

	#
	# Define attribute prob.
	#
	
	# _check_for_recompute should eventually be written in Weave, it's pretty
	# time-consuming.
	def _check_for_recompute(self):

		indices = where(self._self_timestamp_caches == self.timestamp)[0]

		for key in self._pymc_object_parent_keys:
			indices = where(self._parent_timestamp_caches[key][indices] == self.parents[key].timestamp)[0]

		if len(indices)==0:
			self._recompute = True
		else:
			self._recompute = False
			self._cache_index = indices[0]
		return

	def _get_prob(self):
		if self._caching:
			self._check_for_recompute()
			if self._recompute:

				#Recompute
				self._last_prob = self._prob				
				self._prob = self._prob_fun(self._value, **self.parent_values)
			
				#Cache
				_push(self._self_timestamp_caches, self.timestamp)
				_push(self._cached_prob, self._prob)
				for key in self._pymc_object_parent_keys:
					_push(self._parent_timestamp_caches[key], self.parents[key].timestamp)

			else: self._prob = self._cached_prob[self._cache_index]					
			
		else:
			self._last_prob = self._prob
			self._prob = self._prob_fun(self._value, **self.parent_values)
			return self._prob
			
		return self._prob
		
	prob = property(fget = _get_prob)


		
	#	
	# Call this when rejecting a jump.
	#
	def revert(self):
		"""
		Call this when rejecting a jump.
		"""
		self._prob = self._last_prob
		self._value = self._last_value
		self.timestamp -= 1

	#
	# Sample self's value conditional on parents.
	#
	def draw(self):
		"""
		Sample self conditional on parents.
		"""
		if hasattr(self,'draw_fun'):
			self.value = self.draw_fun(**self.parent_values)
		else:
			raise AttributeError, self.__name__+' does not know how to draw its value, use decorator add_draw_fun.'
			

# Was SubModel:
class SamplingMethod(object):
	"""
	This object knows how to make Parameters take single MCMC steps.
	It's sample() method will be called by Model at every MCMC iteration.

	Externally-accessible attributes:
				
		nodes:	The Nodes over which self has jurisdiction.
		
		parameters:	The Parameters over which self has jurisdiction which have isdata = False.
		
		data:		The Parameters over which self has jurisdiction which have isdata = True.
		
		pymc_objects:		The Nodes and Parameters over which self has jurisdiction.
		
		children:	The combined children of all PyMCBases over which self has jurisdiction.
		
		likelihood:	The summed log-probability of self's children conditional on all of 
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
	# Define attribute likelihood
	#
	def _get_likelihood(self):
		return sum([child.prob for child in self.children])

	likelihood = property(fget = _get_likelihood)



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
	def __init__(self, parameter, scale=.1, dist='Normal'):
		SamplingMethod.__init__(self,[parameter])
		self.parameter = parameter
		self.proposal_sig = ones(shape(self.parameter.value)) * abs(self.parameter.value) * scale
		self._dist = dist

	#
	# Do a one-at-a-time Metropolis-Hastings step self's Parameter.
	#		
	def step(self):
		
		# Probability and likelihood for parameter's current value:
		prob = self.parameter.prob
		like = self.likelihood
		
		# Sample a candidate value
		self.propose()
		
		# Probability and likelihood for parameter's proposed value:
		prob_p = self.parameter.prob
		
		# Skip the rest if a bad value is proposed
		if prob_p == -Inf:
			self.parameter.revert()
			return
		
		like_p = self.likelihood
		
		# Test
		if log(random()) > prob_p + like_p - prob - like:
			
			# Revert parameter if fail
			self.parameter.revert()
		
	def propose(self):
		if self._dist == 'RoundedNormal':
			self.parameter.value += round(rnormal(0,self.proposal_sig))
		
		# Default to normal random-walk proposal
		else:
			self.parameter.value += rnormal(0,self.proposal_sig)

	#
	# Tune the proposal width.
	#
	def tune(self):
		pass

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

		
# Constructor of Model doesn't need any arguments, it searches the base namespace for
# any PyMC objects and files them away for sampling commands.
class Model(object):
	"""
	Model manages MCMC loops. It is initialized with no arguments:
	
	A = Model(class, module or dictionary containing PyMC objects and SamplingMethods)
	
	Externally-accessible attributes:
	
		nodes:			All extant Nodes.
		
		parameters:			All extant Parameters with isdata = False.
		
		data:				All extant Parameters with isdata = True.
		
		pymc_objects:				All extant Parameters and Nodes.
		
		sampling_methods:	All extant SamplingMethods.
		
	Externally-accessible methods:
	
		sample(iter,burn,thin):	At each MCMC iteration, calls each sampling_method's step() method.
								Tallies Parameters and Nodes as appropriate.
								
		trace(parameter, slice): Return the trace of parameter, sliced according to slice.
		
		remember(trace_index): Return the entire model to the tallied state indexed by trace_index.
		
		DAG: Draw the model as a directed acyclic graph.
								
		All the plotting functions can probably go on the base namespace and take Parameters as
		arguments.
		
	See also SamplingMethod, OneAtATimeMetropolis, PyMCBase, Parameter, Node, and weight.
	"""

	def __init__(self, input):
	
		self.nodes = set()
		self.parameters = set()
		self.data = set()
		self.sampling_methods = set()
		self._generations = []
		self._prepared = False
		
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

		# File away the PyMC objects
		elif isinstance(item[1],PyMCBase):
			self.__dict__[item[0]] = item[1]
			
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
				

	#
	# Prepare for sampling
	#
	def _prepare(self):
		
		if self._prepared == True:
			return
			
		self._prepared = True

		# Tell all pymc_objects to get ready for sampling		
		self.pymc_objects = self.nodes | self.parameters | self.data
		for pymc_object in self.pymc_objects:
			pymc_object._extend_children()
			pymc_object._prepare()		

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
	# Return a trace
	#
	def trace(self, object, slice = None):

		"""

		Notation: Model is M, Parameter is P

		To return the trace of an P's entire value:
		
			M.trace(M.A)
		
		To the trace of a slice of A's value:
		
			M.trace(M.A, slice)
				
		This notation is not very nice, it would be better to do
		"""
			
		if slice:
			return self._traces[object]
		else:
			return self._traces[object][slice]

	#
	# Initialize traces
	#
	def _init_traces(self, length):
		"""
		init_traces(length)

		Enumerates the pymc_objects that are to be tallied and initializes traces
		for them. 
		
		To be tallyable, a pymc_object has to pass the following criteria:

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
			# Try initializing the trace and writing the current value.
			try:
	
				# Check condition 1 from docstring
				if pymc_object._traceable:

					# Check conditions 2 and 3 from docstring.
					# This is kind of hacky, how the #$^&* do you concatenate
					# a scalar and a tuple?
					self._traces[pymc_object] = zeros(((0,length) + shape(pymc_object.value))[1:])

					# This is kind of hacky too. I want to 1) check that array() works, and
					# 2) silently upcast the trace to the type of the pymc_object ahead of time.
					self._traces[pymc_object][0,] = 0 * array(pymc_object.value)
			
					self._pymc_objects_to_tally.add(pymc_object) 

			except TypeError:
				if self._traces.has_key(pymc_object):
					self._traces.pop(pymc_object)					
		

	#
	# Tally
	#					
	def tally(self):
		"""
		tally()
		
		Records the value of all tallyable pymc_objects.
		"""
		if self._cur_trace_index < self.max_trace_length:
			for pymc_object in self._pymc_objects_to_tally:
				self._traces[pymc_object][self._cur_trace_index,] = pymc_object.value
			
		self._cur_trace_index += 1
		
		
	#
	# Return to a sampled state
	#
	def remember(self, trace_index = None):
		"""
		remember(trace_index = randint(trace length to date))
		
		Sets the value of all tallyable pymc_objects to a value recorded in 
		their traces.
		"""
		if trace_index:
			trace_index = randint(self.cur_trace_index)
			
		for pymc_object in self._pymc_objects_to_tally:
			pymc_object.value = self._traces[pymc_object][trace_index,]
			
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
		self._init_traces((iter-burn)/thin)		

		
		for i in range(iter):
			
			# Tell all the sampling methods to take a step
			for sampling_method in self.sampling_methods:
				sampling_method.step()
								
			# Tally as appropriate.
			if i > burn and (i - burn) % thin == 0:
				self.tally()
				
			if i % 1000 == 0:
				print 'Iteration ', i, ' of ', iter
			
		# Tuning, etc.
		
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
					parameter.draw()

			for datum in self.data:
				loglikes[i] += datum.prob
				
		return loglikes
		
	def DAG(self):
		"""
		Draw the directed acyclic graph for this model.
		"""
		if not self._prepared:
			self._prepare()
		
		# First put down the objects
		for datum in self.data:
			print 'Draw square with ' + datum.__name__ + ' written on it.'
			
		for parameter in self.parameters:
			print 'Draw ellipse with ' + parameter.__name__ + ' written on it.'
			
		for node in self.nodes:
			print 'Draw triangle with ' + node.__name__ + ' written on it.'
			
		# Cluster the samplingmethod members	
		for sampling_method in self.sampling_methods:
			print 'Cluster the following: '
			for pymc_object in sampling_method.pymc_objects:
				print pymc_object.__name__ 
			
		# Draw the parent-child arrows
		for pymc_object in self.pymc_objects:
			for key in pymc_object.parents.iterkeys():
				if isinstance(pymc_object.parents[key],PyMCBase):
					print 'Draw an arrow from ' + pymc_object.parents[key].__name__ + ' to ' + pymc_object.__name__\
							+ ' with ' + key + ' written on it. '
				else:
					print 'Draw a pentagon with ', type(pymc_object.parents[key])
					print 'Then draw an arrow from the pentagon to ' + pymc_object.__name__\
							+ ' with ' + key + ' written on it. '
		
