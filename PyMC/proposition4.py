"""
Classes:

	Node
	Parameter
	Logical

	SamplingMethod
	OneAtATimeMetropolis

	Sampler

Class-instantiation decorators:

	parameter
	logical
	data
	
Support function:

	push
	
For usage examples, see test_prop_4.py


"""
from copy import deepcopy
from numpy import *
from scipy import weave
from scipy.weave import converters
#from check_for_recompute import checkparameter

def push(seq,new_value):
	"""
	Usage:
	push(seq,new_value)
	
	Put a deep copy of new_value at the beginning of seq, and kick out the last value.
	
	This should eventually be written in C.
	"""
	length = len(seq)
	seq[1:length] = seq[0:length-1]
	seq[0] = deepcopy(new_value)


def parameter(**kwargs):
	"""
	Decorator function instantiating the Parameter class. Usage:
	
	@parameter(init_val = some value, parent_1_name = some object, parent_2_name = some object, ...):
	def P(value, parent_1_name, parent_2_name, ...):
		return foo(value, parent_1_name, parent_2_name, ...):
		
	will create a Parameter named P whose log-probability is computed by foo based on the parent
	objects that were passed into the decorator. Example:
	
	@parameter(init_val = value, mu = A, tau = 6.0):
	def P(12.3, mu, tau):
		return normal_like(value, mu[10,63], tau)
		
	creates a parameter called P with two parents, A and 6.0. P.value will be set to
	12.3. When P.prob is computed, it will be set to 
	
	normal_like(P.value, A[10,63], 6.0)				if A is a numpy ndarray
	
	OR
	
	normal_like(P.value, A.value[10,63], 6.0)		if A is a Node.
	
	See also data() and logical(),
	as well as Parameter, Logical and Node.	
	"""

	def __instantiate_parameter(f):
		P = Parameter(prob_fun=f,**kwargs)
		P.__doc__ = f.__doc__
		return P
		
	return __instantiate_parameter


def logical(**kwargs):
	"""
	Decorator function instantiating the Logical class. Usage:
	
	@logical(parent_1_name = some object, parent_2_name = some object, ...):
	def L(parent_1_name, parent_2_name, ...):
		return bar(parent_1_name, parent_2_name, ...):
		
	will create a Logical named L whose value is computed by bar based on the parent objects that
	were passed into the decorator. Example:
	
	@logical(p = .176, q = B):
	def L(p, q):
		return p * q[132]
		
	creates a Logical called L with two parents, .176 and B. When L.value is computed, it will be set to 
	
	B[132] * .176			if B is a numpy ndarray
	
	OR
	
	B.value[132] * .176		if B is a Node.
	
	See also parameter() and data(),
	as well as Parameter, Logical and Node.	
	"""

	def __instantiate_logical(f):
		L =Logical(eval_fun=f,**kwargs)
		L.__doc__ = f.__doc__
		return L
		
	return __instantiate_logical
	

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
	
	See also parameter() and logical(),
	as well as Parameter, Logical and Node.
	"""

	def __instantiate_data(f):
		D = Parameter(prob_fun=f,isdata=True,**kwargs)
		D.__doc__ = f.__doc__
		return D
		
	return __instantiate_data


	
class Node(object):
	"""
	The base PyMC object. Logical and Parameter inherit from this class.
	
	Externally-accessible attributes:
	
		parents :		A dictionary containing parents of self with parameter names.
						Parents can be any type.
					
		parent_values:	A dictionary containing the values of self's parents.
						This descriptor should eventually be written in C.
					
		children :		A set containing children of self.
						Children must be of type Node.
					
		timestamp :		A counter indicating how many times self's value has been updated.
		
	Externally-accessible methods:
	
		init_trace(length):	Initializes trace of given length.
	
		tally():			Writes current value of self to trace.
		
	Node should not usually be instantiated directly.
		
	See also Parameter and Logical,
	as well as parameter(), logical(), and data().
	"""
	def __init__(self, cache_depth = 2, **parents):

		self.parents = parents
		self.children = set()
		self.timestamp = 0		
		self._cache_depth = cache_depth
		
		# Find self's parents that are nodes, to speed up cache checking,
		# and add self to node parents' children sets		

		self._parent_timestamp_caches = {}
		self._node_parent_keys = set()		
		self._parent_values = {}
		
		for key in self.parents.keys():

			if isinstance(self.parents[key],Node):
			
				# Add self to this parent's children set
				self.parents[key].children.add(self)
				
				# Remember that this parent is a Node
				self._node_parent_keys.add(key)
				
				# Initialize a timestamp cache for this parent
				self._parent_timestamp_caches[key] = -1 * ones(self._cache_depth,dtype='int')
				
			else:
				
				# Record a reference to this value
				self._parent_values[key] = self.parents[key]

		self._recompute = True
		self._value = None
		self._match_indices = zeros(self._cache_depth,dtype='int')
		self._cache_index = 1
	
#
# Define the attribute parent_values. This should be eventually be written in C.
#
	# Extract the values of parents that are Nodes
	def _get_parent_values(self):
		for key in self._node_parent_keys:
			self._parent_values[key] = self.parents[key].value
		return self._parent_values
			
	parent_values = property(fget=_get_parent_values)

#		
# Overrideable, if anything needs to be done after Sampler is initialized.
#
	def _prepare(self):
		pass

#		
# Find self's random children. Will be called by Sampler.__init__().
#
	def _extend_children(self):
		need_recursion = False
		logical_children = set()
		for child in self.children:
			if isinstance(child,Logical):
				self.children |= child.children
				logical_children.add(child)
				need_recursion = True
		self.children -= logical_children
		if need_recursion:
			self._extend_children()
		return
		
#
# Not implemented yet
#		
	def init_trace(self, length):
		pass

	def tally(self):
		pass
		
		
		

# Nodes that are deterministic conditional on their parents
class Logical(Node):
	"""
	A Node that is deterministic conditional on its parents.
	
	Externally-accessible attributes:
	
	value :	Value conditional on parents. Retrieved from cache when possible,
			recomputed only when necessary. This descriptor should eventually
			be written in C.
			
	Externally-accessible attributes inherited from Node:
	
		parents
		children
		timestamp
		parent_values
	
	Externally-accessible methods inherited from Node:
	
		init_trace(length)
		tally()
	
	To instantiate: see logical()
	
	See also Parameter and Node,
	as well as parameter(), and data().
	"""

	def __init__(self, eval_fun, **parents):

		Node.__init__(self,**parents)

		self._eval_fun = eval_fun
		self._value = None
		
		# Caches
		self._cached_value = []
		for i in range(self._cache_depth): self._cached_value.append(None)

#		
# Define the attribute value. This should eventually be written in C.
#
	# See if a recompute is necessary.
	def _check_for_recompute(self):
		indices = range(self._cache_depth)
		for key in self._node_parent_keys:
			indices = where(self._parent_timestamp_caches[key][indices] == self.parents[key].timestamp)[0]

		if len(indices)==0:
			self._recompute = True				
		else:
			self._recompute = False
			self._cache_index = indices[0]
		return

	def _get_value(self, *args, **kwargs):

		self._check_for_recompute()

		if self._recompute:

			#Recompute
			self._value = self._eval_fun(**self.parent_values)
			self.timestamp += 1

			# Cache
			push(self._cached_value, self._value)			
			for key in self._node_parent_keys:
				push(self._parent_timestamp_caches[key], self.parents[key].timestamp)

		else: self._value = self._cached_value[self._cache_index]
		return self._value
		
	value = property(fget=_get_value)


class Parameter(Node):
	"""
	A Node that is random conditional on its parents.
	
	Externally-accessible attributes:
	
		value :		Current value. When changed, timestamp is incremented. Cannot be
					changed if isdata = True. This descriptor should eventually be
					written in C.

		prob :		Current probability of self conditional on parents. Retrieved
					from cache when possible, recomputed only when necessary.
					This descriptor should eventually be written in C.
					
		isdata :	A flag indicating whether self is data.

	Externally-accessible attributes inherited from Node:
	
		parents
		children
		timestamp	
		parent_values

	Externally-accessible methods:

		revert():	Return value to last value, decrement timestamp.
		
	Externally-accessible methods inherited from Node:
	
		init_trace(length)
		tally()		
		
	To instantiate with isdata = False: see parameter().
	To instantiate with isdata = True: see data().
	
	See also Node and Logical,
	as well as logical().
	"""

	def __init__(self, prob_fun, init_val=0, isdata=False, **parents):

		Node.__init__(self, **parents)

		self.isdata = isdata
		self._prob_fun = prob_fun
		self._prob = None
		self._last_value = None
		
		# Caches
		self._cached_prob = zeros(self._cache_depth,dtype='float')
		self._self_timestamp_caches = -1 * ones(self._cache_depth,dtype='int')

		if init_val:
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
		self._last_value = deepcopy(self._value)
		self._value = value
		
	value = property(fget=_get_value, fset=_set_value)

#
# Define attribute prob. This should eventually be written in C.
#
	# See if a recompute is necessary
	def _check_for_recompute(self):

		indices = where(self._self_timestamp_caches == self.timestamp)[0]

		for key in self._node_parent_keys:
			indices = where(self._parent_timestamp_caches[key][indices] == self.parents[key].timestamp)[0]

		if len(indices)==0:
			self._recompute = True
		else:
			self._recompute = False
			self._cache_index = indices[0]
		return

	def _get_prob(self):
		self._check_for_recompute()
		if self._recompute:

			#Recompute
			self._prob = self._prob_fun(self._value, **self.parent_values)
			
			#Cache
			push(self._self_timestamp_caches, self.timestamp)
			push(self._cached_prob, self._prob)
			for key in self._node_parent_keys:
				push(self._parent_timestamp_caches[key], self.parents[key].timestamp)

		else: self._prob = self._cached_prob[self._cache_index]					
		return self._prob
		
	prob = property(fget = _get_prob)

#	
# Call this when rejecting a jump.
#
	def revert(self):
		"""
		Call this when rejecting a jump.
		"""
		self._prob = self._cached_prob
		self._value = self._last_value
		self.timestamp -= 1



# Was SubSampler:
class SamplingMethod(object):
	"""
	This object knows how to make Parameters take single MCMC steps.
	It's sample() method will be called by Sampler at every MCMC iteration.

	Externally-accessible attributes:
				
		logicals:	The Logicals over which self has jurisdiction.
		
		parameters:	The Parameters over which self has jurisdiction which have isdata = False.
		
		data:		The Parameters over which self has jurisdiction which have isdata = True.
		
		nodes:		The Logicals and Parameters over which self has jurisdiction.
		
		children:	The combined children of all Nodes over which self has jurisdiction.
		
		likelihood:	The summed log-probability of self's children conditional on all of 
					self's Nodes' current values. These will be recomputed only as necessary.
					This descriptor should eventually be written in C.
		
	Externally accesible methods:
	
		sample():	A single MCMC step for all the Parameters over which self has jurisdiction.
					Must be overridden in subclasses.
					
		tune():		Tunes proposal distribution widths for all self's Parameters.
		
	To instantiate a SamplingMethod called S with jurisdiction over a sequence/set N of Nodes:
	
		S = SamplingMethod(N)
					
	See also OneAtATimeMetropolis and Sampler.
	"""

	def __init__(self, nodes):
	
		self.nodes = set(nodes)
		self.logicals = set()
		self.parameters = set()
		self.data = set()
		self.children = set()
		
		# File away the nodes
		for node in self.nodes:
			if isinstance(node,Logical):
				self.logicals.add(node)
			elif isinstance(node,Parameter):
				if node.isdata:
					self.data.add(node)
				else:
					self.parameters.add(node)
					
		# Find children, no need to find parents; each node takes care of those.
		for node in self.nodes:
			self.children |= node.children
			
		self._extend_children()
			
		self.children -= self.logicals
		self.children -= self.parameters
		self.children -= self.data

#
# Must be overridden in subclasses
#
	def step(self):
		pass
		
#
# Not implemented yet
#
	def tune(self):
		pass

#
# Find nearest random descendants
#		
	def _extend_children(self):
		need_recursion = False
		logical_children = set()
		for child in self.children:
			if isinstance(child,Logical):
				self.children |= child.children
				logical_children.add(child)
				need_recursion = True
		self.children -= logical_children
		if need_recursion:
			self._extend_children()
		return

#	
# Define attribute likelihood
#
	def _get_likelihood(self):
		return sum([child.prob for child in self.children])
		
	likelihood = property(fget = _get_likelihood)
			

# The default SamplingMethod, which Sampler uses to handle singleton parameters.		
class OneAtATimeMetropolis(SamplingMethod):
	"""
	The default SamplingMethod, which Sampler uses to handle singleton parameters.
	
	Applies the one-at-a-time Metropolis-Hastings algorithm to all Parameters over which
	self has jurisdiction.
	
	To instantiate a OneAtATimeMetropolis called M with jurisdiction over a sequence/set N of Nodes:
	
		M = OneAtATimeMetropolis(N)
	
	See also SamplingMethod and Sampler.
	"""
	def __init__(self,nodes):
		SamplingMethod.__init__(self,nodes)

#
# Do a one-at-a-time Metropolis-Hastings step for each of self's Parameters with isdata=False.
#		
	def step(self):
		pass
		
		
# Constructor of Sampler doesn't need any arguments, it searches the base namespace for
# any PyMC objects and files them away for sampling commands.
class Sampler(object):
	"""
	Sampler manages MCMC loops. It is initialized with no arguments:
	
	A = Sampler()
	
	At instantiation, Sampler browses the base namespace and collects all Nodes and SamplingMethods.
	If any Parameters with isdata = False are not already being managed by a SamplingMethod,
	Sampler assigns them to a OneAtATimeMetropolis object.
	
	Externally-accessible attributes:
	
		logicals:			All extant Logicals.
		
		parameters:			All extant Parameters with isdata = False.
		
		data:				All extant Parameters with isdata = True.
		
		nodes:				All extant Parameters and Logicals.
		
		sampling_methods:	All extant SamplingMethods.
		
	Externally-accessible methods:
	
		sample(iter,burn,thin):	At each MCMC iteration, calls each sampling_method's step() method.
								Tallies Parameters and Logicals as appropriate.
								
		All the plotting functions can probably go on the base namespace and take Parameters as
		arguments.
		
	See also SamplingMethod, OneAtATimeMetropolis, Node, Parameter, and Logical.
	"""

	def __init__(self):
	
		self.logicals = set()
		self.parameters = set()
		self.data = set()
		self.sampling_methods = set()
	
		# Search the base namespace and file PyMC objects
		import __main__
		snapshot = __main__.__dict__

		for item in snapshot.itervalues():

			if isinstance(item,Logical):  
				self.logicals.add(item)

			elif isinstance(item,Parameter):
				if item.isdata: 
					self.data.add(item)
				else:  self.parameters.add(item)
				
			elif isinstance(item,SamplingMethod): 
				self.sampling_methods.add(item)

		# Tell all nodes to get ready for sampling		
		self.nodes = self.logicals | self.parameters | self.data
		for node in self.nodes:
			node._extend_children()
			node._prepare()		
		
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
				self.sampling_methods.add(OneAtATimeMetropolis([parameter]))

#
# Run the MCMC loop!
#				
	def sample(self,iter,burn,thin):
		for i in range(iter):
			for sampling_method in self.sampling_methods:
				sampling_method.step()
				
		# Tally as appropriate.