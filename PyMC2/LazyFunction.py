from numpy import array, zeros, ones, arange
from Container import Container
from PyMC2 import Parameter, Node, PyMCBase

class LazyFunction(object):

	def __init__(self, fun, arguments, cache_depth, owner):
		self.arguments = arguments
		self.cache_depth = cache_depth
		self.owner = owner
		
		# Caches of recent computations of self's value
		self.cached_value = []
		for i in range(self.cache_depth): 
			self.cached_value.append(None)
			
		# Some aranges ahead of time for faster looping
		self.cache_range = arange(self.cache_depth)
		self.upper_cache_range = arange(1,self.cache_depth)					
		
		self.file_arguments()
		self.fun = fun
		self.refresh_argument_values()

	# See if a recompute is necessary.
	def check_argument_caches(self):
		for i in self.cache_range:
			mismatch=False

			for j in self.node_range:
				if not self.node_argument_counters[j] == self.node_argument_counter_caches[i][j]:
					mismatch=True
					break

			if not mismatch:
				for j in self.param_range:
					if not self.param_argument_counters[j] == self.param_argument_counter_caches[i][j]:
						mismatch=True
						break

			if not mismatch:
				return i

		# If control reaches here, a mismatch occurred.
		for j in self.node_range:
			for i in self.upper_cache_range:
				self.node_argument_counter_caches[i][j] = self.node_argument_counter_caches[i][j-1]
			self.node_argument_counter_caches[0][j] = self.node_argument_counters[j]

		for j in self.param_range:
			for i in self.upper_cache_range:
				self.param_argument_counter_caches[i][j] = self.param_argument_counter_caches[i][j-1]
			self.param_argument_counter_caches[0][j] = self.param_argument_counters[j]

		return -1;

	def file_arguments(self):

		# A dictionary of those arguments that are PyMC objects or containers.
		self.pymc_object_arguments = {}

		# The argument_values dictionary will get passed to the logp/
		# eval function.		
		self.argument_values = {}

		self.N_node_arguments = 0
		self.N_param_arguments = 0

		# Make sure no arguments are None, and count up the arguments
		# that are parameters and nodes, including those enclosed
		# in PyMC object containers.
		for key in self.arguments.iterkeys():
			assert self.arguments[key] is not None, self.__name__ + ': Error, argument ' + key + ' is None.'
			if isinstance(self.arguments[key], Parameter):
				self.N_param_arguments += 1
			elif isinstance(self.arguments[key], Node):
				self.N_node_arguments += 1
			elif isinstance(self.arguments[key], Container):
				self.N_node_arguments += len(self.arguments[key].nodes)
				self.N_param_arguments += len(self.arguments[key].parameters)

		# More upfront aranges for faster looping.
		self.node_range = arange(self.N_node_arguments)
		self.param_range = arange(self.N_param_arguments)				

		# Initialize array of references to arguments' counters.
		self.node_argument_counters = zeros(self.N_node_arguments,dtype=object)
		self.param_argument_counters = zeros(self.N_param_arguments,dtype=object)

		# Initialize argument counter cache arrays
		self.node_argument_counter_caches = -1*ones((self.cache_depth, self.N_node_arguments), dtype=int)
		self.param_argument_counter_caches = -1*ones((self.cache_depth, self.N_param_arguments), dtype=int)


		# Sync up arguments and children, figure out which arguments are PyMC
		# objects and which are just objects.
		#
		# ultimate_index indexes the arguments, including those enclosed in
		# containers.
		ultimate_index=0
		for key in self.arguments.iterkeys():

			if isinstance(self.arguments[key],PyMCBase):

				# Add self to this argument's children set
				if self.arguments[key] is not self.owner:
					self.arguments[key].children.add(self.owner)

				# Remember that this argument is a PyMCBase.
				# This speeds the _refresh_argument_values method.
				self.pymc_object_arguments[key] = self.arguments[key]

				# Record references to the argument's counter array scalars
				if isinstance(self.arguments[key],Node):
					self.node_argument_counters[ultimate_index] = self.arguments[key].counter

				if isinstance(self.arguments[key],Parameter):
					self.param_argument_counters[ultimate_index] = self.arguments[key].counter

				ultimate_index += 1					

			# Unpack parameters and nodes from containers 
			# for counter=checking purposes.
			elif isinstance(self.arguments[key], Container):			

				# Record references to the argument's parameters' 
				# and nodes' counter array scalars
				for node in self.arguments[key].nodes:
					self.node_argument_counters[ultimate_index] = node.counter
					ultimate_index += 1

				for param in self.arguments[key].paramters:
					self.param_argument_counters[ultimate_index] = param.counter
					ultimate_index += 1					

			# If the argument isn't a PyMC object or PyMC object container,
			# record a reference to its value.
			else:
				self.argument_values[key] = self.arguments[key]

	# Extract the values of arguments that are PyMC objects or containers.
	# Don't worry about unpacking the containers, see their value attribute.
	def refresh_argument_values(self):
		for item in self.pymc_object_arguments.iteritems():
			self.argument_values[item[0]] = item[1].value
			
	def get(self):
		# Recomp is overzealous so far.
		self.refresh_argument_values()
		recomp = self.check_argument_caches()

		if recomp < 0:

			#Recompute
			value = self.fun(**self.argument_values)

			# Cache and increment counter
			del self.cached_value[self.cache_depth-1]
			self.cached_value.insert(0,value)

		else: value = self.cached_value[recomp]

		return recomp, value