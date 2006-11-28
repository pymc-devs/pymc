# Tries to keep the nice decorator syntax with value initialization from proposition3.py,
# but makes Parameter and Node into full-blown callable classes. It felt a little weird to
# give function objects so many attributes... but if the function-only versions turn out
# to be faster, that's a pretty compelling reason.
#
# Parameter and Node attribute self.value is a NodeValue descriptor:
# >>> object.value = 3: sets object._value to 3 and increments object.timestamp
# >>> object.value: returns object._value.
#
# TODO: Put in caches, get_likelihood function for Parameter.
#
# See test_prop_4.py

import copy
from numpy import *

def push(seq,new_value):
	length = len(seq)
	seq[1:length] = seq[0:length-1]
	seq[0] = new_value

def parameter(**kwargs):
	"""Decorator function instantiating the Parameter class."""

	def __instantiate_parameter(f):
		P = Parameter(prob=f,**kwargs)
		P.__doc__ = f.__doc__
		return P
		
	return __instantiate_parameter


def logical(**kwargs):
	"""Decorator function instantiating the Logical class."""

	def __instantiate_logical(f):
		L =Logical(eval_fun=f,**kwargs)
		L.__doc__ = f.__doc__
		return L
		
	return __instantiate_logical
	

def data(**kwargs):
	"""Decorator function instantiating the Parameter class, with flag 'data' set to True."""

	def __instantiate_data(f):
		D = Parameter(prob=f,isdata=True,**kwargs)
		D.__doc__ = f.__doc__
		return D
		
	return __instantiate_data

	
class Node(object):
	"""
	Node and Parameter inherit from this class.
	It handles the parent/children business and the timestamp
	initialization.
	"""
	def __init__(self, cache_depth = 2, **parents):

		self.parents = parents
		
		self._cache_depth = cache_depth
		self._parent_timestamp_caches = {}

		for key in self.parents.keys():
			if isinstance(self.parents[key],Node):
				self.parents[key].children.add(self)
			self._parent_timestamp_caches[key] = -1 * ones(self._cache_depth,dtype='int')

		self.children = set()		
		self._recompute = True
		self.timestamp = 0
		self._value = None


class Logical(Node):

	def __init__(self, eval_fun, **parents):

		Node.__init__(self,**parents)

		self.eval_fun = eval_fun
		self.__doc__ = eval_fun.__doc__
		self._value = None
		self._cached_value = []
		for i in range(self._cache_depth): self._cached_value.append(None)
		
	# Look through caches
	def _check_for_recompute(self):
		indices = range(self._cache_depth)
		for key in self.parents.keys():
			if isinstance(self.parents[key],Node):
				indices = where(self._parent_timestamp_caches[key][indices] == self.parents[key].timestamp)[0]

		if len(indices)==0:
			self._recompute = True				
		else:
			self._recompute = False
			self._cache_index = indices[0]
		return

	# Define the attribute value
	def get_value(self, *args, **kwargs):

		self._check_for_recompute()

		if self._recompute:

			#Recompute
			self._value = self.eval_fun(**self.parents)
			self.timestamp += 1

			# Cache
			push(self._cached_value, self._value)			
			for key in self.parents.keys():
				if isinstance(self.parents[key],Node):
					push(self._parent_timestamp_caches[key], self.parents[key].timestamp)

		else: self._value = self._cached_value[self._cache_index]
		return self._value
		
	value = property(fget=get_value)

class Parameter(Node):

	def __init__(self, prob, value=None, isdata=False, **parents):

		Node.__init__(self,**parents)

		self.isdata = isdata
		self.prob = prob
		self.__doc__ = prob.__doc__
		self._prob = None
		self._cached_prob = zeros(self._cache_depth,dtype='float')
		self._self_timestamp_caches = -1 * ones(self._cache_depth,dtype='int')

		if value:
			self._value = value

	
	# Define the attribute value
	def get_value(self, *args, **kwargs):
		return self._value
		
	def set_value(self, value):
		if self.isdata: print 'Warning, data value updated'
		self.timestamp += 1
		self._value = value
		
	value = property(fget=get_value, fset=set_value)


	# Look through caches
	def _check_for_recompute(self):

		indices = where(self._self_timestamp_caches == self.timestamp)[0]
		for key in self.parents.keys():
			if isinstance(self.parents[key],Node):
				indices = where(self._parent_timestamp_caches[key][indices] == self.parents[key].timestamp)[0]

		if len(indices)==0:
			self._recompute = True
		else:
			self._recompute = False
			self._cache_index = indices[0]
		return
	
	# Return probability
	def __call__(self):
		self._check_for_recompute()
		if self._recompute:

			#Recompute
			self._prob = self.prob(self.value, **self.parents)
			
			#Cache
			push(self._self_timestamp_caches, self.timestamp)
			push(self._cached_prob, self._prob)
			for key in self.parents.keys():
				if isinstance(self.parents[key],Node):
					push(self._parent_timestamp_caches[key], self.parents[key].timestamp)

		else: self._prob = self._cached_prob[self._cache_index]					
		return self._prob
	
	def revert(self):
		self._prob = self._cached_prob
		self._value = self._cached_value
		self.timestamp -= 1
	