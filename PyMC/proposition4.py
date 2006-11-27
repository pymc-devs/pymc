# Tries to keep the nice decorator syntax with value initialization from proposition3.py,
# but makes Parameter and Node into full-blown callable classes. It felt a little weird to
# give function objects so many attributes... but if the function-only versions turn out
# to be faster, that's a pretty compelling reason.
#
# Parameter and Node attribute self.value is a NodeValue descriptor:
# >>> object.value = 3: sets object.__value to 3 and increments object.timestamp
# >>> object.value: returns object.__value.
#
# TODO: Put in caches, get_likelihood function for Parameter.
#
# See test_prop_4.py

import copy

def parameter(**kwargs):
	"""Decorator function instantiating the Parameter class."""

	def __instantiate_parameter(f):
		P = Parameter(prob=f,**kwargs)
		P.__doc__ = f.__doc__
		return P
		
	return __instantiate_parameter

def node(**kwargs):
	"""Decorator function instantiating the Node class."""

	def __instantiate_node(f):
		N = Node(eval_fun=f,**kwargs)
		N.__doc__ = f.__doc__
		return N
		
	return __instantiate_node
	
def data(**kwargs):
	"""Decorator function instantiating the Parameter class, with flag 'data' set to True."""

	def __instantiate_data(f):
		D = Parameter(prob=f,isdata=True,**kwargs)
		D.__doc__ = f.__doc__
		return D
		
	return __instantiate_data
	


class Node(object):

	def __init__(self, eval_fun, **parents):

		self.eval_fun = eval_fun
		self.parent = parents
		self.__doc__ = eval_fun.__doc__		
		self.recompute = True

		self.__value = None
		self.timestamp = 0		

	# Define the attribute value
	def get_value(self, *args, **kwargs):
		self.__check_for_recompute()
		if self.recompute:
			self.__value = self.eval_fun(**self.parent)
			self.timestamp += 1
		return self.__value
		
	value = property(fget=get_value)

	def __check_for_recompute(self):
		# Look through caches
		pass

		
class Parameter(object):

	def __init__(self, prob, value=None, isdata=False, **parents):

		self.isdata = isdata
		self.prob = prob
		self.parent = parents
		self.__doc__ = prob.__doc__		
		self.recompute = True

		self.__prob = None
		self.__value = None
		self.timestamp = 0		

		if value:
			self.__value = value

	# Define the attribute value
	def get_value(self, *args, **kwargs):
		return self.__value
		
	def set_value(self, value):
		if self.isdata: print 'Warning, data value updated'
		self.timestamp += 1
		self.__value = value
		
	value = property(fget=get_value, fset=set_value)

	def __check_for_recompute(self):
		# Look through caches
		pass
	
	def __call__(self):
		# Return probability
		self.__check_for_recompute()
		if self.recompute:
			self.__prob = self.prob(self.value, **self.parent)
		return self.__prob
	
	def revert(self):
		self.__prob = self.__cached_prob
		self.__value = self.__cached_value
		self.timestamp -= 1
	