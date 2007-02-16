"""
Abstract base classes to ease objects' task in recognizing one another.
"""

class PyMCBase(object):
	"""
	All inherit from this class.
	"""
	pass
	
	
class PurePyMCBase(PyMCBase):
	"""
	The pure Python Parameter and Node inherit from this class.
	
	ALL user-defined classes that behave like PyMC objects
	should inherit from this class.
	"""
	pass
	
	
class ParameterBase(PyMCBase):
	"""
	The C _and_ pure Parameters inherit from this class.
	"""
	pass
	
	
class NodeBase(PyMCBase):
	"""
	The C _and_ pure Nodes inherit from this class.
	"""
	pass
	
	
class ContainerBase(object):
	"""
	PyMCObjectContainer inherits from this class.
	"""
	pass
	
def new_set():
	"""
	This is really annoying, but I couldn't figure out how to make a new set in C.
	"""
	return set()
	
def set_iter(A):
	return A.__iter__()
	
def set_length(A):
	return len(A)