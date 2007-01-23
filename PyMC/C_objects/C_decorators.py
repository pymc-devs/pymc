import sys, inspect
from imp import load_dynamic

# Change path to point to the output of setup.py, 
#'PyMCObjects.so' or 'PyMCObjects.dll' probably,
# in directory C_objects/build/lib<system> usually.
path =  '/Users/anand/Documents/renearch/Programming/PyMC/svn/trial/C_objects/build/lib.macosx-10.4-fat-2.4/PYMCObjects.so'
load_dynamic('PyMCObjects',path)

from PyMCObjects import *

def _extract(__func__, kwds, keys):	
	"""
	Used by decorators parameter and node to inspect declarations
	"""
	kwds.update({'doc':__func__.__doc__, 'name':__func__.__name__})
	parents = {}

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
			kwds['eval'] =__func__

	for key in keys:
		if not kwds.has_key(key):
			kwds[key] = None			
			
	for key in ['logp', 'eval']:
		if key in keys:
			if kwds[key] is None:
				kwds[key] = __func__

	# Build parents dictionary by parsing the __func__tion's arguments.
	(args, varargs, varkw, defaults) = inspect.getargspec(__func__)
	try:
		parents.update(dict(zip(args[-len(defaults):], defaults)))

	# No parents at all		
	except TypeError: 
		pass
		
	if parents.has_key('value'):
		value = parents.pop('value')
	else:
		value = None
		
	return (value, parents)

def parameter(__func__=None, **kwds):
	"""
	Decorator function for instantiating parameters. Usages:
	
	Medium:
	
		@parameter
		def A(value = ., parent_name = .,  ...):
			return foo(value, parent_name, ...)
		
		@parameter(trace=trace_object)
		def A(value = ., parent_name = .,  ...):
			return foo(value, parent_name, ...)
			
	Long:

		@parameter
		def A(value = ., parent_name = .,  ...):
			
			def logp(value, parent_name, ...):
				return foo(value, parent_name, ...)
				
			def random(parent_name, ...):
				return bar(parent_name, ...)
				
	
		@parameter(trace=trace_object)
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
		value, parents = _extract(__func__, kwds, keys)
		if not kwds.has_key('isdata'):
			kwds['isdata'] = False
		if kwds['isdata'] == None:
			kwds['isdata'] = False
		kwds['children'] = set()
		return Parameter(value=value, parents=parents, **kwds)		
	keys = ['logp','random','trace','rseed']

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
		
	@node(trace = trace_object)
	def B(parent_name = ., ...)
		return baz(parent_name, ...)		
		
	where baz returns the node B's value conditional
	on its parents.
	"""

	def instantiate_n(__func__):
		junk, parents = _extract(__func__, kwds, keys)
		kwds['children'] = set()
		return Node(parents=parents, **kwds)		
	keys = ['eval','trace']
	
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
	return parameter(__func__, isdata=True, trace = None, **kwds)
