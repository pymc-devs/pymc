from numpy import *
from copy import deepcopy

class dummy_class(object):
	def __init__(self):
		self._value = array([1.,2.])
		
	def _get_value(self):
		return self._value
		
	def _set_value(self, value):
		self.last_value = deepcopy(self._value)
		self._value = value
		print ' Value changed from ',self.last_value,' to ',self._value
		return self._value
		
	value = property(fget=_get_value,fset=_set_value)

A = dummy_class()


print 'Correct: '
A.value = array([3., 3.])

print 'Wrong: '
A.value += array([1., 1.])