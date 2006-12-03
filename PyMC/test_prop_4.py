from test_decorator import normal_like, uniform_like, node_to_NDarray
from numpy import *
from proposition4 import *

@parameter(init_val=20., q=10., p=20.)
def A(value, q, p):
	"""
	A ~ N(q ** 2, p ** 127).
	"""
	return normal_like(value, q ** 2, p / 127.)


@parameter(init_val = 3., mu = array([[2,3],[4,16]], dtype='float'), tau = array([2,3,4,16], dtype='float'))
def B(value, mu, tau):
	"""
	B ~ N(mu[0,1], tau[2]).
	"""
	return normal_like(value, mu[0,1], tau[2])

# C ~ N(A,B), value known
@data(init_val=20., mu=A, tau=B)
def C(value, mu, tau):
	"""
	C ~ N(mu, tau), C's value is known (C is data).
	"""
	return normal_like(value, mu, tau)

@logical(a=A,b=2.)
def D(a,b):
	"""
	D = a ** b.
	"""
	return a ** b

# Take value and prob out for a spin
print "A's old value: ", A.value, " A's old timestamp:", A.timestamp 
print " C's old probability:", C.prob
A.value = 16.

print ""
print "A's new value: ", A.value, " A's new timestamp:", A.timestamp 
print " C's new probability:", C.prob
print ""

# Instantiate a SamplingMethod with one parameter and one logical
A_D_sampling_method = OneAtATimeMetropolis([A,D])
# Introspect it
print "Contents of A_D_sampling_method:"
print A_D_sampling_method.__dict__
print ""

# Instantiate a Sampler.
my_sampler = Sampler()
print "Contents of my_sampler:"
print my_sampler.__dict__
