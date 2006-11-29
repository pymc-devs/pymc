from test_decorator import normal_like, uniform_like, node_to_NDarray
from proposition4 import *

@parameter(value=20., mu=10., tau=20.)
def A(value, mu, tau):
	return normal_like(value, mu, tau)


class Normal(Parameter):
	def __init__(self,value,mu,tau):
		Parameter.__init__(self,normal_like,value=value,mu=mu,tau=tau)

B=Normal(value=3., mu=A, tau=3.)
B.__doc__='Bla bla bla'

@data(value=20., mu=A, tau=23.)
def C(value, mu, tau):
	return normal_like(value, mu, tau)

@logical(a=A,b=2.)
def D(a,b):
	return node_to_NDarray(a) ** node_to_NDarray(b)

# Take value and prob out for a spin
print "A's old value: ", A.value, " A's old timestamp:", A.timestamp, " B's old probability:", B.prob
A.value = 16.
print "A's new value: ", A.value, " A's new timestamp:", A.timestamp, " B's new probability:", B.prob
print ""

# Instantiate a SamplingMethod with one parameter and one logical
A_D_sampling_method = OneAtATimeMetropolis([A,D])
# Introspect it
print "Contents of A_D_sampling_method:"
print A_D_sampling_method.__dict__
print ""

# Instantiate a Sampler. Doesn't need any arguments, it searches the base namespace
# and scoops up all Parameters, Data, and SamplingMethods.
#
# If any Parameters aren't already covered by SamplingMethods, it makes a new
# anonymous SamplingMethod for each.
my_sampler = Sampler()
print "Contents of my_sampler:"
print my_sampler.__dict__