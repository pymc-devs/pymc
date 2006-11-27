from test_decorator import normal_like, uniform_like, node_to_NDarray
from proposition4 import *

@parameter(value=20., mu=10., tau=20.)
def A(value, mu, tau):
	return normal_like(value, mu, tau)


class Normal(Parameter):
	def __init__(self,value,mu,tau):
		Parameter.__init__(self,normal_like,value=value,mu=mu,tau=tau)
B=Normal(value=3., mu=A, tau=3.)


@data(value=20., mu=A, tau=23.)
def C(value, mu, tau):
	return normal_like(value, mu, tau)

@logical(a=A,b=2.)
def D(a,b):
	return node_to_NDarray(a) ** node_to_NDarray(b)
