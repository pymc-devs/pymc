"""
A model for the disasters data with an linearly varying mean:

rate_of_mean ~ N(0,1)
amp_of_mean ~ Exp(3)
disasters[t] ~ Po(intercept_of_mean + slope_of_mean * t)
"""

from proposition5 import *
from numpy.random import exponential as rexpo
from numpy.random import normal as rnormal
from flib import poisson
	
def normal_like(val,mu,tau):
	return -.5 * (val-mu) ** 2 * tau


disasters_array = 	array([ 4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
							3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
							2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
							1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
							0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
							3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
							0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])

# Define data and parameters

@parameter
def params_of_mean(value=array([-.1, 1.]), tau=.1, rate = 4.):
	"""
	Intercept and slope of rate parameter of poisson distribution
	Rate parameter must be positive for t in [0,T]
	
	p(intercept, slope|tau,rate) = 
	N(slope|0,tau) Exp(intercept|rate) 1(intercept>0) 1(intercept + slope * T>0)
	"""

	def logp(value, tau, rate):
		if value[1]>0 and value[1] + value[0] * 110 > 0:
			return normal_like(value[0],0,tau) - rate * value[1]
		else:
			return -inf
		
	def random(tau, rate):
		val = zeros(2)
		val[0] = rnormal(scale=sqrt(1./tau))
		val[1] = rexpo(rate)
		while val[1]<0 or val[1] + val[0] * 110 <= 0:
			val[0] = rnormal(scale=sqrt(1./tau))
			val[1] = rexpo(rate)
		return val
	
@data(caching = True)
						
def disasters(value = disasters_array, params_of_mean = params_of_mean):
	"""Annual occurences of coal mining disasters."""
	val = params_of_mean[1] + params_of_mean[0] * arange(111)
	return poisson(value,val)

