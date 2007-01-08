"""
A model for the disasters data with an linearly varying mean:

rate_of_mean ~ N(0,1)
amp_of_mean ~ Exp(3)
disasters[t] ~ Po(intercept_of_mean + slope_of_mean * t)
"""

from proposition5 import *
from numpy.random import exponential as rexpo
from numpy.random import normal as rnormal
		
def poisson_like(val,rate):
	if (rate>0).all():
		return sum(log(rate) * val - rate)
	else:
		return -Inf
	
def normal_like(val,mu,tau):
	return -.5 * (val-mu) ** 2 * tau


# Define data and parameters

@parameter(init_val=-.1)
def slope_of_mean(tau=.1):
	"""Rate constant of rate parameter of poisson distribution"""
	def logp_fun(value, tau):
		val= normal_like(value,0,tau)
		return val
		
	def random(tau):
		val= rnormal(scale=sqrt(1./tau))
		return val

@parameter(init_val=1.)
def intercept_of_mean(rate = 4.):
	"""Amplitude of rate parameter of poisson distribution."""

	def logp_fun(value, rate):
		if value>0: return -rate * value
		else: return -Inf  
		
	def random(rate):
		val= rexpo(rate)
		return val
	
@data(init_val = array([4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
						3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
						2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
						1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
						0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
						3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
						0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1]), caching = True)
						
def disasters(slope_of_mean = slope_of_mean, intercept_of_mean = intercept_of_mean):
	"""Annual occurences of coal mining disasters."""
	def logp_fun(value, slope_of_mean, intercept_of_mean):
		val = intercept_of_mean + slope_of_mean * arange(111)
		return poisson_like(value,val)

