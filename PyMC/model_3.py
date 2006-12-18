"""
A model for the disasters data with an exponentially changing mean:

rate_of_mean ~ N(0,1)
amp_of_mean ~ Exp(3)
disasters[t] ~ Po(amp_of_mean * exp(rate_of_mean * t))
"""

from proposition5 import *
from numpy.random import exponential as rexpo
from numpy.random import normal as rnormal
		
def poisson_like(val,rate):
	return sum(log(rate) * val - rate)
	
def normal_like(val,mu,tau):
	return -.5 * (val-mu) ** 2 * tau

"""
Define the data and parameters
"""

@add_draw_fun(lambda tau: rnormal(sqrt(1./tau)))
@parameter(init_val=-.1, tau = 1.)
def rate_of_mean(value, tau):
	"""Rate constant of rate parameter of poisson distribution"""
	return normal_like(value,0,tau)

@add_draw_fun(lambda rate: rexpo(rate))
@parameter(init_val=1., rate = 1.)
def amp_of_mean(value, rate):
	"""Amplitude of rate parameter of poisson distribution."""
	if value>0: return -rate * value
	else: return -Inf  
	
@data(init_val = array([4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
						3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
						2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
						1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
						0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
						3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
						0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1]),
	rate_of_mean = rate_of_mean,
	amp_of_mean = amp_of_mean,
	caching = False)
def disasters(value,rate_of_mean, amp_of_mean):
	"""Annual occurences of coal mining disasters."""
	return poisson_like(value,amp_of_mean * exp(rate_of_mean * arange(111)))



