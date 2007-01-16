"""
A model for the disasters data with no changepoint:

global_rate ~ Exp(3.)
disasters[t] ~ Po(global_rate)
"""

from proposition5 import *
from numpy.random import exponential as rexpo

def poisson_like(value, rate):
	return sum(log(rate) * value - rate)



# Define the data and parameters

@parameter(init_val=1.)
def global_rate(rate=3.):
	"""Rate parameter of poisson distribution."""
	
	def logp_fun(value, rate):
		if value>0: return -rate * value
		else: return -Inf 
		
	def random(rate):
		return rexpo(rate)
	

@data(init_val = array([4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
						3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
						2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
						1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
						0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
						3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
						0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1]), caching = False)
def disasters(rate = global_rate):
	"""Annual occurences of coal mining disasters."""
	logp_fun = poisson_like


