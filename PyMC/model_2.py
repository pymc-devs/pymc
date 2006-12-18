"""
A model for the disasters data with no changepoint:

global_rate ~ Exp(3.)
disasters[t] ~ Po(global_rate)
"""

from proposition5 import *
from numpy.random import exponential as rexpo
		
def poisson_like(val,rate):
	return sum(log(rate) * val - rate)

"""
Define the data and parameters
"""

@add_draw_fun(lambda rate: rexpo(rate))
@parameter(init_val=1., rate = 3.)
def global_rate(value, rate):
	"""Rate parameter of poisson distribution."""
	if value>0: return -rate * value
	else: return -Inf 
	

@data(init_val = array([4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
						3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
						2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
						1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
						0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
						3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
						0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1]),
	rate = global_rate,  
	caching = False)
def disasters(value,rate):
	"""Annual occurences of coal mining disasters."""
	return poisson_like(value,rate)



