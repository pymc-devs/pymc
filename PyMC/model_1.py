"""
A model for the disasters data with a changepoint

changepoint ~ U(0,111)
early_mean ~ Exp(1.)
late_mean ~ Exp(1.)
disasters[t] ~ Po(early_mean if t <= switchpoint, late_mean otherwise)


All likelihoods commented out,
- Timestamp: 16s
- No timestamp: 8s

With likelihoods,
- Timestamp: 19.7s
- No timestamp: 17.25 s
"""

from proposition5 import *
from numpy.random import exponential as rexpo
		
def poisson_like(val,rate):
	return sum(log(rate) * val - rate)



# Define data and parameters

@parameter(init_val=50)
def switchpoint(length=110):
	"""Change time for rate parameter."""

	def logp_fun(value, length):
		if value >= 0 and value <= length: return 0.
		else: return -Inf
		
	def random(length):
		return randint(length)


@parameter(init_val=1.)
def early_mean(rate=1.):
	"""Rate parameter of poisson distribution."""

	def logp_fun(value, rate):
		if value>0: return -rate * value
		else: return -Inf 
		
	def random(rate):
		return rexpo(rate)


@parameter(init_val=1.)
def late_mean(rate = 1.):
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
						0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1]), caching = True)
def disasters(	early_mean = early_mean, 
				late_mean = late_mean, 
				switchpoint = switchpoint, 
):
	"""Annual occurences of coal mining disasters."""
	
	def logp_fun(value, early_mean, late_mean, switchpoint):
		return poisson_like(value[:switchpoint],early_mean) + poisson_like(value[switchpoint+1:],late_mean)
	



"""
Make a special SamplingMethod for switchpoint that will keep it on integer values,
and add it to M.
"""
S = OneAtATimeMetropolis(parameter=switchpoint, dist = 'RoundedNormal')


