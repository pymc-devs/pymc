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

"""
Define the data and parameters
"""

@add_draw_fun(lambda length: randint(length))
@parameter(init_val=50, length = 110)
def switchpoint(value, length):
	"""Change time for rate parameter."""
	if value >= 0 and value <= 110: return 0.
	else: return -Inf


@add_draw_fun(lambda rate: rexpo(rate))
@parameter(init_val=1., rate = 1.)
def early_mean(value, rate):
	"""Rate parameter of poisson distribution."""
	if value>0: return -rate * value
	else: return -Inf 


@add_draw_fun(lambda rate: rexpo(rate))
@parameter(init_val=1., rate = 1.)
def late_mean(value, rate):
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
	early_mean = early_mean, 
	late_mean = late_mean, 
	switchpoint = switchpoint, 
	caching = True)
def disasters(value,early_mean,late_mean,switchpoint):
	"""Annual occurences of coal mining disasters."""
	return poisson_like(value[:switchpoint],early_mean) + poisson_like(value[switchpoint+1:],late_mean)
	



"""
Make a special SamplingMethod for switchpoint that will keep it on integer values,
and add it to M.
"""
S = OneAtATimeMetropolis(parameter=switchpoint, dist = 'RoundedNormal')


