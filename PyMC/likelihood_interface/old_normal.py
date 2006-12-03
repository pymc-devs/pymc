from numpy import *
from PyMC import flib
from PyMC.flib import normal as fnormal

def old_normal(x, mu, tau):
	"""Normal log-likelihood"""
	
	if not shape(mu) == shape(tau): raise ParameterError, 'Parameters must have same dimensions in normal_like()'
	
	if ndim(mu) > 1:
		
		return sum(old_normal(y, m, t) for y, m, t in zip(x, mu, tau))
	
	else:
				
		# Ensure array type
		x = atleast_1d(x)
		mu = resize(mu, shape(x))
		tau = resize(tau, shape(x))
				
		return sum(fnormal(y, m, t) for y, m, t in zip(x, mu, tau))
