from PyMC2.flib import rnormal
from numpy import zeros, ones
from numpy.random import normal

mu = zeros(100)
sig = ones(100)
mu = mu + .001


for i in xrange(100000):
	#rnormal(mu,sig,100)
	normal(mu, sig)