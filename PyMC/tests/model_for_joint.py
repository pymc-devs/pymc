from proposition5 import *
from numpy import array, eye
from flib import mvnorm

# For some reason the decorator isn't working with no parents...

mu_A = array([0.,0.])
tau_A = eye(2)
@parameter(init_val = ones(2,dtype=float), mu=mu_A, tau = tau_A)
def A():
	def logp_fun(value,mu,tau):
		return mvnorm(value,mu,tau)


tau_B = eye(2) * 100.		   
@parameter(init_val = ones(2,dtype=float), mu = A, tau = tau_B)
def B():
	def logp_fun(value,mu,tau):
		return mvnorm(value,mu,tau) 

S = Joint([A,B],epoch=1000, memory=10, interval = 10, delay = 0)