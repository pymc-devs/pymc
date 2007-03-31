from PyMC2 import parameter, data, JointMetropolis
from numpy import array, eye, ones
from PyMC2.distributions import multivariate_normal_like

mu_A = array([0.,0.])
tau_A = eye(2)
@parameter
def A(value = ones(2,dtype=float), mu=mu_A, tau = tau_A):
    return multivariate_normal_like(value,mu,tau)

tau_B = eye(2) * 100.          
@parameter
def B(value = ones(2,dtype=float), mu = A, tau = tau_B):
    return multivariate_normal_like(value,mu,tau)

S = JointMetropolis([A,B],epoch=100, memory=10, delay = 0)
