from PyMC import stoch, data, JointMetropolis, Sampler
from numpy import array, eye, ones
from PyMC.distributions import mvnormal_like

mu_A = array([0.,0.])
tau_A = eye(2)
@stoch
def A(value = ones(2,dtype=float), mu=mu_A, tau = tau_A):
    return mvnormal_like(value,mu,tau)

tau_B = eye(2) * 100.          
@stoch
def B(value = ones(2,dtype=float), mu = A, tau = tau_B):
    return mvnormal_like(value,mu,tau)

#S = JointMetropolis([A,B],epoch=100, memory=10, delay = 0)
if __name__=='__main__':
    S = Sampler(locals(), 'ram')
    S.sample(2e6,1e6,10, verbose=True)
    
