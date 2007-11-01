from PyMC import *
from numpy import *

mu_A = array([0.,0.])
tau_A = eye(2)
A=Mvnormal('A',ones(2),mu=mu_A,tau=tau_A)

tau_B = eye(2) * 100.          
B=Mvnormal('B',ones(2),mu=A,tau=tau_B)

#S = JointMetropolis([A,B],epoch=100, memory=10, delay = 0)
if __name__=='__main__':
    S = Sampler(locals(), 'ram')
    S.sample(2e6,1e6,10, verbose=True)
    
