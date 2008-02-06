from pymc import *
from numpy import *

mu_A = array([0.,0.])
tau_A = eye(2)
A=MvNormal('A',mu=mu_A,tau=tau_A)

tau_B = eye(2) * 100.          
B=MvNormal('B',mu=A,tau=tau_B)
