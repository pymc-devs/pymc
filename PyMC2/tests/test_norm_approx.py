from PyMC2 import NormalApproximation
from PyMC2.examples import model_2
from PyMC2.examples import model_for_joint
# N = NormalApproximation(model_2, method = 'fmin_l_bfgs_b')
N = NormalApproximation(model_for_joint, eps=.0000001, method = 'fmin_l_bfgs_b')