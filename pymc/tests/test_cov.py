from numpy.testing import *
from pymc.gp import *
from pymc.gp.cov_funs import matern
from numpy import *
from copy import copy
from test_mean import x, y

C = Covariance(eval_fun = matern.euclidean, diff_degree = 1.4, amp = .4, scale = 1.)
D = copy(C)

class test_cov(NumpyTestCase):
    def check(self):

        assert(C(x).shape == x.shape)
        assert(C(x,x).shape == (len(x), len(x)))
             
        assert(D(y).shape == (y.shape[0],))
        assert(D(y,y).shape == (y.shape[0], y.shape[0]))