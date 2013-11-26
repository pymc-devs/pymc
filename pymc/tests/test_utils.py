from pymc import *
from numpy import *
from numpy.testing import *
import nose
import sys
from pymc import utils

from pymc import six
xrange = six.moves.xrange


class test_logp_of_set(TestCase):
    A = Normal('A', 0, 1)
    B = Gamma('B', 1, 1)
    C = Lambda('C', lambda b=B: sqrt(b))
    D = Gamma('D', C, 1)

    @stochastic
    def E(x=1, value=3):
        if value != 3:
            raise RuntimeError
        else:
            return 0.
    E.value = 2

    def test_logp(self):
        self.B.rand()
        lp1 = utils.logp_of_set(set([self.A, self.B, self.D]))
        assert_almost_equal(lp1, self.A.logp + self.B.logp + self.D.logp, 10)

    def test_ZeroProb(self):
        self.B.value = -1
        for i in xrange(1000):
            try:
                utils.logp_of_set(set([self.A, self.B, self.D, self.E]))
            except:
                cls, inst, tb = sys.exc_info()
                assert(cls is ZeroProbability)

    def test_other_err(self):
        self.B.rand()
        for i in xrange(1000):
            try:
                utils.logp_of_set(set([self.A, self.B, self.D, self.E]))
            except:
                cls, inst, tb = sys.exc_info()
                assert(cls is RuntimeError)


class test_normcdf_input_shape(TestCase):

    def test_normcdf_1d_input(self):
        x = arange(8.)
        utils.normcdf(x)

    def test_normcdf_log_1d_input(self):
        x = arange(8.)
        utils.normcdf(x, log=True)

    def test_normcdf_2d_input(self):
        x = arange(8.).reshape(2, 4)
        utils.normcdf(x)

    def test_normcdf_log_2d_input(self):
        x = arange(8.).reshape(2, 4)
        utils.normcdf(x, log=True)

    def test_normcdf_3d_input(self):
        x = arange(8.).reshape(2, 2, 2)
        utils.normcdf(x)

    def test_normcdf_log_3d_input(self):
        x = arange(8.).reshape(2, 2, 2)
        utils.normcdf(x, log=True)


if __name__ == '__main__':
    C = nose.config.Config(verbosity=1)
    nose.runmodule(config=C)
