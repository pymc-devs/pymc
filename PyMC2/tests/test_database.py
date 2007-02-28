""" Test database backends """

from numpy.testing import *
from PyMC2 import Model, database
from PyMC2.examples import DisasterModel


class test_no_trace(NumpyTestCase):
    def check(self):
        M = Model(DisasterModel, db='no_trace')
        M.sample(1000,500,2, verbose=False)
        try:
            assert_equal(M.e.trace().shape, (0,))
        except AttributeError:
            pass
        
class test_ram(NumpyTestCase):
    def check(self):
        M = Model(DisasterModel, db='ram')
        M.sample(300,100,2, verbose=False)
        assert_equal(M.e.trace().shape, (150,))
    
class test_txt(NumpyTestCase):
    def check(self):
        M = Model(DisasterModel, db='txt')
        M.sample(300,100,2, verbose=False)
        assert_equal(M.e.trace().shape, (150,))

class test_mysql(NumpyTestCase):
    def check(self):
        M = Model(DisasterModel, db='mysql')
        M.sample(300,100,2, verbose=False)
    
class test_sqlite(NumpyTestCase):
    def check(self):
        M = Model(DisasterModel, db='sqlite')
        M.sample(300,100,2, verbose=False)
    
class test_hdf5(NumpyTestCase):
    def check(self):
        M = Model(DisasterModel, db='hdf5')
        M.sample(300,100,2, verbose=False)
    
if __name__ == '__main__':
    NumpyTest().run()
