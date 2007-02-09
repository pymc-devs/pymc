""" Test database backends """

from numpy.testing import *
from PyMC2 import Model, database
from PyMC2.examples import DisasterModel


class test_no_trace(NumpyTestCase):
    M = Model(DisasterModel, dbase='no_trace')
    M.sample(1000,500,2)
    
class test_ram(NumpyTestCase):
   M = Model(DisasterModel, dbase='ram')
   M.sample(1000,500,2)
    
class test_txt(NumpyTestCase):
    M = Model(DisasterModel, dbase='txt')
    M.sample(1000,500,2)

class test_mysql(NumpyTestCase):
    M = Model(DisasterModel, dbase='mysql')
    M.sample(1000,500,2)
    
class test_mysql(NumpyTestCase):
    M = Model(DisasterModel, dbase='sqlite')
    M.sample(1000,500,2)
    
class test_hdf5(NumpyTestCase):
    M = Model(DisasterModel, dbase='hdf5')
    M.sample(1000,500,2)
    
if __name__ == '__main__':
    NumpyTest().run()
