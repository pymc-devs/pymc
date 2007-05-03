""" Test database backends """

from numpy.testing import *
from PyMC2 import Sampler, database
from PyMC2.examples import DisasterModel
import os
try:
    os.system('rm DisasterModel.pickle')
    os.system('rm DisasterModel.sqlite')
    os.system('rm DisasterModel.sqlite-journal')
    os.system('rm DisasterModel.hdf5')
except:
    pass

class test_no_trace(NumpyTestCase):
    def check(self):
        M = Sampler(DisasterModel, db='no_trace')
        M.sample(1000,500,2, verbose=False)
        try:
            assert_array_equal(M.e.trace().shape, (0,))
        except AttributeError:
            pass
        
class test_ram(NumpyTestCase):
    def check(self):
        M = Sampler(DisasterModel, db='ram')
        M.sample(500,100,2, verbose=False)
        assert_array_equal(M.e.trace().shape, (250,))
        M.sample(200,0,1)
        assert_array_equal(M.e.trace().shape, (200,))
        assert_array_equal(M.e.trace(chain=None).shape, (450,))
        
class test_pickle(NumpyTestCase):
    def check(self):
        M = Sampler(DisasterModel, db='pickle')
        M.sample(500,100,2, verbose=False)
        assert_array_equal(M.e.trace().shape, (250,))
        M.db.close()
        
    def check_load(self):
        db = database.pickle.load('DisasterModel.pickle')
        S = Sampler(DisasterModel, db)
        S.sample(200,0,1)
        assert_equal(len(S.e.trace._trace),2)
        assert_array_equal(S.e.trace(chain=None).shape, (450,))
        S.db.close()
##class test_txt(NumpyTestCase):
##    def check(self):
##        M = Sampler(DisasterModel, db='txt')
##        M.sample(300,100,2, verbose=False)
##        assert_equal(M.e.trace().shape, (150,))
##
##class test_mysql(NumpyTestCase):
##    def check(self):
##        M = Sampler(DisasterModel, db='mysql')
##        M.sample(300,100,2, verbose=False)
##    

class test_sqlite(NumpyTestCase):
    def check(self):
        M = Sampler(DisasterModel, db='sqlite')
        M.sample(500,100,2, verbose=False)
        assert_array_equal(M.e.trace().shape, (250,))
        M.db.close()
        
    def check_load(self):
        db = database.sqlite.load('DisasterModel.sqlite')
        S = Sampler(DisasterModel, db)
        S.sample(200,0,1)
        assert_array_equal(S.e.trace(chain=None).shape, (450,))
        S.db.close()
        
##class test_hdf5(NumpyTestCase):
##    def check(self):
##        M = Sampler(DisasterModel, db='hdf5')
##        M.sample(300,100,2, verbose=False)
    
class test_hdf5_tables(NumpyTestCase):
    def check(self):
        S = Sampler(DisasterModel, db='hdf5_tables')
        S.sample(500,100,2, verbose=False)
        assert_array_equal(S.e.trace().shape, (250,))
        S.db.close()
        
    def check_load(self):
        db = database.hdf5_tables.load('DisasterModel.hdf5', 'a')
        S = Sampler(DisasterModel, db)
        S.sample(200,0,1)
        assert_array_equal(S.e.trace(chain=None).shape, (450,))
        S.db.close()
        
    def check_compression(self):
        db = database.hdf5_tables.Database('DisasterModelCompressed.hdf5', complevel=5)
        S = Sampler(DisasterModel,db)
        S.sample(450,100,1, verbose=False)
        assert_array_equal(S.e.trace().shape, (450,))
        S.db.close()
        
if __name__ == '__main__':
    NumpyTest().run()
