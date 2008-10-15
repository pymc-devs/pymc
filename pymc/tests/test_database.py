""" Test database backends """

from numpy.testing import TestCase, assert_array_equal, assert_equal
from pymc.examples import DisasterModel
import os,sys, pdb
import numpy as np
from pymc import MCMC
import pymc.database
import nose

class test_backend_attribution(TestCase):
    def test_raise(self):
        self.assertRaises(AttributeError, MCMC, DisasterModel, 'heysugar')
    def test_import(self):
        self.assertRaises(ImportError, MCMC, DisasterModel, '__test_import__')


class test_no_trace(TestCase):
    def test(self):
        M = MCMC(DisasterModel, db='no_trace')
        M.sample(1000,500,2)
        try:
            assert_array_equal(M.e.trace().shape, (0,))
        except AttributeError:
            pass
        
class test_ram(TestCase):
    def test(self):
        M = MCMC(DisasterModel, db='ram')
        M.sample(500,100,2)
        assert_array_equal(M.e.trace().shape, (200,))
        assert_equal(M.e.trace.length(), 200)
        M.sample(100)
        assert_array_equal(M.e.trace().shape, (100,))
        assert_array_equal(M.e.trace(chain=None).shape, (300,))
        
    def test_regression_155(self):
        """thin > iter"""
        M = MCMC(DisasterModel, db='ram')
        M.sample(10,0,100)
        
class test_txt(TestCase):
    def test(self):
        try:
            os.removedir('txt_data')
        except:
            pass
        S = MCMC(DisasterModel, db='txt', dirname='txt_data', mode='w')
        S.sample(100)
        S.sample(100)
        S.db.close()
    
    def test_load(self):
        db = pymc.database.txt.load('txt_data')
        assert_equal(len(db.e._trace), 2)
        assert_array_equal(db.e().shape, (100,))
        assert_array_equal(db.e(chain=None).shape, (200,))
        """
        # This will not work until getstate() is implemented for txt backend
        S = MCMC(DisasterModel, db)
        S.sample(100)
        S.db.close()
        """
        
        
class test_pickle(TestCase):
    def __init__(*args, **kwds):
        TestCase.__init__(*args, **kwds)
        try: 
            os.remove('Disaster.pickle')
        except:
            pass
            
    def test(self):
        M = MCMC(DisasterModel, db='pickle', name='Disaster')
        M.sample(500,100,2)
        assert_array_equal(M.e.trace().shape, (200,))
        assert_equal(M.e.trace.length(), 200)
        M.db.close()
        
    def test_load(self):
        db = pymc.database.pickle.load('Disaster.pickle')
        S = MCMC(DisasterModel, db)
        S.sample(100,0,1)
        assert_equal(len(S.e.trace._trace),2)
        assert_array_equal(S.e.trace().shape, (100,))
        assert_array_equal(S.e.trace(chain=None).shape, (300,))
        assert_equal(S.e.trace.length(None), 300)
        S.db.close()



class test_mysql(TestCase):
    def test(self):
        if 'mysql' not in dir(pymc.database):
            raise nose.SkipTest
            
        M = MCMC(DisasterModel, db='mysql', name='pymc_test', dbuser='pymc', dbpass='bayesian', dbhost='www.freesql.org')
        M.db.clean()
        M.sample(50,10,thin=2)
        assert_array_equal(M.e.trace().shape, (20,))
        # Test second trace.
        M.sample(10)
        assert_array_equal(M.e.trace().shape, (10,))
        assert_array_equal(M.e.trace(chain=None).shape, (30,))
        assert_equal(M.e.trace.length(chain=1), 20)
        assert_equal(M.e.trace.length(chain=2), 10)
        assert_equal(M.e.trace.length(chain=None), 30)
        assert_equal(M.e.trace.length(chain=-1), 10)
        M.sample(5)
        assert_equal(M.e.trace.length(), 5)
        M.db.close()

    def test_load(self):
        if 'mysql' not in dir(pymc.database):
            raise nose.SkipTest
        
        db = pymc.database.mysql.load(dbname='pymc_test', dbuser='pymc', dbpass='bayesian', dbhost='www.freesql.org')
        assert_array_equal(db.e.length(chain=1), 20)
        assert_array_equal(db.e.length(chain=2), 10)
        assert_array_equal(db.e.length(chain=3), 5)
        assert_array_equal(db.e.length(chain=None), 35)
        S = MCMC(DisasterModel, db='mysql', name='pymc_test', dbuser='pymc', dbpass='bayesian', dbhost='www.freesql.org')
        S.sample(10)
        assert_array_equal(S.e.trace(chain=-1).shape, (10,))
        assert_array_equal(S.e.trace(chain=None).shape, (45,))
        S.db.clean()
        S.db.close()
            

class test_sqlite(TestCase):
    def __init__(*args, **kwds):
        TestCase.__init__(*args, **kwds)
        try:    
            os.remove('Disaster.sqlite')
            os.remove('Disaster.sqlite-journal')
        except:
            pass
            
    def test(self):
        if 'sqlite' not in dir(pymc.database):
            raise nose.SkipTest
    
        M = MCMC(DisasterModel, db='sqlite', name='Disaster')
        M.sample(500,100,2)
        assert_array_equal(M.e.trace().shape, (200,))
        # Test second trace.
        M.sample(100,0,1)
        assert_array_equal(M.e.trace().shape, (100,))
        assert_array_equal(M.e.trace(chain=None).shape, (300,))
        assert_equal(M.e.trace.length(chain=1), 200)
        assert_equal(M.e.trace.length(chain=2), 100)
        assert_equal(M.e.trace.length(chain=None), 300)
        assert_equal(M.e.trace.length(chain=-1), 100)
        M.sample(50)
        assert_equal(M.e.trace.length(), 50)
        M.db.close()
        
    """
    # This will not work until getstate() is implemented for sqlite backend    
    def test_load(self):
        if 'sqlite' not in dir(pymc.database):
            raise nose.SkipTest
        db = pymc.database.sqlite.load('Disaster.sqlite')
        assert_array_equal(db.e.length(chain=1), 200)
        assert_array_equal(db.e.length(chain=2), 100)
        assert_array_equal(db.e.length(chain=3), 50)
        assert_array_equal(db.e.length(chain=None), 350)
        S = MCMC(DisasterModel, db)
        S.sample(100,0,1)
        assert_array_equal(S.e.trace(chain=-1).shape, (100,))
        assert_array_equal(S.e.trace(chain=None).shape, (450,))
        S.db.close()
    """
    
    def test_interactive(self):
        M=MCMC(DisasterModel,db='sqlite')
        M.isample(1000)
        


class test_hdf5(TestCase):
    def __init__(*args, **kwds):
        TestCase.__init__(*args, **kwds)        
        try: 
            os.remove('Disaster.hdf52')
        except:
            pass
    
    def test(self):
        if 'hdf5' not in dir(pymc.database):
            raise nose.SkipTest
            
        S = MCMC(DisasterModel, db='hdf5', name='Disaster')
        S.sample(500,100,2)
        assert_array_equal(S.e.trace().shape, (200,))
        assert_equal(S.e.trace.length(), 200)
        assert_array_equal(S.D.value, DisasterModel.disasters_array)
        assert_equal(S.e.trace().__class__,  np.ndarray)
        S.db.close()
        
    def test_attribute_assignement(self):
        if 'hdf5' not in dir(pymc.database):
            raise nose.SkipTest
            
        arr = np.array([[1,2],[3,4]])
        db = pymc.database.hdf5.load('Disaster.hdf5', 'a')
        db.add_attr('some_list', [1,2,3])
        db.add_attr('some_dict', {'a':5})
        db.add_attr('some_array', arr, array=True)
        assert_array_equal(db.some_list, [1,2,3])
        assert_equal(db.some_dict['a'], 5)
        assert_array_equal(db.some_array.read(), arr)
        db.close()
        del db
        db = pymc.database.hdf5.load('Disaster.hdf5', 'a')
        assert_array_equal(db.some_list, [1,2,3])
        assert_equal(db.some_dict['a'], 5)
        assert_array_equal(db.some_array, arr)
        db.close()  

    def test_hdf5_col(self):
        if 'hdf5' not in dir(pymc.database):
            raise nose.SkipTest
        import tables
        db = pymc.database.hdf5.load('Disaster.hdf5')
        col = db.e.hdf5_col()
        assert col.__class__ == tables.table.Column
        assert_equal(len(col), len(db.e()))
        db.close()
        del db
        
    """   
    # This will not work until getstate() is implemented for hdf5
    def test_compression(self):
        if 'hdf5' not in dir(pymc.database):
            raise nose.SkipTest
            
        try: 
            os.remove('DisasterModelCompressed.hdf5')
        except:
            pass
        db = pymc.database.hdf5.Database('DisasterModelCompressed.hdf5', complevel=5)
        S = MCMC(DisasterModel,db)
        S.sample(450,100,1)
        assert_array_equal(S.e.trace().shape, (350,))
        S.db.close()
        db.close()
        del S
        
    def test_load(self):
        if 'hdf5' not in dir(pymc.database):
            raise nose.SkipTest
            
        db = pymc.database.hdf5.load('Disaster.hdf5', 'a')
        assert_array_equal(db._h5file.root.chain1.PyMCsamples.attrs.D, 
           DisasterModel.disasters_array)
        assert_array_equal(db.D, DisasterModel.disasters_array)
        S = MCMC(DisasterModel, db)
        
        S.sample(100,0,1)
        assert_array_equal(S.e.trace(chain=None).shape, (300,))
        assert_equal(S.e.trace.length(None), 300)
        db.close() # For some reason, the hdf5 file remains open.
        S.db.close()
        
        # test that the step method state was correctly restored.
        sm = S.step_methods.pop()
        assert(sm._accepted+sm._rejected ==600)
    
        
    def test_mode(self):
        if 'hdf5' not in dir(pymc.database):
            raise nose.SkipTest
            
        S = MCMC(DisasterModel, db='hdf5', name='Disaster', mode='w')
        try:
            tables = S.db._gettable(None)
        except LookupError:
            pass
        else:
            raise 'Mode not working'
        S.sample(100)
        S.db.close()
        S = MCMC(DisasterModel, db='hdf5', name='Disaster', mode='a')
        tables = S.db._gettable(None)
        assert_equal(len(tables), 1)
        S.db.close()
        del S
    """    
    
if __name__ == '__main__':
    
    C =nose.config.Config(verbosity=1)
    nose.runmodule(config=C)
    try:
        S.db.close()
    except:
        pass
