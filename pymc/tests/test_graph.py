import pymc as pm
from numpy.testing import *
import os
import numpy as np
import nose
DIR = 'testresults'

def mymodel():
    mu=pm.Normal('mu',0,1)
    N= [pm.Normal('N_%i'%i,mu,1) for i in xrange(10)]
    z = pm.Lambda('z',lambda n=N: np.sum(n))
    @pm.potential
    def y(z=z):
        return -z**2
    return mu,N,z,y
    
def powerset(seq):
    """
    Returns all the subsets of this set. This is a generator.
    """
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]]+item
            yield item
    
class test_graph(TestCase):
    @classmethod
    def setUpClass(self):
        try:
            os.mkdir(DIR)
        except:
            pass
        os.chdir(DIR)

    @classmethod
    def tearDownClass(self):
        os.chdir('..')

    def test_graph(self):
        try:
            import pydot
        except ImportError:
            raise nose.SkipTest
        mu,N,z,y = mymodel()
        for mods in [[mu], [mu,N], [mu,N,z], [mu,N,z,y]]:
            for args in powerset([('collapse_deterministics', True), ('collapse_potentials', True), ('label_edges', False), ('legend', True), ('consts', True)]):
                M = pm.Model(mods)
                pm.graph.graph(M, **dict(args))
                
    def test_moral(self):
        try:
            import pydot
        except ImportError:
            raise nose.SkipTest
        mu,N,z,y = mymodel()
        for mods in [[mu], [mu,N], [mu,N,z], [mu,N,z,y]]:
            M = pm.Model(mods)
            pm.graph.moral_graph(M)




if __name__ == '__main__':
    C =nose.config.Config(verbosity=1)
    nose.runmodule(config=C)

