import pymc as pm
from numpy.testing import *
import numpy as np
import nose
import sys

from pymc import six
xrange = six.moves.xrange

DIR = 'testresults'


def mymodel():
    mu = pm.Normal('mu', 0, 1)
    N = [pm.Normal('N_%i' % i, mu, 1) for i in xrange(3)]
    z1 = pm.Lambda('z1', lambda n=N: np.sum(n))
    z2 = pm.Lambda('z2', lambda n=N: np.sum(n))

    @pm.potential
    def y(z1=z1, z2=z2, mu=mu):
        return 0
    return mu, N, z1, z2, y


def powerset(seq):
    """
    Returns all the subsets of this set. This is a generator.

    From http://blog.technomancy.org/2009/3/17/a-powerset-generator-in-python
    """
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]] + item
            yield item

class test_graph(TestCase):

    @dec.skipif(sys.version_info.major==3)
    def test_graph(self):
        try:
            import pydot
        except ImportError:
            raise nose.SkipTest
        mu, N, z1, z2, y = mymodel()
        for mods in [[mu], [mu, N], [mu, N, z1, z2], [mu, N, z1, z2, y]]:
            for args in powerset([('collapse_deterministics', True), ('collapse_potentials', True), ('label_edges', False), ('legend', True), ('consts', True)]):
                M = pm.Model(mods)
                pm.graph.graph(M, path=DIR, **dict(args))
                
    @dec.skipif(sys.version_info.major==3)
    def test_moral(self):
        try:
            import pydot
        except ImportError:
            raise nose.SkipTest
        mu, N, z1, z2, y = mymodel()
        for mods in [[mu], [mu, N], [mu, N, z1, z2], [mu, N, z1, z2, y]]:
            M = pm.Model(mods)
            pm.graph.moral_graph(M, path=DIR)


if __name__ == '__main__':
    C = nose.config.Config(verbosity=1)
    nose.runmodule(config=C)
