import numpy as np
from pymc import *

from pymc.step_methods.metropolis_hastings import *
from pymc.distributions.mixtures import *
from pymc.distributions.special import gammaln
from pymc.distributions.dist_math import *

import theano
import theano.tensor as T
import pandas
import matplotlib.pylab as plt
import numpy.linalg

theano.config.mode = 'FAST_COMPILE'
theano.config.compute_test_value = 'raise'

mu0 = np.ones((2), dtype=np.float64)
cov0 = np.eye(2, dtype=np.float64)
tau0 = numpy.linalg.inv(cov0)

mu1 = mu0+1.5
cov1 = cov0.copy() * 1.4
cov1[0,1] = 0.8
cov1[1,0] = 0.8
tau1 = numpy.linalg.inv(cov1)

mu2 = mu0-0.5
cov2 = cov0.copy() * 1.4
cov2[0,1] = 0.8
cov2[1,0] = 0.8
tau2 = numpy.linalg.inv(cov2)

mu3 = mu0/0.5 + mu2-0.6
cov3 = cov0.copy() * 1.4
cov3[0,1] = -0.4
cov3[1,0] = -0.4
tau3 = numpy.linalg.inv(cov2)

m2 = np.array([3.0])
aval = np.array([4, 1, 4., 2.])


def test_dirichlet_args(aval, kval, value):
        a = t.dvector('a')
        a.tag.test_value = aval
        
        k = t.dscalar('k')
        k.tag.test_value = kval
        # only defined for sum(value) == 1
        res = bound(
            sum(logpow_relaxed(
                value, a - 1) - gammaln(a), axis=0) + gammaln(sum(a)),

            k > 1,
            all(a > 0))
        rf = theano.function([a, k], res)
        return rf(aval, kval)

print "Testing dirichlet arguments"
print test_dirichlet_args(aval, 4, aval) 

a = constant(aval)

gma = gammaln(a)
gmaf = theano.function([], gma)
print "Gammaln of %r is %r" % (aval, gmaf())
model = Model()
with model:

    k = 4    
    pa = a / T.sum(a)

    p, p_m1 = model.TransformedVar(
        'p', Dirichlet.dist(a, shape=k),
      simplextransform)

    c = Categorical('c', pa)
    gmm = MvGaussianMixture('gmm', shape=mu0.shape, cat_var=c, mus=[mu0,mu1,mu2,mu3], taus=[tau0, tau1,tau2,tau3])
    ' Now, we ensure these arbitrary values sum to one. What do we get ? A nice probability which we can, in turn, sample from (or observe)'
    gmm_p = Deterministic('gmm_p', gmm[0] / T.sum(gmm)) 
    result = Bernoulli('result', p=gmm_p, observed=np.array([0,1,0,0,0,0,1,1,1], dtype=np.int32))
    
def run(n=50000):
    if n == "short":
        n = 50
    with model:
        ' Try this with a Metropolis instance, and watch it fail ..'
        step = MetropolisHastings()
        trace = sample(n, step)
    return trace
if __name__ == '__main__':
    tr = run()
    t1 = pandas.Series(tr['p'][:,0])
    t2 = pandas.Series(tr['p'][:,1])
    t3 = pandas.Series(tr['p'][:,2])
    t4 = pandas.Series(tr['p'][:,3])
    t6 = pandas.Series(tr['c'])
    t7 = pandas.Series(tr['result'])
    df = pandas.DataFrame({'a' : t1,'b' : t2, 'c' : t3, 'd' : t4, 'cat' : t6, result : t7})
    pandas.scatter_matrix(df)
    plt.show()


