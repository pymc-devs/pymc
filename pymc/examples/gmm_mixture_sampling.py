import numpy as np
from pymc import *

from pymc.step_methods.metropolis_hastings import *
from pymc.distributions.mixtures import *
from pymc.distributions.special import gammaln
from pymc.distributions.dist_math import *

import theano
import theano.tensor as T
from theano.ifelse import ifelse as theano_ifelse
import pandas
import matplotlib.pylab as plt
import numpy.linalg

theano.config.mode = 'FAST_COMPILE'
theano.config.compute_test_value = 'raise'

''' 
Example which demonstrates the use of the MvGaussianMixture as a flexible  (i.e. arbitrary) Prior
and the usage of the MetropolisHastings step method with a custom Proposal 
'''

# Some helper methods
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def tmin(a,b):
        return T.switch(T.lt(a,b), a, b)
    
def tmax(a,b):
        return T.switch(T.gt(a,b), a, b)

''' We construct a number of valid 
parameters for the multivariate normal.

In order to ensure that the covariance matrices are positive semidefinite, they are risen to the (matrix-) power of two.
Precision matrices are obtained by inverting the covariance matrices.
'''
def create_model(with_observations=False):

    # First, we create some parameters for a Multivariate Normal
    mu0 = np.ones((2), dtype=np.float64)
    cov0 = np.eye(2, dtype=np.float64) * 0.2
    tau0 = numpy.linalg.inv(cov0)
    
    mu1 = mu0+5.5
    cov1 = cov0.copy() * 1.4
    cov1[0,1] = 0.8
    cov1[1,0] = 0.8
    cov1 = np.linalg.matrix_power(cov1,2)
    tau1 = numpy.linalg.inv(cov1)
    
    mu2 = mu0-2.5
    cov2 = cov0.copy() * 1.4
    cov2[0,1] = 0.8
    cov2[1,0] = 0.8
    cov2 = np.linalg.matrix_power(cov2,2)
    tau2 = numpy.linalg.inv(cov2)
    
    mu3 = mu0*-0.5 + 2.0*mu2
    cov3 = cov0.copy() * 0.9
    cov3[0,1] = -0.8
    cov3[1,0] = -0.8
    cov3 = np.linalg.matrix_power(cov3,2)
    tau3 = numpy.linalg.inv(cov3)
    
    m2 = np.array([3.0])
    aval = np.array([4, 3, 2., 1.])
    a = constant(aval)
    
    model = Model()
    
    with model:
        k = 4    
        p, p_m1 = model.TransformedVar(
            'p', Dirichlet.dist(a, shape=k),
          simplextransform)
    
        c = Categorical('c', p)
        gmm = MvGaussianMixture('gmm', shape=mu0.shape, cat_var=c, mus=[mu0,mu1,mu2,mu3], taus=[tau0, tau1,tau2,tau3])
        ' Now, we ensure these arbitrary values sum to one. What do we get ? A nice probability which we can, in turn, sample from (or observe)'
        fac = T.constant( np.array([10.0], dtype=np.float64))
        gmm_p = Deterministic('gmm_p', (tmax(tmin(gmm[0], fac) / fac, T.constant(np.array([0.0], dtype=np.float64)))))
        pbeta = Beta('pbeta', 1,1)
        if (with_observations):
            result = Bernoulli('result', p=gmm_p[0], observed=np.array([1,0,1], dtype=np.bool))
        else:
            result = Bernoulli('result', p=gmm_p[0]) 
        ' Try this with a Metropolis instance, and watch it fail ..'
        step = MetropolisHastings(vars=model.vars, proposals=[GMMProposal(gmm)])
    return model, step

def plot_scatter_matrix(title, tr, fig=None):
    if (fig is None):
        fig = plt.Figure()
    t6 = pandas.Series(tr['c'])
    t8 = pandas.Series(tr['gmm'][:,0])
    t9 = pandas.Series(tr['gmm'][:,1])
    t10 = pandas.Series(tr['gmm_p'][:,0])
    t11 = pandas.Series(tr['pbeta'])
    df = pandas.DataFrame({'cat' : t6, 'gmm_0' : t8, 'gmm_1' : t9, 'p' : t10, 'pbeta' : t11})
    pandas.scatter_matrix(df)
    plt.title(title)
    return fig

def plot_p_prior_vs_posterior(prior_trace, posterior_trace, fig=None):
    if (fig is None):
        fig = plt.Figure()
    pr = pandas.Series(prior_trace['gmm_p'][:,0])
    po = pandas.Series(posterior_trace['gmm_p'][:,0])
    plt.hist(pr, bins=40, range=(-0.1, 1.1), color='b', alpha=0.5, label='Prior')
    plt.hist(po, bins=40, range=(-0.1, 1.1), color='g', alpha=0.5, label='Posterior')
    plt.legend()
    plt.show()

def run(n=10000):
    m1, step1 = create_model(False)
    m2, step2 = create_model(True)
    with m1:
        trace1 = sample(n, step1)
    with m2:
        trace2 = sample(n, step2)
    fig1 = plot_scatter_matrix('Prior Scatter Matrix', trace1)
    plt.show()
    fig2 = plot_scatter_matrix('Posterior Scatter Matrix', trace2)
    plt.show()
    plot_p_prior_vs_posterior(trace1, trace2)
    plt.show()
    
if __name__ == '__main__':
    run()
    print "Done"
    

