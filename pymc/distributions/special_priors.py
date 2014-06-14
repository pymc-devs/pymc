
from .multivariate import InverseWishart
from .continuous import InverseGamma
from sys import float_info
import numpy as np
from pymc.distributions.continuous import InverseGamma
from theano.sandbox.linalg.ops import diag, psd

def create_noninformative_covariance_prior(name1, name2, d, model=None):
    '''
    Construct a two part noninformative prior for the covariance matrix
    following Huang and Wang, "Simple Marginally Noninformative Prior Distributions
    for Covariance Matrices" ( http://ba.stat.cmu.edu/journal/2013/vol08/issue02/huang.pdf )
    
    The resulting prior has an almost flat half-t distribution over the variance of the variables,
    while efficiently having an uniform-prior (range [-1,1] for the correlation coefficients.
        
    Arguments:
        name1: Name for the Inverse Wishart distribution which will be created
        name2: Name for the Inverse Gamma distribution which will be created as a prior for the diagonal elements of the inv_S param of the Inverse Wishart
        d: Dimensionality, i.e. number of variables to create a joint covariance prior for.
        model: (optional) the model
    Returns:
    A tuple consisting of the covariance prior (an InverseWishart), and the hyperprior (InverseGamma) for the
    diagonal elements of the InverseWishart inv_S parameter
    '''

    A = float_info.max / 4. # Large number
    d_ones = np.ones(d, dtype=np.float64)
    a_hyperprior = InverseGamma(name=name2, d_ones/2., d_ones / A, model=model)
    '  Note that the InverseWishart in this library is parameterized with the inverse of S, So we do not divide by a_hyperprior, but use it more directly.'
    cov_prior = InverseWishart(name=name1, d+1, diag(4. * a_hyperprior), model=model)
    return cov_prior, a_hyperprior