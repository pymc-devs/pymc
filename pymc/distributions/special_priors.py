
from .multivariate import InverseWishart
from .continuous import InverseGamma
from sys import float_info
import numpy as np
from pymc.distributions.continuous import InverseGamma
from theano.sandbox.linalg.ops import diag, psd

def huang_wang_covariance_prior(name1, name2, d, observed_scatter_matrix=None, observed_sample_size=0, model=None):
    '''
    Construct a noninformative or informative prior for a mv normal covariance matrix
    following Huang and Wang, "Simple Marginally Noninformative Prior Distributions
    for Covariance Matrices" ( http://ba.stat.cmu.edu/journal/2013/vol08/issue02/huang.pdf )
    
    If no observations are included, the resulting prior has an almost flat half-t distribution
    over the variance of the variables, while efficiently having an 
    uniform-prior (range [-1,1] for the correlation coefficients.
    
    This prior does not introduce a dependency between variance and correlation strength, as happens with
    a simple InverseWishart prior.
        
    Arguments:
        name1: Name for the Inverse Wishart distribution which will be created
        name2: Name for the Inverse Gamma distribution which will be created as a prior for the diagonal elements of the inv_S param of the Inverse Wishart
        d: Dimensionality, i.e. number of variables to create a joint covariance prior for.
        observed_scatter_matrix: d x d dimensional scatter matrix of (possibly virtual) observations to combine the noninformative prior with, to form an informative prior (or posterior).
        model: (optional) the model
    Returns:
    A tuple consisting of the covariance prior (an InverseWishart), and the hyperprior (InverseGamma) for the
    diagonal elements of the InverseWishart inv_S parameter
    '''

    A = float_info.max / 4. # Large number
    d_ones = np.ones(d, dtype=np.float64)
    a_hyperprior = InverseGamma(name=name2, d_ones/2., d_ones / A, model=model)
    S = diag(4. / a_hyperprior)
    if (observed_scatter_matrix is not None):
        S = S + observed_scatter_matrix
    cov_prior = InverseWishart(name=name1, d+1+observed_sample_size, S, model=model)
    return cov_prior, a_hyperprior