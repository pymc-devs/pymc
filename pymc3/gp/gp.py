import numpy as np
from .mean import Zero
from ..distributions import MvNormal, Continuous, draw_values, generate_samples
from theano.tensor.nlinalg import matrix_inverse, det
import theano.tensor as tt
import theano
from scipy import stats
from ..model import modelcontext

__all__ = ['ConjugatePred']        

def ConjugatePred(name, K, X, Y, Z, sigma=0):

    S_xz = K(X, Z)
    S_zz = K(Z)
    S_inv = matrix_inverse(K(X))

    # Posterior mean
    m_post = tt.dot(tt.dot(S_xz.T, S_inv), Y)
    # Posterior covariance
    S_post = S_zz - tt.dot(tt.dot(S_xz.T, S_inv), S_xz) + tt.eye(Z.shape[0])*(sigma**2)

    return MvNormal(name, m_post, S_post, shape=Z.shape[0])
