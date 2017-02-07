import numpy as np
from .mean import Zero
from ..distributions import MvNormal, Continuous, draw_values, generate_samples
from theano.tensor.nlinalg import matrix_inverse, det
import theano.tensor as tt
import theano
from scipy import stats
from ..model import modelcontext

__all__ = ['GP', 'GPPred']

class GP(Continuous):
    """Gausian process
    
    Parameters
    ----------
    mean_func : Mean
        Mean function of Gaussian process
    cov_func : Covariance
        Covariance function of Gaussian process
    sigma : scalar or array
        Observation noise (defaults to zero)
    """
    def __init__(self, mean_func=None, cov_func=None, sigma=0, *args, **kwargs):
        super(GP, self).__init__(*args, **kwargs)
        
        if mean_func is None:
            self.M = Zero()
        else:
            self.M = mean_func
            
        if cov_func is None:
            raise ValueError('A covariance function must be specified for GPP')
        self.K = cov_func
        
        self.sigma = sigma
                
    def random(self, point=None, size=None, **kwargs):
        X = kwargs.pop('X')
        mu, cov = draw_values([self.M(X).squeeze(), self.K(X)], point=point)

        def _random(mean, cov, size=None):
            return stats.multivariate_normal.rvs(
                mean, cov, None if size == mean.shape else size)

        samples = generate_samples(_random,
                                   mean=mu, cov=cov,
                                   dist_shape=self.shape,
                                   broadcast_shape=mu.shape,
                                   size=size)
        return samples

    def logp(self, X, Y):
        mu = self.M(X)
        k = mu.shape[0]
        tau = matrix_inverse((self.K + tt.eye(k)*self.sigma)(X))
        value = Y

        delta = value - mu

        result = k * tt.log(2 * np.pi) + tt.log(1. / det(tau))
        result += (delta.dot(tau) * delta).sum(axis=delta.ndim - 1)
        return -1 / 2. * result
        

def GPPred(name=None, gp=None, Z=None, obs_noise=True, model=None):
    
    model = modelcontext(model)
    
    if gp is None:
        raise ValueError('A GP object must be passed to predict')
            
    if name is None:
        name = gp.name + '_pred'
    
    if Z is None:
        raise ValueError('A grid of points must be passed to predict')

    K = gp.distribution.K 
    if obs_noise:
        K = K + gp.distribution.sigma
        
    data = [v for v in model.observed_RVs if v.name==gp.name][0].data

    X = data['X']
    Y = data['Y']
    
    S_xz = K(X, Z)
    S_zz = K(Z)
    S_inv = matrix_inverse(K(X))

    # Posterior mean
    m_post = tt.dot(tt.dot(S_xz.T, S_inv), Y)
    # Posterior covariance
    S_post = S_zz - tt.dot(tt.dot(S_xz.T, S_inv), S_xz)

    return MvNormal(name, m_post, S_post, shape=Z.shape[0])
    
