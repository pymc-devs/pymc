import numpy as np
from scipy import stats
from tqdm import tqdm

from theano.tensor.nlinalg import matrix_inverse, det
import theano.tensor as tt
import theano

from .mean import Zero
from ..distributions import MvNormal, Continuous, draw_values, generate_samples
from ..model import modelcontext


__all__ = ['GP', 'sample_gp']

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
        

def sample_gp(trace, gp, X_values, samples=None, obs_noise=True, model=None, random_seed=None, progressbar=True):
    """Generate samples from a posterior Gaussian process.

    Parameters
    ----------
    trace : backend, list, or MultiTrace
        Trace generated from MCMC sampling.
    gp : Gaussian process object
        The GP variable to sample from.
    X_values : array
        Grid of values at which to sample GP.
    samples : int
        Number of posterior predictive samples to generate. Defaults to the
        length of `trace`
    obs_noise : bool
        Flag for including observation noise in sample. Defaults to True.
    model : Model 
        Model used to generate `trace`. Optional if in `with` context manager.
    n_jobs : integer > 0
        Number of CPUs to use for sampling.
    progressbar : bool
        Flag for showing progress bar.
    
    Returns
    -------
    Array of samples from posterior GP evaluated at Z.
    """
    model = modelcontext(model)
    
    if samples is None:
        samples = len(trace)
    
    if random_seed:
        np.random.seed(random_seed)
    
    if progressbar:
        indices = tqdm(np.random.randint(0, len(trace), samples), total=samples)
    else:
        indices = randint(0, len(trace), samples)

    K = gp.distribution.K 
        
    data = [v for v in model.observed_RVs if v.name==gp.name][0].data

    X = data['X']
    Y = data['Y']
    Z = X_values
    
    S_xz = K(X, Z)
    S_zz = K(Z)
    if obs_noise:
        S_inv = matrix_inverse(K(X) + tt.eye(X.shape[0])*gp.distribution.sigma)
    else:
        S_inv = matrix_inverse(K(X))

    # Posterior mean
    m_post = tt.dot(tt.dot(S_xz.T, S_inv), Y)
    # Posterior covariance
    S_post = S_zz - tt.dot(tt.dot(S_xz.T, S_inv), S_xz)

    gp_post = MvNormal.dist(m_post, S_post, shape=Z.shape[0])
    
    samples = [gp_post.random(point=trace[idx]) for idx in indices]
    
    return np.array(samples)
