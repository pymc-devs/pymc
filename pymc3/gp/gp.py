import numpy as np
from scipy import stats
from tqdm import tqdm

from theano.tensor.nlinalg import matrix_inverse
import theano.tensor as tt

from .mean import Zero, Mean
from .cov import Covariance
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
    X : array
        Grid of points to evaluate Gaussian process over. Only required if the 
        GP is not an observed variable.
    sigma : scalar or array
        Observation standard deviation (defaults to zero)
    """
    def __init__(self, mean_func=None, cov_func=None, X=None, sigma=0, *args, **kwargs):
        
        if mean_func is None:
            self.M = Zero()
        else:
            if not isinstance(mean_func, Mean):
                raise ValueError('mean_func must be a subclass of Mean')
            self.M = mean_func
            
        if cov_func is None:
            raise ValueError('A covariance function must be specified for GPP')
        if not isinstance(cov_func, Covariance):
            raise ValueError('cov_func must be a subclass of Covariance')
        self.K = cov_func
        
        self.sigma = sigma
        
        if X is not None:
            self.X = X
            self.mean = self.mode = self.M(X)
            kwargs.setdefault("shape", X.squeeze().shape)
            
        super(GP, self).__init__(*args, **kwargs)
                
    def random(self, point=None, size=None, **kwargs):
        X = self.X
        mu, cov = draw_values([self.M(X).squeeze(), self.K(X) + np.eye(X.shape[0])*self.sigma**2], point=point)
        
        def _random(mean, cov, size=None):
            return stats.multivariate_normal.rvs(
                mean, cov, None if size == mean.shape else size)

        samples = generate_samples(_random,
                                   mean=mu, cov=cov,
                                   dist_shape=mu.shape,
                                   broadcast_shape=mu.shape,
                                   size=size)
        return samples

    def logp(self, Y, X=None):
        if X is None:
            X = self.X
        mu = self.M(X).squeeze()
        Sigma = self.K(X) + tt.eye(X.shape[0])*self.sigma**2

        return MvNormal.dist(mu, Sigma).logp(Y)
        

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
    random_seed : integer > 0
        Random number seed for sampling.
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
        indices = np.random.randint(0, len(trace), samples)

    K = gp.distribution.K 
        
    data = [v for v in model.observed_RVs if v.name==gp.name][0].data

    X = data['X']
    Y = data['Y']
    Z = X_values
    
    S_xz = K(X, Z)
    S_zz = K(Z)
    if obs_noise:
        S_inv = matrix_inverse(K(X) + tt.eye(X.shape[0])*gp.distribution.sigma**2)
    else:
        S_inv = matrix_inverse(K(X))

    # Posterior mean
    m_post = tt.dot(tt.dot(S_xz.T, S_inv), Y)
    # Posterior covariance
    S_post = S_zz - tt.dot(tt.dot(S_xz.T, S_inv), S_xz)

    gp_post = MvNormal.dist(m_post, S_post, shape=Z.shape[0])
    
    samples = [gp_post.random(point=trace[idx]) for idx in indices]
    
    return np.array(samples)
