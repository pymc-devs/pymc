import numpy as np
from scipy import stats
from tqdm import tqdm

from theano.tensor.nlinalg import matrix_inverse
import theano.tensor as tt

from .mean import Zero, Mean
from .cov import Covariance
from ..distributions import TensorType, Normal, MvNormal, Continuous, draw_values, generate_samples
from ..model import Model, modelcontext, Deterministic, ObservedRV, FreeRV
from ..vartypes import string_types

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
    def __new__(cls, name, *args, **kwargs):
        #if name is _Unpickling:
        #    return object.__new__(cls)  # for pickle
        try:
            model = Model.get_context()
        except TypeError:
            raise TypeError("No model on context stack, which is needed to "
                            "use the Normal('x', 0,1) syntax. "
                            "Add a 'with model:' block")

        if isinstance(name, string_types):
            total_size = kwargs.pop('total_size', None)
            dist = cls.dist(*args, **kwargs)
            if abs(dist.sigma2 - 1e-6) > 1e-10:
                # If dist.sigma2 is not a small float, it is a normal GP
                return model.Var(name, dist, None, total_size)
            else:
                v = model.Var(name + "_n", Normal.dist(mu=0.0, sd=1.0, shape=dist.X.shape[0]))
                f = Deterministic(name, tt.dot(dist.Lx, v))
                return f
        else:
            raise TypeError("Name needs to be a string but got: {}".format(name))

    def __init__(self, name="gp", mean_func=None, cov_func=None, X=None,
                       sigma2=None, model=None, *args, **kwargs):
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

        if sigma2 is None:
            self.sigma2 = 1e-6
        else:
            self.sigma2 = sigma2
        self.K = cov_func
        self.triangular_solve = tt.slinalg.Solve("lower_triangular", lower=True)
        self.Kx = self.K(X) + self.sigma2 * tt.eye(X.shape[0])
        self.Lx = tt.slinalg.cholesky(self.Kx)
        if X is not None:
            self.X = X
            self.mean = self.mode = self.M(X)
            kwargs.setdefault("shape", X.squeeze().shape)
        super(GP, self).__init__(*args, **kwargs)

    def logp(self, Y, X=None):
        print("here")
        if X is not None:
            Lx = tt.slinalg.cholesky(self.K(X) + self.sigma2 * tt.eye(X.shape[0]))
            mu = self.M(X)
        else:
            Lx = self.Lx
            mu = self.mean
        return MvNormal.dist(mu, chol=Lx).logp(Y)

    @classmethod
    def _conditional(self, X, Z=None):
        if Z is None:
            S = self.Kx
            Lx = self.Lx
            mu = self.mean
        else:
            K_xz = self.K(X, Z)
            K_zz = self.K(Z, Z)
            A = self.triangular_solve(self.Lx, K_xz)
            S = K_zz - tt.dot(tt.transpose(A), A)
            Lx = tt.slinalg.cholesky(S + self.sigma2 * tt.eye(X.shape[0]))
            mu = tt.dot(tt.transpose(A), v).T  ## WHERE IS V?
        return mu, Lx

    def random(self, point=None, size=None, **kwargs):
        X = self.X
        Z = kwargs.pop("Z", None)
        mu, L = self._conditional(X, Z)
        return MvNormal.dist(mu, chol=L).random(point, size, **kwargs)


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
