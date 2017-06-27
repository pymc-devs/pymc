import numpy as np
from scipy import stats
from tqdm import tqdm

from theano.tensor.nlinalg import matrix_inverse
import theano.tensor.nlinalg
import theano.tensor.slinalg
import theano.tensor as tt

from .mean import Zero, Mean
from .cov import Covariance
from ..distributions import MvNormal, Continuous, draw_values, generate_samples
from ..model import modelcontext
from ..distributions.dist_math import Cholesky

__all__ = ['GPFullConjugate', 'GPFullConjugate', 'sample_gp']

class GPBase(Continuous):
    def random(self, point=None, size=None, X_values=None, obs_noise=False, y=None, **kwargs):
        if X_values is None:
            # draw from prior
            mean, cov = self.prior(obs_noise)
        else:
            # draw from conditional
            mean, cov = self.conditional(X_values, y, obs_noise)

        mu, cov = draw_values([mean, cov], point=point)
        def _random(mean, cov, size=None):
            return stats.multivariate_normal.rvs(
                mean, cov, None if size == mean.shape else size)

        samples = generate_samples(_random,
                                   mean=mu, cov=cov,
                                   dist_shape=mu.shape,
                                   broadcast_shape=mu.shape,
                                   size=size)
        return samples

    def prior(self, obs_noise=False):
        raise NotImplementedError

    def conditional(self, Z, y, obs_noise=False):
        raise NotImplementedError

    def logp(self, y):
        raise NotImplementedError


class GPFullConjugate(GPBase):
    """Gausian process

    Parameters
    ----------
    X : array
        Grid of points to evaluate Gaussian process over.
    mean_func : Mean
        Mean function of Gaussian process
    cov_func : Covariance
        Covariance function of Gaussian process
    sigma2 : scalar or array
        Observation variance (defaults to zero)
    """
    def __init__(self, X, mean_func=None, cov_func=None, sigma2=None, *args, **kwargs):
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
        self.sigma2 = sigma2
        self.cholesky = Cholesky(nofail=True, lower=True)
        self.solve_lower = tt.slinalg.Solve(A_structure="lower_triangular")
        self.solve_upper = tt.slinalg.Solve(A_structure="upper_triangular")

        self.X = X
        self.nf = self.X.shape[0]

        self.mean = self.mode = self.M(X)
        kwargs.setdefault("shape", X.squeeze().shape)
        super(GPFullConjugate, self).__init__(*args, **kwargs)

    def prior(self, obs_noise=False):
        mean = self.M(self.X)
        if obs_noise:
            cov = self.K(self.X) + self.sigma2 * tt.eye(self.nf)
        else:
            cov = self.K(self.X) + 1e-6 * tt.eye(self.nf)
        return mean, cov

    def conditional(self, Z, y, obs_noise=False):
        nz = Z.shape[0]
        Kxx = self.K(self.X)
        Kxz = self.K(self.X, Z)
        Kzz = self.K(Z)

        r = y - self.M(self.X)
        L = self.cholesky(Kxx + self.sigma2 * tt.eye(self.nf))
        A = self.solve_lower(L, Kxz)
        V = self.solve_lower(L, r)
        mean = tt.dot(tt.transpose(A), V) + self.M(Z)
        if obs_noise:
            cov = Kzz - tt.dot(tt.transpose(A), A) + self.sigma2 * tt.eye(nz)
        else:
            cov = Kzz - tt.dot(tt.transpose(A), A) + 1e-6 * tt.eye(nz)
        return mean, cov

    def logp(self, y):
        mean = self.M(self.X)
        L = self.cholesky(self.K(self.X) + self.sigma2 * tt.eye(self.nf))
        return MvNormal.dist(mu=mean, chol=L).logp(y)


class GPSparseConjugate(GPBase):
    """Sparse Gausian Process for IID Normal likelihoods.  Either VFE or FITC
    """
    def __init__(self, X, mean_func=None, cov_func=None, inducing_points=None, sigma2=None, approx="FITC", *args, **kwargs):
        if approx.upper() not in ["VFE", "FITC"]:
            raise ValueError("'FITC' or 'VFE' are the implemented GP approximations")
        else:
            if approx.upper() == "FITC":
                self.fitc = True
            else: #VFE
                self.fitc = False

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
        self.sigma2 = sigma2
        self.cholesky = Cholesky(nofail=True, lower=True)
        self.solve_lower = tt.slinalg.Solve(A_structure="lower_triangular")
        self.solve_upper = tt.slinalg.Solve(A_structure="upper_triangular")

        self.X = X
        self.nf = self.X.shape[0]

        self.mean = self.mode = self.M(X)
        kwargs.setdefault("shape", X.squeeze().shape)

        self.Xu = inducing_points
        self.nu = self.Xu.shape[0]
        super(GPSparseConjugate, self).__init__(*args, **kwargs)

    def prior(self, obs_noise=False):
        Kuu, Kuf, Kffd, Luu, A = self._common1()
        Qff = tt.dot(tt.transpose(A), A)
        mean = self.M(self.X)

        if obs_noise:
            cov = Qff - (tt.diag(Qff) - Kdiag) + self.sigma2 * tt.eye(self.nf)
        else:
            cov = Qff - (tt.diag(Qff) - Kdiag) + 1e-6 * tt.eye(self.nf)
        return mean, cov

    def conditional(self, Xs, y, obs_noise=False):
        Kuu = self.K(self.Xu, self.Xu) + 1e-6 * tt.eye(self.nu)
        Kuf = self.K(self.Xu, self.X)
        Luu = self.cholesky(Kuu)
        A = self.solve_lower(Luu, Kuf)
        Qffd = tt.sum(A * A, 0)
        if self.fitc:
            Kffd = self.K(self.X, diag=True)
            Lamd = tt.clip(Kffd - Qffd, 0.0, np.inf) + self.sigma2
        else: # VFE
            Lamd = tt.ones_like(Qffd) * self.sigma2
        A_l = A / Lamd
        L_B = self.cholesky(tt.eye(self.nu) + tt.dot(A_l, tt.transpose(A)))
        r = y - self.M(self.X)
        r_l = r / Lamd
        c = self.solve_lower(L_B, tt.dot(A, r_l))
        Kus = self.K(self.Xu, Xs)
        As = self.solve_lower(Luu, Kus)
        mean = tt.dot(tt.transpose(As), self.solve_upper(tt.transpose(L_B), c))
        C = self.solve_lower(L_B, As)
        if obs_noise:
            cov = self.K(Xs, Xs) - tt.dot(tt.transpose(As), As)\
                                 + tt.dot(tt.transpose(C), C)\
                                 + self.sigma2*tt.eye(Xs.shape[0])
        else:
            cov = self.K(Xs, Xs) - tt.dot(tt.transpose(As), As)\
                                 + tt.dot(tt.transpose(C), C)\
                                 + 1e-6 * tt.eye(Xs.shape[0])
        return mean, cov

    def logp(self, y):
        Kuu = self.K(self.Xu, self.Xu) + 1e-6 * tt.eye(self.nu)
        Kuf = self.K(self.Xu, self.X)
        Luu = self.cholesky(Kuu)
        A = self.solve_lower(Luu, Kuf)
        Qffd = tt.sum(A * A, 0)
        if self.fitc:
            Kffd = self.K(self.X, diag=True)
            Lamd = tt.clip(Kffd - Qffd, 0.0, np.inf) + self.sigma2
            trace = 0.0
        else: # VFE
            Lamd = tt.ones_like(Qffd) * self.sigma2
            trace = (1.0 / (2.0 * self.sigma2)) * (tt.sum(self.K(self.X, diag=True)) - tt.sum(tt.sum(A * A, 0)))
        A_l = A / Lamd
        L_B = self.cholesky(tt.eye(self.nu) + tt.dot(A_l, tt.transpose(A)))
        r = y - self.M(self.X)
        r_l = r / Lamd
        c = self.solve_lower(L_B, tt.dot(A, r_l))
        constant = 0.5 * self.nf * tt.log(2.0 * np.pi)
        logdet = 0.5 * tt.sum(tt.log(Lamd)) + tt.sum(tt.log(tt.diag(L_B)))
        quadratic = 0.5 * (tt.dot(r, r_l) - tt.dot(c, c))
        return -1.0 * (constant + logdet + quadratic + trace)


def sample_gp(trace, gp, X_values, samples=None, obs_noise=True, model=None, random_seed=None, progressbar=True):
    """Generate samples from a posterior Gaussian process.

    Parameters
    ----------
    trace : backend, list, or MultiTrace
        Trace generated from MCMC sampling.
    gp : Gaussian process object
        The GP variable to sample from.
    X_values : array
        Grid of new values at which to sample GP.  If `None`, returns
        samples from the prior.
    samples : int
        Number of posterior predictive samples to generate. Defaults to the
        length of `trace`
    obs_noise : bool
        Flag for including observation noise in sample. Defaults to False.
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

    y = [v for v in model.observed_RVs if v.name==gp.name][0]
    samples = [gp.distribution.random(point=trace[idx], X_values=X_values, y=y, obs_noise=obs_noise) for idx in indices]
    return np.array(samples)
