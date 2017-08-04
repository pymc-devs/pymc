import numpy as np

import theano
import theano.tensor as tt
import theano.tensor.slinalg

import pymc3 as pm
from pymc3.gp.cov import Covariance
from pymc3.gp.util import conditioned_vars

__all__ = ['Latent', 'Marginal', 'TP', 'MarginalSparse']


cholesky = pm.distributions.dist_math.Cholesky(nofail=True, lower=True)
solve_lower = tt.slinalg.Solve(A_structure='lower_triangular')
solve_upper = tt.slinalg.Solve(A_structure='upper_triangular')

def stabilize(K):
    """ adds small diagonal to a covariance matrix """
    return K + 1e-6 * tt.identity_like(K)


class GPBase(object):
    """
    Base class
    """
    def __init__(self, mean_func=None, cov_func=None):
        # check if not None, args are correct subclasses.
        # easy for users to get this wrong
        if mean_func is None:
            mean_func = pm.gp.mean.Zero()

        if cov_func is None:
            cov_func = pm.gp.cov.Constant(0.0)

        self.mean_func = mean_func
        self.cov_func = cov_func

    @property
    def cov_total(self):
        total = getattr(self, "_cov_total", None)
        if total is None:
            return self.cov_func
        else:
            return total

    @cov_total.setter
    def cov_total(self, new_cov_total):
        self._cov_total = new_cov_total

    @property
    def mean_total(self):
        total = getattr(self, "_mean_total", None)
        if total is None:
            return self.mean_func
        else:
            return total

    @mean_total.setter
    def mean_total(self, new_mean_total):
        self._mean_total = new_mean_total

    def __add__(self, other):
        same_attrs = set(self.__dict__.keys()) == set(other.__dict__.keys())
        if not isinstance(self, type(other)) and not same_attrs:
            raise ValueError("cant add different GP types")

        # set cov_func and mean_func of new GP
        cov_total = self.cov_func + other.cov_func
        mean_total = self.mean_func + other.mean_func

        # update self and other mean and cov totals
        self.cov_total, self.mean_total = (cov_total, mean_total)
        other.cov_total, other.mean_total = (cov_total, mean_total)
        new_gp = self.__class__(mean_total, cov_total)
        return new_gp

    def prior(self, name, X, *args, **kwargs):
        raise NotImplementedError

    def marginal_likelihood(self, name, X, *args, **kwargs):
        raise NotImplementedError

    def conditional(self, name, n_points, Xnew, *args, **kwargs):
        raise NotImplementedError


@conditioned_vars(["X", "f"])
class Latent(GPBase):
    """ Where the GP f isnt integrated out, and is sampled explicitly
    """
    def __init__(self, mean_func=None, cov_func=None):
        super(Latent, self).__init__(mean_func, cov_func)

    def _build_prior(self, name, n_points, X, reparameterize=True):
        mu = self.mean_func(X)
        chol = cholesky(stabilize(self.cov_func(X)))
        if reparameterize:
            v = pm.Normal(name + "_rotated_", mu=0.0, sd=1.0, shape=n_points)
            f = pm.Deterministic(name, mu + tt.dot(chol, v))
        else:
            f = pm.MvNormal(name, mu=mu, chol=chol, shape=n_points)
        self.X = X
        self.f = f
        return f

    def prior(self, name, n_points, X, reparameterize=True):
        f = self._build_prior(name, n_points, X, reparameterize)
        return f

    def _build_conditional(self, Xnew, X, f):
        Kxx = self.cov_total(X)
        Kxs = self.cov_func(X, Xnew)
        Kss = self.cov_func(Xnew)
        L = cholesky(stabilize(Kxx))
        A = solve_lower(L, Kxs)
        cov = Kss - tt.dot(tt.transpose(A), A)
        chol = cholesky(stabilize(cov))
        v = solve_lower(L, f - self.mean_total(X))
        mu = self.mean_func(Xnew) + tt.dot(tt.transpose(A), v)
        return mu, chol

    def conditional(self, name, n_points, Xnew, X=None, f=None):
        if X is None: X = self.X
        if f is None: f = self.f
        mu, chol = self._build_conditional(Xnew, X, f)
        return pm.MvNormal(name, mu=mu, chol=chol, shape=n_points)


@conditioned_vars(["X", "f", "nu"])
class TP(Latent):
    """ StudentT process
    https://www.cs.cmu.edu/~andrewgw/tprocess.pdf
    """
    def __init__(self, mean_func=None, cov_func=None, nu=None):
        if nu is None:
            raise ValueError("T Process requires a degrees of freedom parameter, 'nu'")
        self.nu = nu
        super(TP, self).__init__(mean_func, cov_func)

    def __add__(self, other):
        raise ValueError("Student T processes aren't additive")

    def _build_prior(self, name, n_points, X, nu):
        mu = self.mean_func(X)
        chol = cholesky(stabilize(self.cov_func(X)))

        chi2 = pm.ChiSquared("chi2_", nu)
        v = pm.Normal(name + "_rotated_", mu=0.0, sd=1.0, shape=n_points)
        f = pm.Deterministic(name, (tt.sqrt(nu) / chi2) * (mu + tt.dot(chol, v)))

        self.X = X
        self.f = f
        self.nu = nu
        return f

    def prior(self, name, n_points, X, nu):
        f = self._build_prior(name, n_points, X, nu)
        return f

    def _build_conditional(self, Xnew, X, f, nu):
        Kxx = self.cov_total(X)
        Kxs = self.cov_func(X, Xnew)
        Kss = self.cov_func(Xnew)
        L = cholesky(stabilize(Kxx))
        A = solve_lower(L, Kxs)
        cov = Kss - tt.dot(tt.transpose(A), A)

        v = solve_lower(L, f - self.mean_total(X))
        mu = self.mean_func(Xnew) + tt.dot(tt.transpose(A), v)

        beta = tt.dot(v, v)
        nu2 = nu + X.shape[0]

        covT = (nu + beta - 2)/(nu2 - 2) * cov
        chol = cholesky(stabilize(covT))
        return nu2, mu, chol

    def conditional(self, name, n_points, Xnew, X=None, f=None, nu=None):
        if X is None: X = self.X
        if f is None: f = self.f
        if nu is None: nu = self.nu
        nu2, mu, chol = self._build_conditional(Xnew, X, f, nu)
        return pm.MvStudentT(name, nu=nu2, mu=mu, chol=chol, shape=n_points)


@conditioned_vars(["X", "y", "noise"])
class Marginal(GPBase):
    # should I rename 'prior' method to 'marginal_likelihood'?

    def __init__(self, mean_func=None, cov_func=None):
        super(Marginal, self).__init__(mean_func, cov_func)

    def _build_marginal_likelihood(self, X, noise):
        mu = self.mean_func(X)
        Kxx = self.cov_total(X)
        Knx = noise(X)
        cov = Kxx + Knx
        chol = cholesky(stabilize(cov))
        return mu, chol

    def marginal_likelihood(self, name, n_points, X, y, noise, is_observed=True):
        if not isinstance(noise, Covariance):
            noise = pm.gp.cov.WhiteNoise(noise)
        mu, chol = self._build_marginal_likelihood(X, noise)
        self.X = X
        self.y = y
        self.noise = noise
        if is_observed:
            return pm.MvNormal(name, mu=mu, chol=chol, observed=y)
        else:
            return pm.MvNormal(name, mu=mu, chol=chol, size=n_points)

    def _build_conditional(self, Xnew, X, y, noise, pred_noise):
        Kxx = self.cov_total(X)
        Kxs = self.cov_func(X, Xnew)
        Kss = self.cov_func(Xnew)
        Knx = noise(X)
        rxx = y - self.mean_total(X)
        L = cholesky(stabilize(Kxx) + Knx)
        A = solve_lower(L, Kxs)
        v = solve_lower(L, rxx)
        mu = self.mean_func(Xnew) + tt.dot(tt.transpose(A), v)
        if pred_noise:
            cov = cov_func_noise(Xnew) + Kss - tt.dot(tt.transpose(A), A)
        else:
            cov = stabilize(Kss) - tt.dot(tt.transpose(A), A)
        chol = cholesky(cov)
        return mu, chol

    def conditional(self, name, n_points, Xnew, X=None, y=None,
                    noise=None, pred_noise=False):
        if X is None: X = self.X
        if y is None: y = self.y
        if noise is None:
            noise = self.noise
        else:
            if not isinstance(noise, Covariance):
                noise = pm.gp.cov.WhiteNoise(noise)
        mu, chol = self._build_conditional(Xnew, X, y, noise, pred_noise)
        return pm.MvNormal(name, mu=mu, chol=chol, shape=n_points)


@conditioned_vars(["X", "Xu", "y", "sigma"])
class MarginalSparse(GPBase):
    _available_approx = ["FITC", "VFE", "DTC"]
    """ FITC and VFE sparse approximations
    """
    def __init__(self, mean_func=None, cov_func=None, approx="FITC"):
        self.approx = approx
        super(MarginalSparse, self).__init__(mean_func, cov_func)

    def __add__(self, other):
        # new_gp will default to FITC approx
        new_gp = super(MarginalSparse, self).__add__(other)
        # make sure new gp has correct approx
        if not self.approx == other.approx:
            raise ValueError("Cant add GPs with different approximations")
        new_gp.approx = self.approx
        return new_gp

    def _build_marginal_likelihood_logp(self, X, Xu, y, sigma):
        sigma2 = tt.square(sigma)
        Kuu = self.cov_func(Xu)
        Kuf = self.cov_func(Xu, X)
        Luu = cholesky(stabilize(Kuu))
        A = solve_lower(Luu, Kuf)
        Qffd = tt.sum(A * A, 0)
        if self.approx not in self._available_approx:
            raise NotImplementedError(self.approx)
        elif self.approx == "FITC":
            Kffd = self.cov_func(X, diag=True)
            Lamd = tt.clip(Kffd - Qffd, 0.0, np.inf) + sigma2
            trace = 0.0
        elif self.approx == "VFE":
            Lamd = tt.ones_like(Qffd) * sigma2
            trace = ((1.0 / (2.0 * sigma2)) *
                     (tt.sum(self.cov_func(X, diag=True)) -
                      tt.sum(tt.sum(A * A, 0))))
        else: # DTC
            Lamd = tt.ones_like(Qffd) * sigma2
            trace = 0.0
        A_l = A / Lamd
        L_B = cholesky(tt.eye(Xu.shape[0]) + tt.dot(A_l, tt.transpose(A)))
        r = y - self.mean_func(X)
        r_l = r / Lamd
        c = solve_lower(L_B, tt.dot(A, r_l))
        constant = 0.5 * X.shape[0] * tt.log(2.0 * np.pi)
        logdet = 0.5 * tt.sum(tt.log(Lamd)) + tt.sum(tt.log(tt.diag(L_B)))
        quadratic = 0.5 * (tt.dot(r, r_l) - tt.dot(c, c))
        return -1.0 * (constant + logdet + quadratic + trace)

    def marginal_likelihood(self, name, n_points, X, Xu, y, sigma, is_observed=True):
        self.X = X
        self.Xu = Xu
        self.y = y
        self.sigma = sigma
        logp = lambda y: self._build_marginal_likelihood_logp(X, Xu, y, sigma)
        if is_observed:
            return pm.DensityDist(name, logp, observed=y)
        else:
            return pm.DensityDist(name, logp, size=n_points) # need size? if not, dont need size arg

    def _build_conditional(self, Xnew, Xu, X, y, sigma, pred_noise):
        sigma2 = tt.square(sigma)
        Kuu = self.cov_func(Xu)
        Kuf = self.cov_func(Xu, X)
        Luu = cholesky(stabilize(Kuu))
        A = solve_lower(Luu, Kuf)
        Qffd = tt.sum(A * A, 0)
        if self.approx not in self._available_approx:
            raise NotImplementedError(self.approx)
        elif self.approx == "FITC":
            Kffd = self.cov_func(X, diag=True)
            Lamd = tt.clip(Kffd - Qffd, 0.0, np.inf) + sigma2
        else: # VFE or DTC
            Lamd = tt.ones_like(Qffd) * sigma2
        A_l = A / Lamd
        L_B = cholesky(tt.eye(Xu.shape[0]) + tt.dot(A_l, tt.transpose(A)))
        r = y - self.mean_func(X)
        r_l = r / Lamd
        c = solve_lower(L_B, tt.dot(A, r_l))
        Kus = self.cov_func(Xu, Xnew)
        As = solve_lower(Luu, Kus)
        mean = (self.mean_func(Xnew) +
                tt.dot(tt.transpose(As), solve_upper(tt.transpose(L_B), c)))
        C = solve_lower(L_B, As)
        if pred_noise:
            cov = (self.cov_func(Xnew) - tt.dot(tt.transpose(As), As) +
                   tt.dot(tt.transpose(C), C) + sigma2*tt.eye(Xnew.shape[0]))
        else:
            cov = (self.cov_func(Xnew) - tt.dot(tt.transpose(As), As) +
                   tt.dot(tt.transpose(C), C))
        return mean, stabilize(cov)

    def conditional(self, name, n_points, Xnew, Xu=None, X=None, y=None,
                    sigma=None, pred_noise=False):
        if Xu is None: Xu = self.Xu
        if X is None: X = self.X
        if y is None: y = self.y
        if sigma is None: sigma = self.sigma
        mu, chol = self._build_conditional(Xnew, Xu, X, y, sigma, pred_noise)
        return pm.MvNormal(name, mu=mu, chol=chol, shape=n_points)




