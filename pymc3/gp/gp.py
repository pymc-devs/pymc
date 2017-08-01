import numpy as np

import theano
import theano.tensor as tt
import theano.tensor.slinalg

import pymc3 as pm
from .cov import Covariance

__all__ = ['GPLatent', 'GPMarginal', 'TProcess', 'GPMarginalSparse']


cholesky = pm.distributions.dist_math.Cholesky(nofail=True, lower=True)
solve_lower = tt.slinalg.Solve(A_structure='lower_triangular')
solve_upper = tt.slinalg.Solve(A_structure='upper_triangular')

def stabilize(K):
    """ adds small diagonal to a covariance matrix """
    return K + 1e-6 * tt.identity_like(K)

# condition you give variables, kwargs you set as attrs.
# kwargs are in input to 'prior', not 'conditional'



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

    @property
    def X(self):
        inputs = getattr(self, "_X", None)
        if inputs is None:
            raise AttributeError("X not set, required argument")
        else:
            return inputs

    @X.setter
    def X(self, inputs):
        self._X = inputs


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

    def prior(self, name, n_points, X, **kwargs):
        raise NotImplementedError

    def conditional(self, name, n_points, Xnew, **kwargs):
        raise NotImplementedError


class Latent(GPBase):
    """ Where the GP f isnt integrated out, and is sampled explicitly
    """
    def __init__(self, mean_func=None, cov_func=None):
        super(Latent, self).__init__(mean_func, cov_func)

    @property
    def f(self):
        latent_func = getattr(self, "_f", None)
        if latent_func is None:
            raise AttributeError("f not set, required argument")
        else:
            return latent_func

    @f.setter
    def f(self, latent_func):
        self._f = latent_func

    def prior(self, name, n_points, X, reparameterize=True):
        f = self._build_prior(name, n_points, X, reparameterize)
        return f

    def conditional(self, name, n_points, Xnew, X=None, f=None):
        if X is None:
            X = self.X
        if f is None:
            f = self.f
        mu, chol = self._build_conditional(Xnew, X, f)
        return pm.MvNormal(name, mu=mu, chol=chol, shape=n_points)

    def _build_prior(self, name, n_points, X, reparameterize=True):
        mu = self.mean_func(X)
        chol = cholesky(stabilize(self.cov_func(X)))
        if reparameterize:
            rotated = pm.Normal(name + "_rotated_", mu=0.0, sd=1.0, shape=n_points)
            f = pm.Deterministic(name, mu + tt.dot(chol, rotated))
        else:
            f = pm.MvNormal(name, mu=mu, chol=chol, shape=n_points)
        self.X = X
        self.f = f
        return f

    def _build_conditional(self, Xnew, X, f):
        Kxx = self.cov_total(X)
        Kxs = self.cov_func(X, Xnew)
        Kss = self.cov_func(Xnew)
        L = cholesky(stabilize(Kxx))
        A = solve_lower(L, Kxs)
        cov = Kss - tt.dot(tt.transpose(A), A)
        rotated = solve_lower(L, f - self.mean_total(X))
        mu = self.mean_func(Xnew) + tt.dot(tt.transpose(A), rotated)
        chol = cholesky(stabilize(cov))
        return mu, chol



class Marginal(GPBase):
    def __init__(self, mean_func=None, cov_func=None):
        super(Marginal, self).__init__(mean_func, cov_func)

    @property
    def y(self):
        obs_data = getattr(self, "_y", None)
        if obs_data is None:
            raise AttributeError("y not set, required argument")
        else:
            return obs_data

    @y.setter
    def y(self, obs_data):
        self._y = obs_data

    @property
    def cov_noise(self):
        noise = getattr(self, "_noise", None)
        if noise is None:
            raise AttributeError("y not set, required argument")
        else:
            return noise

    @cov_noise.setter
    def cov_noise(self, noise):
        self._cov_noise = noise

    def _build_prior(self, X, cov_noise):
        mu = self.mean_func(X)
        Kxx = self.cov_total(X)
        Knx = cov_noise(X)
        cov = Kxx + Knx
        chol = cholesky(stabilize(cov))
        return mu, chol

    def prior(self, name, n_points, X, y, noise):
        if not isinstance(noise, Covariance):
            cov_noise = pm.gp.cov.WhiteNoise(noise)
        else:
            cov_noise = noise
        mu, chol = self._build_prior(X, cov_noise)
        self.X = X
        self.y = y
        self.cov_noise = cov_noise
        return pm.MvNormal(name, mu=mu, chol=chol, shape=n_points, observed=y)

    def _build_conditional(self, Xnew, X, y, cov_noise, include_noise):
        Kxx = self.cov_total(X)
        Kxs = self.cov_func(X, Xnew)
        Kss = self.cov_func(Xnew)
        Knx = cov_noise(X)
        rxx = y - self.mean_total(X)
        L = cholesky(stabilize(Kxx) + Knx)
        A = solve_lower(L, Kxs)
        v = solve_lower(L, rxx)
        mu = self.mean_func(Xnew) + tt.dot(tt.transpose(A), v)
        if include_noise:
            cov = cov_func_noise(Xnew) + Kss - tt.dot(tt.transpose(A), A)
        else:
            cov = stabilize(Kss) - tt.dot(tt.transpose(A), A)
        chol = cholesky(cov)
        return mu, chol

    def conditional(self, name, n_points, Xnew, X=None, y=None,
                    noise=None, include_noise=False):
        if X is None:
            X = self.X
        if y is None:
            y = self.y
        if noise is None:
            cov_noise = self.cov_noise
        else:
            if not isinstance(noise, Covariance):
                cov_noise = pm.gp.cov.WhiteNoise(noise)
            else:
                cov_noise = noise
        mu, chol = self._build_conditional(Xnew, X, y, cov_noise, include_noise)
        return pm.MvNormal(name, mu=mu, chol=chol, shape=n_points)


class MarginalSparse(GPBase):
    """ FITC and VFE sparse approximations
    """
    def __init__(self, cov_func, mean_func=None, approx=None):
        if approx is None:
            approx = "FITC"
        self.approx = approx
        super(MarginalSparse, self).__init__(cov_func, mean_func)

    def __add__(self, other):
        raise NotImplementedError("Additive Latent GP's not implemented")

    def kmeans_inducing_points(self, n_inducing, X):
        from scipy.cluster.vq import kmeans
        # first whiten X
        if isinstance(X, tt.TensorConstant):
            X = X.value
        elif isinstance(X, (np.ndarray, tuple, list)):
            X = np.asarray(X)
        else:
            raise ValueError(("To use K-means initialization, "
                              "please provide X as a type that "
                              "can be cast to np.ndarray, instead "
                              "of {}".format(type(X))))
        scaling = np.std(X, 0)
        # if std of a column is very small (zero), don't normalize that column
        scaling[scaling <= 1e-6] = 1.0
        Xw = X / scaling
        Xu, distortion = kmeans(Xw, n_inducing)
        return Xu * scaling

    def _prior_required_val(self, condition, **kwargs):
        X, y, Xu, sigma = (condition["X"], condition["y"],
                           condition["Xu"], condition["sigma"])
        return X, y, Xu, sigma

    def _build_prior_logp(self, X, y, Xu, sigma):
        sigma2 = tt.square(sigma)
        Kuu = self.cov_func(Xu)
        Kuf = self.cov_func(Xu, X)
        Luu = cholesky(stabilize(Kuu))
        A = solve_lower(Luu, Kuf)
        Qffd = tt.sum(A * A, 0)
        if self.approx == "FITC":
            Kffd = self.cov_func(X, diag=True)
            Lamd = tt.clip(Kffd - Qffd, 0.0, np.inf) + sigma2
            trace = 0.0
        elif self.approx == "VFE":
            Lamd = tt.ones_like(Qffd) * sigma2
            trace = ((1.0 / (2.0 * sigma2)) *
                     (tt.sum(self.cov_func(X, diag=True)) -
                      tt.sum(tt.sum(A * A, 0))))
        else:
            raise NotImplementedError(self.approx)
        A_l = A / Lamd
        L_B = cholesky(tt.eye(Xu.shape[0]) + tt.dot(A_l, tt.transpose(A)))
        r = y - self.mean_func(X)
        r_l = r / Lamd
        c = solve_lower(L_B, tt.dot(A, r_l))
        constant = 0.5 * self.size * tt.log(2.0 * np.pi)
        logdet = 0.5 * tt.sum(tt.log(Lamd)) + tt.sum(tt.log(tt.diag(L_B)))
        quadratic = 0.5 * (tt.dot(r, r_l) - tt.dot(c, c))
        return -1.0 * (constant + logdet + quadratic + trace)

    def prior(self, name, n_points, condition, **kwargs):
        X, y, Xu, sigma = self._prior_required_vals(condition, **kwargs)
        logp = lambda y: self._build_prior_logp(X, y, Xu, sigma)
        return pm.DensityDist(name, logp, observed=y)

    def _cond_required_val(self, condition, **kwargs):
        X, y, Xu, Xnew, sigma = (condition["X"], condition["y"], condition["Xu"],
                               condition["Xnew"], condition["sigma"])
        include_noise = kwargs.pop("include_noise", False)
        return X, y, Xu, Xnew, sigma, include_noise

    def _build_conditional(self, X, y, Xnew, Xu, sigma, include_noise):
        sigma2 = tt.square(sigma)
        Kuu = self.cov_func(Xu)
        Kuf = self.cov_func(Xu, X)
        Luu = cholesky(stabilize(Kuu))
        A = solve_lower(Luu, Kuf)
        Qffd = tt.sum(A * A, 0)
        if self.approx == "FITC":
            Kffd = self.cov_func(X, diag=True)
            Lamd = tt.clip(Kffd - Qffd, 0.0, np.inf) + sigma2
        elif self.approx == "VFE":
            Lamd = tt.ones_like(Qffd) * sigma2
        else:
            raise NotImplementedError(self.approx)
        A_l = A / Lamd
        L_B = cholesky(tt.eye(Xu.shape[0]) + tt.dot(A_l, tt.transpose(A)))
        r = y - self.mean_func(X)
        r_l = r / Lamd
        c = solve_lower(L_B, tt.dot(A, r_l))
        Kus = self.cov_func(Xu, Xnew)
        As = solve_lower(Luu, Kus)
        mean = self.mean_func(Xnew) + tt.dot(tt.transpose(As), solve_upper(tt.transpose(L_B), c))
        C = solve_lower(L_B, As)
        if include_noise:
            cov = (self.cov_func(Xnew) - tt.dot(tt.transpose(As), As) +
                   tt.dot(tt.transpose(C), C) + sigma2*tt.eye(Xnew.shape[0]))
        else:
            cov = (self.cov_func(Xnew) - tt.dot(tt.transpose(As), As) +
                   tt.dot(tt.transpose(C), C))
        return mean, stabilize(cov)

    def conditional(self, name, n_points, condition, **kwargs):
        X, y, Xnew, Xu, sigma, include_noise = self._cond_required_vals(condition, **kwargs)
        mu, chol = self._build_conditional(X, y, Xnew, Xu, sigma)
        return pm.MvNormal(name, mu=mu, chol=chol, shape=n_points)



class TP(Latent):
    """ StudentT process
    this isnt quite right i think, check https://www.cs.cmu.edu/~andrewgw/tprocess.pdf
    """
    def __add__(self, other):
        raise ValueError("Student T processes aren't additive")

    def _prior_rv(self, X, nu):
        rotated = pm.Normal(self.name + "_rotated_", mu=0.0, sd=1.0, shape=self.size)
        mu, chol = self._build_prior(X)
        # below follows
        # https://stats.stackexchange.com/questions/68476/drawing-from-the-multivariate-students-t-distribution
        # Pymc3's MvStudentT doesnt do sqrt(self.v_chi2)
        v_chi2 = pm.ChiSquared("v_chi2_", nu)
        f = pm.Deterministic(self.name, ((tt.sqrt(nu) / v_chi2) *
                                   (mu + tt.dot(chol, rotated))))
        f.rotated = rotated
        return f

    def conditional(self, X, Xnew, **kwargs):
        mu, chol = self._build_conditional(X, Xnew)
        return pm.MvStudentT(self.name, self.nu, mu=mu, chol=chol, shape=self.size)


