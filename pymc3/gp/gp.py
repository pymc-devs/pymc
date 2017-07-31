import numpy as np

import theano
import theano.tensor as tt
import theano.tensor.slinalg

import pymc3 as pm

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
    def __init__(self, cov_func, mean_func=None):
        if mean_func is None:
            self.mean_func = pm.gp.mean.Zero()
        else:
            self.mean_func = mean_func
        self.cov_func = cov_func
        self.mean_func = mean_func

    def __add__(self, other):
        same_attrs = set(self.__dict__.keys()) == set(other.__dict__.keys())
        if not isinstance(self, type(other)) or same_attrs:
            raise ValueError("cant add different GP types")
        cov_func = self.cov_func + other.cov_func
        mean_func = self.mean_func + other.mean_func
        # copy other attrs, like approx
        return type(self)(cov_func, mean_func)

    def prior(self, name, condition, n_points, **kwargs):
        return self._prior(name, condition, n_points, **kwargs)

    def conditional(self, name, condition, n_points, **kwargs):
        return self._conditional(name, condition, n_points, **kwargs)

    def _prior(self, **kwargs):
        raise NotImplementedError

    def _conditional(self, name, Xnew, **kwargs):
        raise NotImplementedError


class Latent(GPBase):
    """ Where the GP f isnt integrated out, and is sampled explicitly
    """
    def __init__(self, cov_func, mean_func=None):
        super(Latent, self).__init__(cov_func, mean_func)

    def prior(self, name, condition, n_points, **kwargs):
        x = condition["x"]
        f = self._build_prior(name, x, n_points)
        return f

    def conditional(self, name, condition, n_points, **kwargs):
        Xs = condition["Xnew"]
        X = condition["X"]
        f = condition["f"]
        mu, chol = self._build_conditional(X, f, Xs)
        return pm.MvNormal(name, mu=mu, chol=chol, shape=n_points)

    def _build_prior(self, name, X, n_points):
        mu = self.mean_func(X)
        chol = cholesky(stabilize(self.cov_func(X)))
        rotated = pm.Normal(name + "_rotated_", mu=0.0, sd=1.0, shape=n_points)
        f = pm.Deterministic(self.name, mu + tt.dot(chol, rotated))
        f.rotated = rotated
        return f

    def _build_conditional(self, X, f, Xs):
        Kxx = self.cov_func(X)
        Kxs = self.cov_func(X, Xs)
        Kss = self.cov_func(Xs)
        L = cholesky(stabilize(Kxx))
        A = solve_lower(L, Kxs)
        cov = Kss - tt.dot(tt.transpose(A), A)
        if not hasattr(f, "rotated"):
            pass
        else:
            mu = self.mean_func(Xs) + tt.dot(tt.transpose(A), self.rotated)
        chol = cholesky(stabilize(cov))
        return mu, chol



class Marginal(GPBase):
    def __init__(self, cov_func, mean_func=None):
        super(Marginal, self).__init__(cov_func, mean_func)

    def _prior_required_val(self, condition, **kwargs):
        X = condition["X"]
        y = condition["y"]
        sigma = condition.pop("sigma", None)
        cov_func_noise = condition.pop("cov_func_noise", None)
        cov_noise = self._to_noise_func(sigma, cov_func_noise)
        return X, y, cov_noise

    def _build_prior(self, X, cov_noise):
        mu = self.mean_func(X)
        Kxx = self.cov_func(X)
        Knx = cov_noise(X)
        cov = Kxx + Knx
        chol = cholesky(stabilize(cov))
        return mu, chol

    def prior(self, name, condition, n_points, **kwargs):
        X, y, cov_noise = self._prior_required_vals(self, condition, **kwargs)
        mu, chol = self._build_prior(X, cov_noise)
        return pm.MvNormal(name, mu=mu, chol=chol, shape=n_points, observed=y)

    def _cond_required_val(self, condition, **kwargs):
        Xs = condition["Xnew"]
        X = condition["X"]
        y = condition["y"]
        sigma = condition.pop("sigma", None)
        cov_func_noise = condition.pop("cov_func_noise", None)
        cov_noise = self._to_noise_func(sigma, cov_func_noise)
        return X, y, Xs, cov_noise

    def _build_conditional(self, X, y, Xs, cov_noise, include_noise):
        Kxx = self.cov_func(X)
        Kxs = self.cov_func(X, Xs)
        Kss = self.cov_func(Xs)
        Knx = cov_noise(X)
        r = y - self.mean_func(X)
        L = cholesky(stabilize(Kxx) + Knx)
        A = solve_lower(L, Kxs)
        v = solve_lower(L, r)
        mu = self.mean_func(Xs) + tt.dot(tt.transpose(A), v)
        if include_noise:
            cov = cov_func_noise(Xs) + Kss - tt.dot(tt.transpose(A), A)
        else:
            cov = stabilize(Kss) - tt.dot(tt.transpose(A), A)
        chol = cholesky(cov)
        return mu, chol

    def conditional(self, name, condition, n_points, **kwargs):
        X, y, Xs, cov_noise, include_noise = self._cond_required_vals(condition, **kwargs)
        mu, chol = self._build_conditional(X, y, Xs, cov_noise, include_noise)
        return pm.MvNormal(name, mu=mu, chol=chol, shape=n_points)

    @staticmethod
    def _to_noise_func(sigma, cov_func_noise):
        # if sigma given, convert it to WhiteNoise covariance function
        if sigma is not None and cov_func_noise is not None:
            raise ValueError('Must provide one of sigma or cov_func_noise')
        if sigma is None and cov_func_noise is None:
            raise ValueError('Must provide a value or a prior for the noise variance')
        if sigma is not None and cov_func_noise is None:
            cov_func_noise = pm.gp.cov.WhiteNoise(sigma)
        return cov_func_noise



class MarginalSparse(GPBase):
    """ FITC and VFE sparse approximations
    """
    def __init__(self, cov_func, mean_func=None, approx=None):
        if approx is None:
            approx = "FITC"
        self.approx = approx
        super(GPMarginalSparse, self).__init__(cov_func, mean_func)

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

    def prior(self, name, condition, n_points, **kwargs):
        X, y, Xu, sigma = self._prior_required_vals(condition, **kwargs)
        logp = lambda y: self._build_prior_logp(X, y, Xu, sigma)
        return pm.DensityDist(name, logp, observed=y)

    def _cond_required_val(self, condition, **kwargs):
        X, y, Xu, Xs, sigma, include_noise = (condition["X"], condition["y"],
            condition["Xu"], condition["Xnew"], condition["sigma"], kwargs["incude_noise"])
        return X, y, Xu, Xs, sigma, include_noise

    def _build_conditional(self, X, y, Xs, Xu, sigma, include_noise):
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
        Kus = self.cov_func(Xu, Xs)
        As = solve_lower(Luu, Kus)
        mean = self.mean_func(Xs) + tt.dot(tt.transpose(As), solve_upper(tt.transpose(L_B), c))
        C = solve_lower(L_B, As)
        if include_noise:
            cov = (self.cov_func(Xs) - tt.dot(tt.transpose(As), As) +
                   tt.dot(tt.transpose(C), C) + sigma2*tt.eye(Xs.shape[0]))
        else:
            cov = (self.cov_func(Xs) - tt.dot(tt.transpose(As), As) +
                   tt.dot(tt.transpose(C), C))
        return mean, stabilize(cov)

    def conditional(self, name, condition, n_points, **kwargs):
        X, y, Xs, Xu, sigma, include_noise = self._cond_required_vals(condition, **kwargs)
        mu, chol = self._build_conditional(X, y, Xs, Xu, sigma)
        return pm.MvNormal(name, mu=mu, chol=chol, shape=n_points)



class TP(GPLatent):
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

    def conditional(self, X, Xs, **kwargs):
        mu, chol = self._build_conditional(X, Xs)
        return pm.MvStudentT(self.name, self.nu, mu=mu, chol=chol, shape=self.size)


