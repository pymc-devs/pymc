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


class GPBase(object):
    """
    Base class
    """
    def __init__(self, cov_func):
        self.cov_func = cov_func
        # force user to run `__call__` before `conditioned_on`
        self._ready = False

    def __add__(self, other):
        if not isinstance(self, type(other)):
            raise ValueError("cant add different GP types")
        cov_func = self.cov_func + other.cov_func
        return type(self)(cov_func)

    def __call__(self, name, size, mean_func):
        """Set state of GP constructor, separate function parameters from
        conditioning variables (data, random variables)
        This helps syntax, makes code clearer when defining likelihood, then
        predictive distribution off same gp object.
        """
        self.name = name
        self.size = size
        self.mean_func = mean_func

        # force user to run `__call__` before `conditioned_on`
        self._ready = True
        return self

    def conditioned_on(self, X, Xs=None, **kwargs):
        if self._ready:
            self._ready = False
            if Xs is None:
                # likelihood distribution
                X = tt.as_tensor_variable(X)
                return self._prior_rv(X, **kwargs)
            else:
                # predictive distribution
                #if not self._required_values_available:
                #    raise ValueError("have to train gp on data first")
                X = tt.as_tensor_variable(X)
                Xs = tt.as_tensor_variable(Xs)
                kwargs.update({"Xs": Xs})
                return self._predictive_rv(X, **kwargs)
        else:
            raise ValueError("use syntax gp(name, size, mean).conditioned_on(...)")

    def _prior_rv(self, X, **kwargs):
        raise NotImplementedError

    def _predictive_rv(self, Xs, X, **kwargs):
        raise NotImplementedError



class GPLatent(GPBase):
    """ Where the GP f isnt integrated out, and is sampled explicitly
    """
    def __init__(self, cov_func):
        super(GPLatent, self).__init__(cov_func)

    def __call__(self, name, size, mean_func):
        return super(GPLatent, self).__call__(name, size, mean_func)

    def _build_predictive(self, X, f, Xs):
        Kxx = self.cov_func(X)
        Kxs = self.cov_func(X, Xs)
        Kss = self.cov_func(Xs)
        L = cholesky(stabilize(Kxx))
        A = solve_lower(L, Kxs)
        cov = Kss - tt.dot(tt.transpose(A), A)
        mu = self.mean_func(Xs) + tt.dot(tt.transpose(A), f.rotated)
        chol = cholesky(stabilize(cov))
        return mu, chol

    def _build_prior(self, X):
        mu = self.mean_func(X)
        chol = cholesky(stabilize(self.cov_func(X)))
        return mu, chol

    def _prior_rv(self, X):
        rotated = pm.Normal(self.name + "_rotated_", mu=0.0, sd=1.0, shape=self.size)
        mu, chol = self._build_prior(X)
        # docstring of mvnormal uses:
        #   tt.dot(chol, self.v.T).T
        f = pm.Deterministic(self.name, mu + tt.dot(chol, rotated))
        # attach a reference to the rotated random variable to
        #   the deterministic f on the way out
        f.rotated = rotated
        return f

    def _predictive_rv(self, X, f, Xs):
        mu, chol = self._build_predictive(X, f, Xs)
        return pm.MvNormal(self.name, mu=mu, chol=chol, shape=self.size)



class TProcess(GPLatent):
    """ StudentT process
    """
    def __init__(self, cov_func):
        super(GPLatentStudentT, self).__init__(cov_func)

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

    def _predictive_rv(self, X, f, Xs, nu):
        mu, chol = self._build_predictive(X, f, Xs)
        return pm.MvStudentT(self.name, nu, mu=mu, chol=chol, shape=self.size)



class GPMarginal(GPBase):

    def __init__(self, cov_func):
        super(GPMarginal, self).__init__(cov_func)

    def __call__(self, name, size, mean_func, include_noise=False):
        self.include_noise = include_noise
        return super(GPMarginal, self).__call__(name, size, mean_func)

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

    def _build_prior(self, X, cov_func_noise):
        mu = self.mean_func(X)
        Kxx = self.cov_func(X)
        Knx = cov_func_noise(X)
        cov = Kxx + Knx
        chol = cholesky(stabilize(cov))
        return mu, chol

    def _build_predictive(self, X, y, Xs, cov_func_noise):
        Kxx = self.cov_func(X)
        Kxs = self.cov_func(X, Xs)
        Kss = self.cov_func(Xs)
        Knx = cov_func_noise(X)
        r = y - self.mean_func(X)
        L = cholesky(stabilize(Kxx) + Knx)
        A = solve_lower(L, Kxs)
        v = solve_lower(L, r)
        mu = self.mean_func(Xs) + tt.dot(tt.transpose(A), v)
        if self.include_noise:
            cov = cov_func_noise(Xs) + Kss - tt.dot(tt.transpose(A), A)
        else:
            cov = stabilize(Kss) - tt.dot(tt.transpose(A), A)
        chol = cholesky(cov)
        return mu, chol

    def _prior_rv(self, X, y, sigma=None, cov_func_noise=None):
        cov_func_noise = self._to_noise_func(sigma, cov_func_noise)
        mu, chol = self._build_prior(X, cov_func_noise)
        return pm.MvNormal(self.name, mu=mu, chol=chol, shape=self.size, observed=y)

    def _predictive_rv(self, X, y, Xs, sigma=None, cov_func_noise=None):
        cov_func_noise = self._to_noise_func(sigma, cov_func_noise)
        mu, chol = self._build_predictive(X, y, Xs, cov_func_noise)
        return pm.MvNormal(self.name, mu=mu, chol=chol, shape=self.size)



class GPMarginalSparse(GPBase):
    """ FITC and VFE sparse approximations
    """
    def __init__(self, cov_func, approx=None):
        if approx is None:
            approx = "FITC"
        self.approx = approx
        super(GPMarginalSparse, self).__init__(cov_func)

    # overriding __add__, since whether its vfe or fitc determines its 'type'
    def __add__(self, other):
        if not (isinstance(self, type(other)) and self.approx == other.approx):
            raise ValueError("cant add different GP types")
        cov_func = self.cov_func + other.cov_func
        return type(self)(cov_func)

    def __call__(self, name, size, mean_func, include_noise=False):
        self.include_noise = include_noise
        return super(GPMarginalSparse, self).__call__(name, size, mean_func)

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

    def _build_prior_logp(self, X, Xu, y, sigma):
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

    def _build_predictive(self, X, Xu, y, Xs, sigma):
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
        if self.include_noise:
            cov = (self.cov_func(Xs) - tt.dot(tt.transpose(As), As) +
                   tt.dot(tt.transpose(C), C) + sigma2*tt.eye(Xs.shape[0]))
        else:
            cov = (self.cov_func(Xs) - tt.dot(tt.transpose(As), As) +
                   tt.dot(tt.transpose(C), C))
        return mean, stabilize(cov)

    def _prior_rv(self, X, Xu, y, sigma):
        logp = lambda y: self._build_prior_logp(X, Xu, y, sigma)
        return pm.DensityDist(self.name, logp, observed=y)

    def _predictive_rv(self, X, Xu, y, Xs, sigma):
        mu, chol = self._build_predictive(X, Xu, y, Xs, sigma)
        return pm.MvNormal(self.name, mu=mu, chol=chol, shape=self.size)
