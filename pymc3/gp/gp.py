import numpy as np
from scipy import stats
from tqdm import tqdm

from theano import printing
from theano.tensor.nlinalg import matrix_inverse
import theano.tensor as tt
from theano.tensor.var import TensorVariable
import theano

from .mean import Zero, Mean
from .cov import Covariance
from ..distributions import NoDistribution, Distribution, TensorType, Normal, MvNormal, Continuous, draw_values, generate_samples
from ..model import Model, modelcontext, Deterministic, ObservedRV, FreeRV, Factor
from ..vartypes import string_types

from ..distributions import transforms

__all__ = ['GP2', 'GPOriginal', 'GP', 'sample_gp']


class GP(Distribution):
    def __new__(cls, name, *args, **kwargs):
        try:
            model = Model.get_context()
        except TypeError:
            raise TypeError("No model on context stack, which is needed to "
                            "use the Normal('x', 0,1) syntax. "
                            "Add a 'with model:' block")

        if isinstance(name, string_types):
            total_size = kwargs.pop('total_size', None)

            distribution = cls.dist(*args, **kwargs)
            v = model.Var(name + "_rotated_", Normal.dist(mu=0.0, sd=1.0, shape=distribution.shape))
            f = tt.dot(distribution.chol, v)

            f.name = name
            f.dshape = tuple
            f.dsize = int(np.prod(distribution.shape))
            f.distribution = distribution
            f.distribution.rotated = v

            f.tag.test_value = np.ones(distribution.shape, distribution.dtype) * distribution.default()
            f.logp_elemwiset = lambda x: 0
            f.total_size = total_size
            f.model = model

            #incorporate_methods(source=distribution, destination=self,
            #                    methods=['random'],
            #                    wrapper=InstanceMethod)

            model.deterministics.append(f)
            return f
        else:
            raise TypeError("Name needs to be a string but got: {}".format(name))

    @property
    def init_value(self):
        """Convenience attribute to return tag.test_value"""
        return self.tag.test_value

    @classmethod
    def dist(cls, *args, **kwargs):
        dist = object.__new__(cls)
        dist._construct(*args, **kwargs)
        return dist

    def _construct(self, mean_func=None, cov_func=None, X=None, sigma2=None, dtype=None, *args, **kwargs):
        if mean_func is None:
            mean_func = Zero()
        else:
            if not isinstance(mean_func, Mean):
                raise ValueError('mean_func must be a subclass of Mean')

        if cov_func is None:
            raise ValueError('A covariance function must be specified for GP')
        if not isinstance(cov_func, Covariance):
            raise ValueError('cov_func must be a subclass of Covariance')

        if sigma2 is None:
            # set sigma2 to some small value to prevent numerical instability in Cholesky decomp.
            sigma2 = 1e-6
        else:
            pass

        self.X = tt.as_tensor_variable(X)
        self.n = tt.shape(X)[0]

        self.mean_func = mean_func
        self.cov_func = cov_func
        self.sigma2 = sigma2

        Kx = cov_func(X) + sigma2 * tt.eye(self.n)
        self.solve = tt.slinalg.Solve("lower_triangular", lower=True)
        self.Kx = Kx
        self.chol = tt.slinalg.cholesky(Kx)
        self.mu = tt.squeeze(mean_func(X))

        self.prior_sample = MvNormal.dist(chol=self.chol, mu=self.mu).random()
        super(GP, self).__init__(shape=np.shape(X)[0], dtype=theano.config.floatX,
                                 testval=None, defaults=["prior_sample"],
                                 *args, **kwargs)
    def logp(self, x):
        return 0

    def random(self, point=None, size=None, Z=None, **kwargs):
        # this version seems to pretty much repeat samples from trace
        mu, K = self.conditional(Z)
        return MvNormal.dist(mu=mu, cov=K).random(point, size, **kwargs)

    #def random(self, point=None, size=None, Z=None, **kwargs):
    #    v = self.rotated.random(point, size, **kwargs)
    #    chol = draw_values([self.chol], point)
    #    return np.dot(chol, v.T).T

    def conditional(self, Z=None):
        if Z is None:
            Z = tt.as_tensor_variable(self.X)
            Kxz = self.cov_func(self.X, Z)
            Kzz = self.cov_func(Z, Z)
            mu = tt.squeeze(self.mean_func(Z))
            #print(Z, Kxz, Kzz, mu, self.chol, self.sigma2, tt.shape(Z)[0], self.rotated)
        else:
            Kxz = self.Kx
            Kzz = self.Kx
            mu = self.mu
        A = self.solve(self.chol, Kxz)
        #print(A)
        K_post = Kzz - tt.dot(tt.transpose(A), A) + self.sigma2 * tt.eye(tt.shape(Z)[0])
        mu_post = mu + tt.squeeze(tt.transpose(tt.dot(tt.transpose(A), self.rotated)))
        return mu_post, K_post




def GP2(name="gp", mean_func=None, cov_func=None, X=None, sigma2=None, model=None, *args, **kwargs):
    if mean_func is None:
        mean_func = Zero()
    else:
        if not isinstance(mean_func, Mean):
            raise ValueError('mean_func must be a subclass of Mean')

    if cov_func is None:
        raise ValueError('A covariance function must be specified for GP')
    if not isinstance(cov_func, Covariance):
        raise ValueError('cov_func must be a subclass of Covariance')

    if sigma2 is None:
        # set sigma2 to some small value to prevent numerical instability in Cholesky decomp.
        use_conjugate = False
        sigma2 = 1e-6
    else:
        use_conjugate = True

    n = X.shape[0]
    kwargs.setdefault("shape", n)

    # put params inside args and kwargs because of Distribution call to __new__
    kwargs.setdefault("mean_func", mean_func)
    kwargs.setdefault("cov_func", cov_func)
    kwargs.setdefault("sigma2", sigma2)
    kwargs.setdefault("X", X)

    Kx = cov_func(X) + sigma2 * tt.eye(n)
    kwargs.setdefault("solve", tt.slinalg.Solve("lower_triangular", lower=True))
    kwargs.setdefault("Kx", Kx)
    kwargs.setdefault("chol", tt.slinalg.cholesky(Kx))
    kwargs.setdefault("mu", tt.squeeze(mean_func(X)))

    args = list(args)
    args.insert(0, name)

    class GPConjugate(MvNormal):
        def __init__(self, *args, **kwargs):
            mu = kwargs.pop("mu")
            chol = kwargs.pop("chol")
            super(GPConjugate, self).__init__(mu=mu, chol=chol)

            self.X = kwargs["X"]
            self.shape = kwargs["shape"]
            self.sigma2 = kwargs["sigma2"]
            self.solve = kwargs["solve"]

            self.cov_func = kwargs["cov_func"]
            self.Kx = kwargs["Kx"]
            self.mean_func = kwargs["mean_func"]

        def conditional(self, Z=None):
            if Z is None:
                Kxz = self.Kx
                Kzz = self.Kx
                mu = self.mu
            else:
                Kxz = self.cov_func(self.X, Z)
                Kzz = self.cov_func(Z, Z)
                mu = tt.squeeze(self.mean_func(Z))
            A = self.solve(tt.transpose(self.chol), self.solve(self.chol, y))
            v = self.solve(self.chol, Kxz)
            if post_pred:
                K_post = Kzz - tt.dot(tt.transpose(v), v) + self.sigma2 * tt.eye(tt.shape(Kzz)[0])
            else:
                K_post = Kzz - tt.dot(tt.transpose(v), v) + 1e-8 * tt.eye(tt.shape(Kzz)[0])
            mu_post = mu + tt.dot(tt.transpose(Kxz), A)
            return mu_post, K_post

    class GPFull(object):
        def __init__(self, *args, **kwargs):
            self._been_called = False

            self.name = args[0]

            self.X = kwargs["X"]
            self.shape = kwargs["shape"]
            self.sigma2 = kwargs["sigma2"]
            self.solve = kwargs["solve"]

            self.cov_func = kwargs["cov_func"]
            self.Kx = kwargs["Kx"]
            self.chol = kwargs["chol"]

            self.mean_func = kwargs["mean_func"]
            self.mu = kwargs["mu"]

        def __call__(self):
            if not self._been_called:
                self._been_called = True
            else:
                raise RuntimeError("Can only be called once")
            self.v = Normal(self.name + "_rotated_", mu=0.0, sd=1.0, shape=self.shape)
            f = Deterministic(self.name, tt.dot(self.chol, self.v))
            # attach gp class to deterministic f
            f.gp = self
            return f

        def conditional(self, Z=None):
            if Z is None:
                Kxz = self.Kx
                Kzz = self.Kx
                mu = self.mu
            else:
                Z = tt.as_tensor_variable(Z)
                Kxz = self.cov_func(self.X, Z)
                Kzz = self.cov_func(Z, Z)
                mu = tt.squeeze(self.mean_func(Z))
            A = self.solve(self.chol, Kxz)
            K_post = Kzz - tt.dot(tt.transpose(A), A) + self.sigma2 * tt.eye(tt.shape(Z)[0])
            mu_post = mu + tt.squeeze(tt.transpose(tt.dot(tt.transpose(A), self.v)))
            return mu_post, K_post

    if use_conjugate:
        gp = GPConjugate(*args, **kwargs)
        return gp
    else:
        gp = GPFull(*args, **kwargs)
        #f = gp()
        #f.gp = gp
        return gp()





class GPOriginal(Continuous):
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







