# Copyright (c) Anand Patil, 2007

__docformat__ = 'reStructuredText'
__all__ = ['FullRankCovariance']


from numpy import *
from numpy.linalg import cholesky, LinAlgError
from .GPutils import regularize_array, trisolve
from .linalg_utils import dpotrf_wrap
from .Covariance import Covariance
from .incomplete_chol import ichol, ichol_continue

from pymc import six
xrange = six.moves.xrange


class FullRankCovariance(Covariance):

    """
    C=FullRankCovariance(eval_fun, **params)

    A GP covariance.

    All linear algebra done with dense BLAS, so attempts to invert/ factorize
    numerically singular covariance matrices will cause errors. On the other
    hand, computations will be faster than with Covariance for full-rank
    covariance matrices.

    :Arguments:

        -   `eval_fun`: A function that takes either a single value x or two values x and y,
            followed by an arbitrary number of keyword parameters. x and y will be of shape
            (n,n_dim), where n is any integer and n_dim is the dimensionality of the space, or
            shape (n). In the latter case n_dim should be assumed to be 1.

        -   `params`: Parameters to be passed to eval_fun.


    :SeeAlso: Mean, BasisCovariance, SeparableBasisCovariance, Realization, observe
    """

    def __init__(self, eval_fun, nugget=None, **params):

        self.ndim = None
        self.observed = False
        self.obs_mesh = None
        self.obs_V = None
        self.Uo = None
        self.obs_piv = None
        self.obs_len = None
        self.full_piv = None
        self.full_obs_mesh = None
        self.basiscov = False

        self.eval_fun = eval_fun
        self.params = params
        self.nugget = nugget

        # Sorry... the diagonal calls are done using f2py for speed.
        # def diag_cov_fun(xe):
        #     return self.eval_fun(xe,xe,**self.params)
        #
        # self.diag_cov_fun = diag_cov_fun

    def cholesky(self, x, observed=True, nugget=None, return_eval_also=False):
        """

        U = C.cholesky(x[, observed=True, nugget=None])

        Computes Cholesky factorization of self(x,x).

        :Arguments:

            -   `x`: The input array on which to evaluate the covariance.

            -   `observed`: If 'True', any observations are taken into account
                when computing the Cholesky factor. If not, the unobserved
                version of self is used.

            -   `nugget`: The 'nugget' parameter, which will essentially be
                added to the diagonal of C(x,x) before Cholesky factorizing.
        """

        # Number of points in x.
        N_new = x.shape[0]

        U = self.__call__(x, x, regularize=False, observed=observed)
        if return_eval_also:
            C_eval = U.copy('F')

        if nugget is not None:
            for i in xrange(N_new):
                U[i, i] += nugget[i]

        # print self.params, x.shape, observed, nugget

        info = dpotrf_wrap(U)
        if info > 0:
            raise LinAlgError(
                "Matrix does not appear to be positive definite by row %i. Consider another Covariance subclass, such as NearlyFullRankCovariance." %
                info)

        if return_eval_also:
            return U, C_eval
        else:
            return U

    def continue_cholesky(
            self, x, x_old, U_old, observed=True, nugget=None, return_eval_also=False):
        """

        U = C.continue_cholesky(x, x_old, U_old[, observed=True, nugget=None])

        Computes Cholesky factorization of self(z,z). Assumes the Cholesky
        factorization of self(x_old, x_old) has already been computed.

        :Arguments:

            -   `x`: The input array on which to evaluate the Cholesky factorization.

            -   `x_old`: The input array on which the Cholesky factorization has been
                computed.

            -   `U_old`: The Cholesky factorization of C(x_old, x_old).

            -   `observed`: If 'True', any observations are taken into account
                when computing the Cholesky factor. If not, the unobserved
                version of self is used.

            -   `nugget`: The 'nugget' parameter, which will essentially be
                added to the diagonal of C(x,x) before Cholesky factorizing.
        """

        # Concatenation of the old points and new points.
        xtot = vstack((x_old, x))

        # Number of old points.
        N_old = x_old.shape[0]

        # Number of new points.
        N_new = x.shape[0]

        U_new = self.__call__(x, x, regularize=False, observed=observed)

        # not really implemented yet.
        if nugget is not None:
            for i in xrange(N_new):
                U_new[i, i] += nugget[i]

        U = asmatrix(
            zeros((N_new + N_old,
                   N_old + N_new),
                  dtype=float,
                  order='F'))
        U[:N_old, :N_old] = U_old

        offdiag = self.__call__(
            x=x_old,
            y=x,
            observed=observed,
            regularize=False)
        trisolve(U_old, offdiag, uplo='U', transa='T', inplace=True)
        U[:N_old, N_old:] = offdiag

        U_new -= offdiag.T * offdiag
        if return_eval_also:
            C_eval = U_new.copy('F')

        info = dpotrf_wrap(U_new)
        if info > 0:
            raise LinAlgError(
                "Matrix does not appear to be positive definite by row %i. Consider another Covariance subclass, such as NearlyFullRankCovariance." %
                info)

        U[N_old:, N_old:] = U_new
        if return_eval_also:
            return U, U_new, C_eval
        else:
            return U

    def __call__(self, x, y=None, observed=True,
                 regularize=True, return_Uo_Cxo=False):
        out = Covariance.__call__(
            self,
            x,
            y,
            observed,
            regularize,
            return_Uo_Cxo=return_Uo_Cxo)

        if self.nugget is None:
            return out

        if return_Uo_Cxo:
            out, Uo_Cxo = out
        if x is y:
            for i in xrange(out.shape[0]):
                out[i, i] += self.nugget
        elif y is None:
            out += self.nugget
        if return_Uo_Cxo:
            return out, Uo_Cxo
        else:
            return out

    def observe(self, obs_mesh, obs_V, output_type='r'):
        """
        Observes self on obs_mesh with observation variance obs_V.
        Output_type controls the information returned:

        'r' : returns information needed by Realization objects.
        'o' : returns information needed by function observe.
        's' : returns information needed by the Gaussian process
              submodel.
        """

        # Number of spatial dimensions.
        ndim = obs_mesh.shape[1]

        if self.ndim is not None:
            if not ndim == self.ndim:
                raise ValueError(
                    "Dimension of observation mesh is not equal to dimension of base mesh.")
        else:
            self.ndim = ndim

        # print ndim

        # =====================================
        # = If self hasn't been observed yet: =
        # =====================================
        if not self.observed:

            # If self has not been observed, get the Cholesky factor of self(obs_mesh, obs_mesh)
            # and the side information and store it.

            # Number of observation points so far is 0.
            N_old = 0
            N_new = obs_mesh.shape[0]

            if output_type == 's':
                U, C_eval = self.cholesky(
                    obs_mesh, nugget=obs_V, observed=False, return_eval_also=True)
                U_new = U
            else:
                U = self.cholesky(obs_mesh, nugget=obs_V, observed=False)

            # Upper-triangular Cholesky factor of self(obs_mesh, obs_mesh)
            self.full_Uo = U
            self.Uo = U

            # Pivots.
            piv_new = arange(N_new)
            self.full_piv = piv_new
            self.obs_piv = piv_new

            # Remember full observation mesh.
            self.full_obs_mesh = obs_mesh

            # relevant slice is the positive-definite indices, which get into
            # obs_mesh_*. See documentation.
            relevant_slice = self.obs_piv

            self.obs_mesh = obs_mesh
            self.obs_V = obs_V
            self.obs_len = N_new

        # =======================================
        # = If self has been observed already:  =
        # =======================================
        else:

            # If self has been observed, get the Cholesky factor of the _full_ observation mesh (new
            # and old observations) using continue_cholesky, along with side
            # information, and store it.

            # Number of observations so far.
            N_old = self.full_obs_mesh.shape[0]

            # Number of new observations.
            N_new = obs_mesh.shape[0]

            # Call to self.continue_cholesky.
            if output_type == 's':
                U, U_new, C_eval = self.continue_cholesky(x=obs_mesh,
                                                          x_old=self.full_obs_mesh,
                                                          U_old=self.full_Uo,
                                                          observed=False,
                                                          nugget=obs_V,
                                                          return_eval_also=True)

            else:
                U = self.continue_cholesky(x=obs_mesh,
                                           x_old=self.full_obs_mesh,
                                           U_old=self.full_Uo,
                                           observed=False,
                                           nugget=obs_V)

            # Full Cholesky factor of self(obs_mesh, obs_mesh), where obs_mesh
            # is the combined observation mesh.
            self.full_Uo = U

            # Square upper-triangular Cholesky factor of self(obs_mesh_*,
            # obs_mesh_*). See documentation.
            self.Uo = self.full_Uo

            # Pivots.
            piv_new = arange(N_old + N_new)
            self.obs_piv = piv_new
            self.full_piv = piv_new

            # Concatenate old and new observation meshes.
            self.full_obs_mesh = vstack((self.full_obs_mesh, obs_mesh))
            self.obs_mesh = self.full_obs_mesh
            self.obs_V = hstack((self.obs_V, obs_V))

            # Length of obs_mesh_*.
            self.obs_len = N_old + N_new

        self.observed = True
        # Output expected by Realization
        if output_type == 'r':
            return slice(None, None, None), obs_mesh, self.full_Uo[
                N_old:N_new + N_old, N_old:N_new + N_old], self.full_Uo[:N_old, N_old:N_new + N_old]

        # Ouptut expected by observe
        if output_type == 'o':
            return slice(None, None, None), obs_mesh

        # Output expected by the GP submodel
        if output_type == 's':
            return U_new, C_eval, self.full_Uo[:N_old, N_old:N_new + N_old]
