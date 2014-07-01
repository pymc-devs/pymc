# TODO: Make a better class for bases. Should be either a multidimensional generator or should output
# a multidimensional ndarray, and dots into the basis shouldn't require it to be a matrix.
# Note tensordot can do the dot products you need.

# Make an object-valued ndarray whose elements are index tuples.
# Make it a matrix whose elements are what you want, and then reshape it multidimensional, and see where the indices
# have to end up.

# Copyright (c) Anand Patil, 2007

__docformat__ = 'reStructuredText'

__all__ = ['BasisCovariance', 'SeparableBasisCovariance']

from numpy import *
from numpy.linalg import eigh, solve, cholesky, LinAlgError
from .GPutils import regularize_array, trisolve
from .linalg_utils import basis_diag_call
from .incomplete_chol import ichol_basis, ichol_full
from .Covariance import Covariance

from pymc import six
xrange = six.moves.xrange


class BasisCovariance(Covariance):

    """
    C=BasisCovariance(basis, coef_cov, relative_precision, **params)

    Realizations generated using such covariances will be of the form:
        R(x) = \sum basis(x).ravel()[i] * c[i],
        c ~ N(mu, coef_cov)

    :Arguments:

        -   `basis`:    An n-dimensional array of functions. Each should take arguments of shape (m, n),
                        where m may be any integer, followed by arbitrary keyword parameters.

        -   `coef_cov`: An array of shape basis.shape or basis.shape*2 (tuple multiplication). If
                        shape=basis.shape, the coefficients are assumed to be independent. Otherwise
                        coef_cov[i[1],...,i[n], j[1],...,j[n]] gives the prior covariance of coefficients
                        i[1],...,i[n] and j[1],...,j[n].

        -   `params`:   Parameters to be passed to basis.

        -   `relative_precision`: See documentation.


    :SeeAlso: Mean, Realization, Covariance, SeparableBasisCovariance, observe
    """

    def __init__(self, basis, coef_cov,
                 relative_precision=1.0E-15, **params):

        self.observed = False
        self.obs_mesh = None
        self.obs_V = None
        self.Uo = None
        self.obs_piv = None
        self.obs_len = None
        self.full_piv = None
        self.full_obs_mesh = None
        self.basiscov = True

        self.basis = ravel(basis)
        self.shape = self.get_shape_from_basis(basis)
        self.n = prod(self.shape)
        self.ndim = len(self.shape)

        self.relative_precision = relative_precision
        self.params = params

        if coef_cov.shape == self.shape:
            self.coef_cov = asmatrix(diag(coef_cov.ravel()))
        elif coef_cov.shape == self.shape * 2:
            self.coef_cov = asmatrix(coef_cov.reshape((self.n, self.n)))
        else:
            raise ValueError(
                "Covariance tensor's shape must be basis.shape or basis.shape*2 (using tuple multiplication).")

        # Cholesky factor the covariance matrix of the coefficients.
        U, m, piv = ichol_full(c=self.coef_cov, reltol=relative_precision)
        self.coef_U = asmatrix(U[:m, argsort(piv)])

        # Rank of the coefficient covariance.
        self.unobs_m = m
        self.m = m

        # Record the unobserved Cholesky factor of the coefficient covariance.
        self.unobs_coef_U = self.coef_U.copy()
        self.observed = False

    def get_shape_from_basis(self, basis):
        return shape(basis)

    def eval_basis(self, x, regularize=True):
        """
        basis_mat = C.eval_basis(x)

        Evaluates self's basis functions on x and returns them stacked
        in a matrix. basis_mat[i,j] gives basis function i evaluated at
        x[j,:].
        """
        if regularize:
            x = regularize_array(x)

        out = zeros((self.n, x.shape[0]), dtype=float, order='F')

        for i in xrange(self.n):
            out[i] = self.basis[i](x, **self.params)

        return out

    def cholesky(self, x, apply_pivot=True,
                 observed=True, nugget=None, regularize=True):
        __doc__ = Covariance.cholesky.__doc__

        if regularize:
            x = regularize_array(x)

        # The pivots are just 1:N, N being the shape of x.
        piv_return = arange(x.shape[1], dtype=int)

        # The rank of the Cholesky factor is just the rank of the
        # coefficient covariance matrix.
        if observed:
            coef_U = self.coef_U
            m = self.m
        else:
            coef_U = self.unobs_coef_U
            m = self.unobs_m

        # The Cholesky factor is the Cholesky factor of the basis times
        # the basis evaluated on x. This isn't triangular, but it does have
        # the property U.T*U = self(x,x).
        U_return = asmatrix(self.eval_basis(x, regularize=False))
        U_return = coef_U * U_return

        if apply_pivot:
            # Good for users.
            return U_return
        else:
            # Good for self.observe.
            return {'piv': piv_return, 'U': U_return}

    def continue_cholesky(self, x, x_old, chol_dict_old,
                          apply_pivot=True, observed=True, nugget=None):
        __doc__ = Covariance.continue_cholesky.__doc__

        # Stack the old and new x's.
        xtot = vstack((x_old, x))

        # Extract information about the Cholesky factor of self(x_old).
        U_old = chol_dict_old['U']
        m_old = U_old.shape[0]

        # The pivots will just be 1:N.
        piv_return = arange(xtot.shape[1], dtype=int)

        # The rank of the Cholesky factor is the rank of the Cholesky factor of the
        # coefficient covariance.
        if observed:
            coef_U = self.coef_U
            m = self.m
        else:
            coef_U = self.unobs_coef_U
            m = self.unobs_m

        # The number of old and new observations.
        N_old = x_old.shape[0]
        N_new = x.shape[0]

        # The Cholesky factor is the Cholesky factor of the coefficient covariance
        # times the basis function.
        U_new = asmatrix(self.eval_basis(x, regularize=False))
        U_new = coef_U * U_new

        U_return = hstack((U_old, U_new))

        if apply_pivot:
            # Good for users.
            return U_return
        else:
            # Good for self.observe.
            return {'piv': piv_return, 'U': U_return}

    def observe(self, obs_mesh, obs_V, output_type='o'):
        __doc__ = Covariance.observe.__doc__

        ndim = obs_mesh.shape[1]
        nobs = obs_mesh.shape[0]

        if self.ndim is not None:
            if not ndim == self.ndim:
                raise ValueError(
                    "Dimension of observation mesh is not equal to dimension of base mesh.")
        else:
            self.ndim = ndim

        # bases_o is the basis evaluated on the observation mesh.
        basis_o = asmatrix(self.eval_basis(obs_mesh, regularize=False))

        # chol(basis_o.T * coef_cov * basis_o)
        chol_inner = self.coef_U * basis_o

        if output_type == 's':
            C_eval = dot(chol_inner.T, chol_inner).copy('F')
            U_eval = linalg.cholesky(C_eval).T.copy('F')
            U = U_eval
            m = U_eval.shape[0]
            piv = arange(m)
        else:
            # chol(basis_o.T * coef_cov * basis_o + diag(obs_V)). Really should do this as low-rank update of covariance
            # of V.

            U, piv, m = ichol_basis(
                basis=chol_inner, nug=obs_V, reltol=self.relative_precision)

        U = asmatrix(U)
        piv_new = piv[:m]
        self.obs_piv = piv_new
        obs_mesh_new = obs_mesh[piv_new, :]

        self.Uo = U[:m, :m]

        # chol(basis_o.T * coef_cov * basis_o + diag(obs_V)) ^ -T * basis_o.T
        self.Uo_cov = trisolve(
            self.Uo,
            basis_o[:,
                    piv[:m]].T,
            uplo='U',
            transa='T')

        # chol(basis_o.T * coef_cov * basis_o + diag(obs_V)).T.I * basis_o.T *
        # coef_cov
        self.Uo_cov = self.Uo_cov * self.coef_cov

        # coef_cov = coef_cov - coef_cov * basis_o * (basis_o.T * coef_cov *
        # basis_o + diag(obs_V)).I * basis_o.T * coef_cov
        self.coef_cov = self.coef_cov - self.Uo_cov.T * self.Uo_cov

        # coef_U = chol(coef_cov)
        U, m, piv = ichol_full(c=self.coef_cov, reltol=self.relative_precision)
        U = asmatrix(U)
        self.coef_U = U[:m, argsort(piv)]
        self.m = m

        if output_type == 'o':
            return piv_new, obs_mesh_new

        if output_type == 's':
            return U_eval, C_eval, basis_o

        raise ValueError('Output type not recognized.')

    def __call__(self, x, y=None, observed=True,
                 regularize=True, return_Uo_Cxo=False):

        # Record the initial shape of x and regularize it.
        orig_shape = shape(x)
        if len(orig_shape) > 1:
            orig_shape = orig_shape[:-1]

        if regularize:
            x = regularize_array(x)

        ndimx = x.shape[-1]
        lenx = x.shape[0]

        # Get the correct version of the Cholesky factor of the coefficient
        # covariance.
        if observed:
            coef_U = self.coef_U
        else:
            coef_U = self.unobs_coef_U

        # Safety.
        if self.ndim is not None:
            if not self.ndim == ndimx:
                raise ValueError(
                    "The number of spatial dimensions of x does not match the number of spatial dimensions of the Covariance instance's base mesh.")

        # Evaluate the Cholesky factor of self's evaluation on x.
        # Will be observed or not depending on which version of coef_U
        # is used.
        basis_x_ = self.eval_basis(x, regularize=False)
        basis_x = coef_U * basis_x_

        # ==========================================================
        # = If only one argument is provided, return the diagonal: =
        # ==========================================================
        if y is None:
            # Diagonal calls done in Fortran for speed.
            V = basis_diag_call(basis_x)
            if return_Uo_Cxo:
                return V.reshape(orig_shape), basis_x_
            else:
                return V.reshape(orig_shape)

        # ===========================================================
        # = If the same argument is provided twice, save some work: =
        # ===========================================================
        if y is x:
            if return_Uo_Cxo:
                return basis_x.T * basis_x, basis_x_
            else:
                return basis_x

        # =========================================
        # = If y and x are different, do needful: =
        # =========================================
        else:

            # Regularize y and record its original shape.
            if regularize:
                y = regularize_array(y)

            ndimy = y.shape[-1]
            leny = y.shape[0]

            if not ndimx == ndimy:
                raise ValueError(
                    'The last dimension of x and y (the number of spatial dimensions) must be the same.')

            # Evaluate the Cholesky factor of self's evaluation on y.
            # Will be observed or not depending on which version of coef_U
            # is used.
            basis_y = self.eval_basis(y, regularize=False)
            basis_y = coef_U * basis_y

            return basis_x.T * basis_y

    # Helper methods for Mean instances.
    def _unobs_reg(self, M):
        # reg_mat = chol(self.basis_o.T * self.coef_cov * self.basis_o + diag(obs_V)).T.I * self.basis_o.T * self.coef_cov *
        # chol(self(obs_mesh_*, obs_mesh_*)).T.I * M.dev
        return self.Uo_cov.T * \
            asmatrix(trisolve(self.Uo, M.dev, uplo='U', transa='T')).T

    def _obs_reg(self, M, dev_new, m_old):
        # reg_mat = chol(self.basis_o.T * self.coef_cov * self.basis_o + diag(obs_V)).T.I * self.basis_o.T * self.coef_cov *
        # chol(self(obs_mesh_*, obs_mesh_*)).T.I * M.dev
        M.reg_mat = M.reg_mat + self.Uo_cov.T * asmatrix(
            trisolve(self.Uo, dev_new, uplo='U', transa='T')).T
        return M.reg_mat

    def _obs_eval(self, M, M_out, x, Uo_Cxo=None):
        basis_x = Uo_Cxo if Uo_Cxo is not None else self.eval_basis(
            x, regularize=False)
        M_out += asarray(dot(basis_x.T, M.reg_mat)).squeeze()
        return M_out

    def _mean_under_new(self, M, obs_mesh_new):
        if not M.observed:
            return asarray(M.eval_fun(obs_mesh_new, **M.params)).ravel()
        else:
            return M.__call__(obs_mesh_new, regularize=False)


class SeparableBasisCovariance(BasisCovariance):

    """
    C=SeparableBasisCovariance(basis, coef_cov, relative_precision, **params)

    Realizations generated using such covariances will be of the form:
        R(x) = \sum_{i[i],...,i[n]} basis[1][i[i]](x) * ... * basis[n][i[n]](x) * c[i[1],...,i[n]]
        c ~ N(mu, coef_cov)

    :Arguments:

        -   `basis`: An n-dimensional array of functions. Each should take an argument x
            of shape  (n,ndim), where n is any integer and ndim is the dimensionality of
            the space, or shape (n). In the latter case ndim should be assumed to be 1.

        -   `coef_cov`: An array of shape (i[1],...,i[n]) or (i[1],...,i[n])*2 (tuple multiplication). If
            shape=basis.shape, the coefficients are assumed to be independent. Otherwise
            coef_cov[i[1],...,i[n], j[1],...,j[n]] gives the prior covariance of coefficients
            i[1],...,i[n] and j[1],...,j[n].

        -   `params`: Parameters to be passed to basis.

        -   `relative_precision`: See documentation.


    :SeeAlso: Mean, Realization, Covariance, SeparableBasisCovariance, observe
    """

    def __init__(self, basis, coef_cov,
                 relative_precision=1.0E-15, **params):
        BasisCovariance.__init__(
            self,
            basis,
            coef_cov,
            relative_precision,
            **params)
        self.basis = basis
        self.n_per_dim = []
        for i in xrange(self.ndim):
            self.n_per_dim.append(len(self.basis[i]))

    def get_shape_from_basis(self, basis):
        return tuple([len(dim_basis) for dim_basis in basis])

    def eval_basis(self, x, regularize=True):
        """
        basis_mat = C.eval_basis(x)

        Evaluates self's basis functions on x and returns them stacked
        in a matrix. basis_mat[i,j] gives basis function i (formed by
        multiplying basis functions) evaluated at x[j,:].
        """
        # Make object of same shape as self.basis, fill in with evals of each individual basis factor.
        # Make object of same shape as diag(coef_cov), fill in with products of those evals.
        # Reshape and return.
        if regularize:
            x = regularize_array(x)

        out = zeros(self.shape + (x.shape[0],), dtype=float, order='F')

        # Evaluate the basis factors
        basis_factors = []
        for i in xrange(self.ndim):
            basis_factors.append([])
            for j in xrange(self.n_per_dim[i]):
                basis_factors[i].append(self.basis[i][j](x, **self.params))

        out = ones((self.n, x.shape[0]), dtype=float)
        out_reshaped = out.reshape(self.shape + (x.shape[0],))

        for ind in ndindex(self.shape):
            for dim in xrange(self.ndim):
                out_reshaped[ind] *= basis_factors[dim][ind[dim]]

        return out
