# Copyright (c) Anand Patil, 2007

__docformat__ = 'reStructuredText'

__all__ = ['NearlyFullRankCovariance']

from numpy import *
from numpy.linalg import cholesky, LinAlgError
from .GPutils import regularize_array, trisolve
from .linalg_utils import diag_call
from .incomplete_chol import ichol_full
from .Covariance import Covariance

from pymc import six
xrange = six.moves.xrange


class NearlyFullRankCovariance(Covariance):

    """
    C=NearlyFullRankCovariance(eval_fun, relative_precision, **params)

    A GP covariance. Good for situations where self's evaluation
    on observation locations is nearly full-rank, but not quite. Evaluation
    of the matrix can be parallelized, but Cholesky decompositions are done
    using a serial incomplete algorithm.

    :Arguments:

        -   `eval_fun`: A function that takes either a single value x or two values x and y,
            followed by an arbitrary number of keyword parameters. x and y will be of shape
            (n,n_dim), where n is any integer and n_dim is the dimensionality of the space, or
            shape (n). In the latter case n_dim should be assumed to be 1.

        -   `params`: Parameters to be passed to eval_fun.

        -   `relative_precision`: See documentation.


    :SeeAlso: Mean, BasisCovariance, SeparableBasisCovariance, Realization, observe
    """

    def __init__(self, eval_fun, relative_precision=1.0E-15, **params):
        Covariance.__init__(
            self,
            eval_fun,
            relative_precision,
            rank_limit=0,
            **params)

    def cholesky(self, x, apply_pivot=True, observed=True,
                 nugget=None, regularize=True, rank_limit=0):
        """

        U = C.cholesky(x[, observed=True, nugget=None, rank_limit=0])

        {'pivots': piv, 'U': U} = \
        C.cholesky(x, apply_pivot = False[, observed=True, nugget=None])

        Computes incomplete Cholesky factorization of self(x,x).


        :Arguments:

            -   `x`: The input array on which to evaluate the covariance.

            -   `apply_pivot`: A flag. If it's set to 'True', it returns a
                matrix U (not necessarily triangular) such that U.T*U=C(x,x).
                If it's set to 'False', the return value is a dictionary.
                Item 'pivots' is a vector of pivots, and item 'U' is an
                upper-triangular matrix (not necessarily square) such that
                U[:,argsort(piv)].T * U[:,argsort(piv)] = C(x,x).

            -   `observed`: If 'True', any observations are taken into account
                when computing the Cholesky factor. If not, the unobserved
                version of self is used.

            -   `nugget`: The 'nugget' parameter, which will essentially be
                added to the diagonal of C(x,x) before Cholesky factorizing.

            -   `rank_limit`: If rank_limit > 0, the factor will have at most
                rank_limit rows.
        """
        if rank_limit > 0:
            raise ValueError(
                'NearlyFullRankCovariance does not accept a rank_limit argument. Use Covariance instead.')

        if regularize:
            x = regularize_array(x)

        # Number of points in x.
        N_new = x.shape[0]

        # Special fast version for single points.
        if N_new == 1:
            V = self.__call__(x, regularize=False, observed=observed)
            if nugget is not None:
                V += nugget
            U = asmatrix(sqrt(V))
            # print U
            if not apply_pivot:
                return {'pivots': array([0]), 'U': U}
            else:
                return U

        C = self.__call__(x, x, regularize=False, observed=observed)
        if nugget is not None:
            for i in xrange(N_new):
                C[i, i] += nugget[i]

        # =======================================
        # = Call to Fortran function ichol_full =
        # =======================================
        U, m, piv = ichol_full(c=C, reltol=self.relative_precision)

        U = asmatrix(U)

        # Arrange output matrix and return.
        if m < 0:
            raise ValueError(
                "Matrix does not appear to be positive semidefinite")

        if not apply_pivot:
            # Useful for self.observe and Realization.__call__. U is upper
            # triangular.
            U = U[:m, :]
            return {'pivots': piv, 'U': U}

        else:
            # Useful for users. U.T*U = C(x,x)
            return U[:m, argsort(piv)]

    def continue_cholesky(self, x, x_old, chol_dict_old, apply_pivot=True,
                          observed=True, nugget=None, regularize=True, assume_full_rank=False, rank_limit=0):
        """

        U = C.continue_cholesky(x, x_old, chol_dict_old[, observed=True, nugget=None,
                rank_limit=0])


        Returns {'pivots': piv, 'U': U}


        Computes incomplete Cholesky factorization of self(z,z). Here z is the
        concatenation of x and x_old. Assumes the Cholesky factorization of
        self(x_old, x_old) has already been computed.


        :Arguments:

            -   `x`: The input array on which to evaluate the Cholesky factorization.

            -   `x_old`: The input array on which the Cholesky factorization has been
                computed.

            -   `chol_dict_old`: A dictionary with kbasis_ys ['pivots', 'U']. Would be the
                output of either this method or C.cholesky().

            -   `apply_pivot`: A flag. If it's set to 'True', it returns a
                matrix U (not necessarily triangular) such that U.T*U=C(x,x).
                If it's set to 'False', the return value is a dictionary.
                Item 'pivots' is a vector of pivots, and item 'U' is an
                upper-triangular matrix (not necessarily square) such that
                U[:,argsort(piv)].T * U[:,argsort(piv)] = C(x,x).

            -   `observed`: If 'True', any observations are taken into account
                when computing the Cholesky factor. If not, the unobserved
                version of self is used.

            -   `nugget`: The 'nugget' parameter, which will essentially be
                added to the diagonal of C(x,x) before Cholesky factorizing.

            -   `rank_limit`: If rank_limit > 0, the factor will have at most
                rank_limit rows.
        """
        if regularize:
            x = regularize_array(x)

        if rank_limit > 0:
            raise ValueError(
                'NearlyFullRankCovariance does not accept a rank_limit argument. Use Covariance instead.')

        if rank_limit > 0:
            raise ValueError(
                'NearlyFullRankCovariance does not accept a rank_limit argument. Use Covariance instead.')

        # Concatenation of the old points and new points.
        xtot = vstack((x_old, x))

        # Extract information from chol_dict_old.
        U_old = chol_dict_old['U']
        m_old = U_old.shape[0]
        piv_old = chol_dict_old['pivots']

        # Number of old points.
        N_old = x_old.shape[0]

        # Number of new points.
        N_new = x.shape[0]

        # Compute off-diagonal part of Cholesky factor
        offdiag = self.__call__(
            x=x_old[
                piv_old[
                    :m_old],
                :],
            y=x,
            observed=observed,
            regularize=False)
        trisolve(U_old[:, :m_old], offdiag, uplo='U', transa='T', inplace=True)

        # Compute new diagonal part of Cholesky factor
        C_new = self.__call__(x=x, y=x, observed=observed, regularize=False)
        if nugget is not None:
            for i in xrange(N_new):
                C_new[i, i] += nugget[i]
        C_new -= offdiag.T * offdiag
        if not assume_full_rank:
            U_new, m_new, piv_new = ichol_full(
                c=C_new, reltol=self.relative_precision)
        else:
            U_new = cholesky(C_new).T
            m_new = U_new.shape[0]
            piv_new = arange(m_new)
        U_new = asmatrix(U_new[:m_new, :])
        U = asmatrix(
            zeros(
                (m_new +
                 m_old,
                 N_old +
                 N_new),
                dtype=float,
                order='F'))

        # Top portion of U
        U[:m_old, :m_old] = U_old[:, :m_old]
        U[:m_old, N_new + m_old:] = U_old[:, m_old:]
        offdiag = offdiag[:, piv_new]
        U[:m_old, m_old:N_new + m_old] = offdiag

        # Lower portion of U
        U[m_old:, m_old:m_old + N_new] = U_new
        if m_old < N_old and m_new > 0:
            offdiag_lower = self.__call__(x=x[piv_new[:m_new], :],
                                          y=x_old[piv_old[m_old:], :], observed=observed, regularize=False)
            offdiag_lower -= offdiag[:, :m_new].T * U[:m_old, m_old + N_new:]
            trisolve(
                U_new[
                    :,
                    :m_new],
                offdiag_lower,
                uplo='U',
                transa='T',
                inplace=True)
            U[m_old:, m_old + N_new:] = offdiag_lower

        # Rank and pivots
        m = m_old + m_new
        piv = hstack((piv_old[:m_old], piv_new + N_old, piv_old[m_old:]))

        # Arrange output matrix and return.
        if m < 0:
            raise ValueError('Matrix does not appear positive semidefinite.')

        if not apply_pivot:
            # Useful for self.observe. U is upper triangular.
            if assume_full_rank:
                return {'pivots': piv, 'U': U, 'C_eval': C_new, 'U_new': U_new}
            else:
                return {'pivots': piv, 'U': U}

        else:
            # Useful for the user. U.T * U = C(x,x).
            return U[:, argsort(piv)]


# def clean_array(A):
#     for i in xrange(A.shape[0]):
#         for j in xrange(A.shape[1]):
#             if abs(A[i,j])<1e-10:
#                 A[i,j]=0
#
# if __name__=='__main__':
#     from cov_funs import matern
#     C = NearlyFullRankCovariance(eval_fun = matern.euclidean, diff_degree = 1.4, amp = .4, scale = 1.)
#     first=C.cholesky([1,1,2],apply_pivot=False)
#     second=C.continue_cholesky([3,4],regularize_array([1,1,2]),first)
#     clean_array(first['U'])
#     clean_array(second)
#     print second.T*second
#     print
#     print C([1,1,2,3,4],[1,1,2,3,4])
