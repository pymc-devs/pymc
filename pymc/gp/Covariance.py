# Copyright (c) Anand Patil, 2007

__docformat__='reStructuredText'
__all__ = ['Covariance']


from numpy import *
from numpy.linalg import cholesky, LinAlgError
from .GPutils import regularize_array, trisolve, square_and_sum
from .linalg_utils import diag_call, dpotrf_wrap
from .incomplete_chol import ichol, ichol_continue

from pymc import six
xrange = six.moves.xrange

class Covariance(object):

    """
    C=Covariance(eval_fun, relative_precision, **params)

    A GP covariance.

    :Arguments:

        -   `eval_fun`: A function that takes either a single value x or two values x and y,
            followed by an arbitrary number of keyword parameters. x and y will be of shape
            (n,n_dim), where n is any integer and n_dim is the dimensionality of the space, or
            shape (n). In the latter case n_dim should be assumed to be 1.

        -   `params`: Parameters to be passed to eval_fun.

        -   `relative_precision`: See documentation.


    :SeeAlso: Mean, BasisCovariance, SeparableBasisCovariance, Realization, observe
    """


    def __init__(self, eval_fun, relative_precision = 1.0E-15, rank_limit=0, **params):

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
        self.relative_precision = relative_precision
        self.rank_limit = rank_limit


    def cholesky(self, x, apply_pivot = True, observed=True, nugget=None, regularize=True, rank_limit=0):
        """

        U = C.cholesky(x[, observed=True, nugget=None, rank_limit=00])


        {'pivots': piv, 'U': U} = \
        C.cholesky(x, apply_pivot = False[, observed=True, nugget=None])


        Computes incomplete Cholesky factorization of self(x,x), without
        actually evaluating the matrix first.

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

        if regularize:
            x=regularize_array(x)

        # Number of points in x.
        N_new = x.shape[0]

        # diagonal
        diag = self.__call__(x, y=None, regularize=False, observed=observed)

        if nugget is not None:
            diag += nugget.ravel()

        # Special fast version for single points.
        if N_new==1:
            U=asmatrix(sqrt(diag))
            # print U
            if not apply_pivot:
                return {'pivots': array([0]), 'U': U}
            else:
                return U

        # Create the diagonal and the get-row function differently depending on whether self
        # has been observed. If self hasn't been observed, send the calls straight to eval_fun
        # to skip the extra formatting.


        # get-row function
        # TODO: Forbid threading here due to callbacks.
        def rowfun(i,xpiv,rowvec):
            """
            A function that can be used to overwrite an input array with rows.
            """
            rowvec[i:]=self.__call__(x=xpiv[i-1,:].reshape((1,-1)), y=xpiv[i:,:], regularize=False, observed=observed)


        # ==================================
        # = Call to Fortran function ichol =
        # ==================================
        if rank_limit == 0:
            rank_limit = N_new
        U, m, piv = ichol(diag=diag, reltol=self.relative_precision, rowfun=rowfun, x=x, rl=min(rank_limit,N_new))
        U = asmatrix(U)


        # Arrange output matrix and return.
        if m<0:
            raise ValueError("Matrix does not appear to be positive semidefinite")

        if not apply_pivot:
            # Useful for self.observe and Realization.__call__. U is upper triangular.
            U = U[:m,:]
            return {'pivots': piv, 'U': U}
        else:
            # Useful for users. U.T*U = C(x,x)
            return U[:m,argsort(piv)]


    def continue_cholesky(self, x, x_old, chol_dict_old, apply_pivot = True, observed=True, nugget=None, regularize=True, assume_full_rank = False, rank_limit=0):
        """

        U = C.continue_cholesky(x, x_old, chol_dict_old[, observed=True, nugget=None, 
            rank_limit=0])


        returns {'pivots': piv, 'U': U}


        Computes incomplete Cholesky factorization of self(z,z), without
        actually evaluating the matrix first. Here z is the concatenation of x
        and x_old. Assumes the Cholesky factorization of self(x_old, x_old) has
        already been computed. 


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
            x=regularize_array(x)

        # Concatenation of the old points and new points.
        xtot = vstack((x_old,x))

        # Extract information from chol_dict_old.
        U_old = chol_dict_old['U']
        m_old = U_old.shape[0]
        piv_old = chol_dict_old['pivots']

        # Number of old points.
        N_old = x_old.shape[0]

        # Number of new points.
        N_new = x.shape[0]
        if rank_limit == 0:
            m_new_max = N_new
        else:
            m_new_max = min(N_new,max(0,rank_limit-m_old))


        # get-row function
        def rowfun(i,xpiv,rowvec):
            """
            A function that can be used to overwrite an input array with superdiagonal rows.
            """
            rowvec[i:] = self.__call__(x=xpiv[i-1,:].reshape(1,-1), y=xpiv[i:,:], regularize=False, observed = observed)


        # diagonal
        diag = self.__call__(x, y=None, regularize=False, observed = observed)


        # not really implemented yet.
        if nugget is not None:
            diag += nugget.ravel()


        # Arrange U for input to ichol. See documentation.
        U = asmatrix(zeros((m_new_max + m_old, N_old + N_new), dtype=float, order='F'))
        U[:m_old, :m_old] = U_old[:,:m_old]
        U[:m_old,N_new+m_old:] = U_old[:,m_old:]

        offdiag = self.__call__(x=x_old[piv_old[:m_old],:], y=x, observed=observed, regularize=False)
        trisolve(U_old[:,:m_old],offdiag,uplo='U',transa='T', inplace=True)
        U[:m_old, m_old:N_new+m_old] = offdiag


        # Initialize pivot vector:
        # [old_posdef_pivots  new_pivots  old_singular_pivots]
        #   - old_posdef_pivots are the indices of the rows that made it into the Cholesky factor so far.
        #   - old_singular_pivots are the indices of the rows that haven't made it into the Cholesky factor so far.
        #   - new_pivots are the indices of the rows that are going to be incorporated now.
        piv = zeros(N_new + N_old, dtype=int)
        piv[:m_old] = piv_old[:m_old]
        piv[N_new + m_old:] = piv_old[m_old:]
        piv[m_old:N_new + m_old] = arange(N_new)+N_old


        # ============================================
        # = Call to Fortran function ichol_continue. =
        # ============================================


        # Early return if rank is all used up.
        if m_new_max > 0:

            # ============================================
            # = Call to Fortran function ichol_continue. =
            # ============================================
            if not assume_full_rank:
                m, piv = ichol_continue(U, diag = diag, reltol = self.relative_precision, rowfun = rowfun, piv=piv, x=xtot[piv,:], mold=m_old)
            else:
                m = m_old + N_new
                C_eval = self.__call__(x,x,observed=True,regularize=False)
                U2 = cholesky(C_eval).T
                U[m_old:,m_old:N_new+m_old] = U2

                if m_old < N_old:
                    offdiag2 = self.__call__(x=x, y=x_old[piv_old[m_old:]], observed=observed, regularize=False)
                    trisolve(U2,offdiag2,uplo='U',transa='T',inplace=True)
                    U[m_old:,N_new+m_old:] = offdiag2

        else:
            m = m_old



        # Arrange output matrix and return.
        if m<0:
            raise ValueError('Matrix does not appear positive semidefinite.')

        if not apply_pivot:
            # Useful for self.observe. U is upper triangular.
            U = U[:m,:]
            if assume_full_rank:
                return {'pivots': piv, 'U': U, 'C_eval':C_eval, 'U_new': U2}
            else:
                return {'pivots': piv, 'U': U}

        else:
            # Useful for the user. U.T * U = C(x,x).
            return U[:m,argsort(piv)]


    def observe(self, obs_mesh, obs_V, output_type='r'):
        """
        Observes self on obs_mesh with observation variance obs_V.
        Output_type controls the information returned:
        
        'r' : returns information needed by Realization objects.
        'o' : returns information needed by function observe.
        's' : returns information needed by the Gaussian process
              submodel.
        """

        # print 'C.observe called'

        # Number of spatial dimensions.
        ndim = obs_mesh.shape[1]

        if self.ndim is not None:
            if not ndim==self.ndim:
                raise ValueError("Dimension of observation mesh is not equal to dimension of base mesh.")
        else:
            self.ndim = ndim

        # print ndim

        # =====================================
        # = If self hasn't been observed yet: =
        # =====================================
        if not self.observed:

            # If self has not been observed, get the Cholesky factor of self(obs_mesh, obs_mesh)
            # and the side information and store it.

            # Rank so far is 0.
            m_old = 0

            # Number of observation points so far is 0.
            N_old = 0

            if output_type != 's':
                obs_dict = self.cholesky(obs_mesh, apply_pivot = False, nugget = obs_V, regularize=False, rank_limit = self.rank_limit)
            else:
                C_eval = self.__call__(obs_mesh,obs_mesh,regularize=False)
                U = C_eval.copy('F')
                for i in xrange(U.shape[0]):
                    U[i,i] += obs_V[i]
                info = dpotrf_wrap(U)
                if info>0:
                    raise LinAlgError("Matrix does not appear to be positive definite by row %i. Could not observe with assume_full_rank=True." %info)
                obs_dict = {'U': U,'pivots': arange(U.shape[0]),'U_new':U,'C_eval':C_eval}
            obs_dict_new = obs_dict

            # Rank of self(obs_mesh, obs_mesh)
            m_new = obs_dict['U'].shape[0]

            # Upper-triangular Cholesky factor of self(obs_mesh, obs_mesh)
            self.full_Uo = obs_dict['U']
            # print (self.full_Uo[:,argsort(obs_dict['pivots'])].T*self.full_Uo[:,argsort(obs_dict['pivots'])] - self(obs_mesh,obs_mesh)).max()


            # Upper-triangular square Cholesky factor of self(obs_mesh_*, obs_mesh_*). See documentation.
            self.Uo = obs_dict['U'][:,:m_new]


            # Pivots.
            piv_new = obs_dict['pivots']
            self.full_piv = piv_new
            self.obs_piv = piv_new[:m_new]


            # Remember full observation mesh.
            self.full_obs_mesh = obs_mesh

            # relevant slice is the positive-definite indices, which get into obs_mesh_*. See documentation.
            relevant_slice = self.obs_piv

            # obs_mesh_new is obs_mesh_* from documentation.
            obs_mesh_new = obs_mesh[relevant_slice,:]


            self.obs_mesh = obs_mesh_new
            self.obs_V = obs_V[piv_new]
            self.obs_len = m_new


        # =======================================
        # = If self has been observed already:  =
        # =======================================
        else:

            # If self has been observed, get the Cholesky factor of the _full_ observation mesh (new
            # and old observations) using continue_cholesky, along with side information, and store it.

            # Extract information from self's existing attributes related to the observation mesh..
            obs_old, piv_old = self.Uo, self.obs_piv

            # Rank of self's evaluation on the observation mesh so far.
            m_old = len(self.obs_piv)

            # Number of observations so far.
            N_old = self.full_obs_mesh.shape[0]

            # Number of new observations.
            N_new = obs_mesh.shape[0]

            # Call to self.continue_cholesky.
            obs_dict_new = self.continue_cholesky(x=obs_mesh,
                                                x_old = self.full_obs_mesh,
                                                chol_dict_old = {'U': self.full_Uo, 'pivots': self.full_piv},
                                                apply_pivot = False,
                                                observed = False,
                                                regularize=False,
                                                nugget = obs_V,
                                                assume_full_rank = output_type=='s',
                                                rank_limit = self.rank_limit)

            if output_type=='s':
                C_eval = obs_dict_new['C_eval']

            # Full Cholesky factor of self(obs_mesh, obs_mesh), where obs_mesh is the combined observation mesh.
            self.full_Uo = obs_dict_new['U']

            # Rank of self(obs_mesh, obs_mesh)
            m_new = self.full_Uo.shape[0]

            # Square upper-triangular Cholesky factor of self(obs_mesh_*, obs_mesh_*). See documentation.
            self.Uo=self.full_Uo[:,:m_new]

            # Pivots.
            piv_new = obs_dict_new['pivots']
            self.obs_piv = piv_new[:m_new]
            self.full_piv = piv_new

            # Concatenate old and new observation meshes.
            self.full_obs_mesh = vstack((self.full_obs_mesh, obs_mesh))
            relevant_slice = piv_new[m_old:m_new] - N_old
            obs_mesh_new = obs_mesh[relevant_slice,:]

            # Remember obs_mesh_* and corresponding observation variances.
            self.obs_mesh = vstack((self.obs_mesh, obs_mesh[relevant_slice,:]))
            self.obs_V = hstack((self.obs_V, obs_V[relevant_slice]))

            # Length of obs_mesh_*.
            self.obs_len = m_new

        self.observed = True
        # Output expected by Realization
        if output_type == 'r':
            return relevant_slice, obs_mesh_new, self.full_Uo[m_old:,argsort(piv_new)[N_old:]], self.full_Uo[:m_old, argsort(piv_new)[N_old:]]
            
        # Ouptut expected by observe
        if output_type == 'o':
            return relevant_slice, obs_mesh_new
            
        # Output expected by the GP submodel
        if output_type=='s':
            return obs_dict_new['U_new'], obs_dict_new['C_eval'], self.full_Uo[:m_old, argsort(piv_new)[N_old:]]



    def __call__(self, x, y=None, observed=True, regularize=True, return_Uo_Cxo=False):

        if y is x:
            symm=True
        else:
            symm=False

        # Remember shape of x, and then 'regularize' it.
        orig_shape = shape(x)
        if len(orig_shape)>1:
            orig_shape = orig_shape[:-1]

        if regularize:
            x=regularize_array(x)

        ndimx = x.shape[-1]
        lenx = x.shape[0]

        if return_Uo_Cxo:
            Uo_Cxo = None

        # Safety
        if self.ndim is not None:
            if not self.ndim == ndimx:
                raise ValueError("The number of spatial dimensions of x, "+\
                                    ndimx.__str__()+\
                                    ", does not match the number of spatial dimensions of the Covariance instance's base mesh, "+\
                                    self.ndim.__str__()+".")


        # If there are observation points, prepare self(obs_mesh, x)
        # and chol(self(obs_mesh, obs_mesh)).T.I * self(obs_mesh, x)



        # ==========================================================
        # = If only one argument is provided, return the diagonal. =
        # ==========================================================
        if y is None:
            # Special fast-path for functions that have an 'amp' parameter
            if hasattr(self.eval_fun, 'diag_call'):
                V = self.eval_fun.diag_call(x, **self.params)
            # Otherwise, evaluate the diagonal in a loop.
            else:
                V=empty(lenx,dtype=float)
                for i in xrange(lenx):
                    this_x = x[i].reshape((1,-1))
                    V[i] = self.eval_fun(this_x, this_x, **self.params)
            if self.observed and observed:
                sqpart = empty(lenx,dtype=float)
                Cxo = self.eval_fun(self.obs_mesh, x, **self.params)
                Uo_Cxo = trisolve(self.Uo, Cxo, uplo='U', transa='T')
                square_and_sum(Uo_Cxo, sqpart)
                V -= sqpart

            if return_Uo_Cxo:    
                return V.reshape(orig_shape), Uo_Cxo
            else:
                return V.reshape(orig_shape)

        else:

            # ====================================================
            # = # If x and y are the same array, save some work: =
            # ====================================================
            if symm:
                C=self.eval_fun(x,x,symm=True,**self.params)
                # Update return value using observations.
                if self.observed and observed:
                    Cxo = self.eval_fun(self.obs_mesh, x, **self.params)
                    Uo_Cxo = trisolve(self.Uo, Cxo, uplo='U', transa='T')
                    C -= Uo_Cxo.T * Uo_Cxo

                if return_Uo_Cxo:
                    return C, Uo_Cxo
                else:
                    return C


            # ======================================
            # = # If x and y are different arrays: =
            # ======================================
            else:

                if regularize:
                    y=regularize_array(y)

                ndimy = y.shape[-1]
                leny = y.shape[0]

                if not ndimx==ndimy:
                    raise ValueError('The last dimension of x and y (the number of spatial dimensions) must be the same.')

                C = self.eval_fun(x,y,**self.params)

                # Update return value using observations.
                if self.observed and observed:

                    # If there are observation points, prepare self(obs_mesh, y)
                    # and chol(self(obs_mesh, obs_mesh)).T.I * self(obs_mesh, y)
                    Cyo = self.eval_fun(self.obs_mesh, y, **self.params)
                    Uo_Cyo = trisolve(self.Uo, Cyo,uplo='U', transa='T')
                    C -= Uo_Cxo.T * Uo_Cyo

                return C


    # Methods for Mean instances' benefit:

    def _unobs_reg(self, M):
        # reg_mat = chol(C(obs_mesh_*, obs_mesh_*)).T.I * M.dev
        return asmatrix(trisolve(self.Uo, M.dev.T, uplo='U',transa='T')).T

    def _obs_reg(self, M, dev_new, m_old):
        # reg_mat = chol(C(obs_mesh_*, obs_mesh_*)).T.I * M.dev
        reg_mat_new = -1.*dot(self.Uo[:m_old,m_old:].T , trisolve(self.Uo[:m_old,:m_old], M.dev, uplo='U', transa='T')).T
        trisolve(self.Uo[m_old:,m_old:].T, reg_mat_new, 'L', inplace=True)
        reg_mat_new += asmatrix(trisolve(self.Uo[m_old:,m_old:], dev_new.T, uplo='U', transa='T')).T
        return asmatrix(vstack((M.reg_mat,reg_mat_new)))
        
    def _obs_eval(self, M, M_out, x, Uo_Cxo=None):
        if Uo_Cxo is None:
            Uo_Cxo = trisolve(M.Uo, self(M.obs_mesh, x, observed = False), uplo='U', transa='T')
        M_out += dot(asarray(M.reg_mat).squeeze(),asarray(Uo_Cxo)).squeeze()
        return M_out

    def _mean_under_new(self, M, obs_mesh_new):
        return asarray(M.eval_fun(obs_mesh_new, **M.params)).ravel()
