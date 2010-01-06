# Copyright (c) Anand Patil, 2007

__docformat__='reStructuredText'
__all__ = ['Realization', 'StandardRealization', 'BasisRealization']


from numpy import *
from numpy.random import normal
from numpy.linalg import cholesky, eigh, solve
from Covariance import Covariance
from BasisCovariance import BasisCovariance
from Mean import Mean
from GPutils import observe, trisolve, regularize_array, caching_call
from linalg_utils import check_repeats, remove_duplicates
import copy


def Realization(M, C, *args, **kwargs):
    """
    f = Realization(M, C[, init_mesh, init_vals, check_repeats = True, regularize = True])


    Returns a realization from a Gaussian process.


    :Arguments:

        -   `M`: A Gaussian process mean function.

        -   `C`: A Covariance instance.

        -   `init_mesh`: An optional ndarray giving mesh at which f's initial value will be specified.

        -   `init_vals`: An ndarray giving the value of f over init_mesh.

        -   `regularize`: If init_mesh is not shaped as (n, ndim), where ndim is the dimension of
            the space, regularize should be True.

        -   `check_repeats: Determines whether calls to the GP realization will be checked against
            previous calls before evaluation.

    :SeeAlso: Mean, Covariance, BasisCovariance, observe, GP
    """
    if isinstance(C, BasisCovariance):
        return BasisRealization(M, C, *args, **kwargs)
    else:
        return StandardRealization(M, C, *args, **kwargs)

class StandardRealization(object):
    """
    f = Realization(M, C[, init_mesh, init_vals, check_repeats = True, regularize = True])


    Returns a realization from a Gaussian process.


    :Arguments:

        -   `M`: A Gaussian process mean function.

        -   `C`: A Covariance instance.

        -   `init_mesh`: An optional ndarray giving mesh at which f's initial value
            will be specified.

        -   `init_vals`: An ndarray giving the value of f over init_mesh.

        -   `regularize`: If init_mesh is not shaped as (n, ndim), where ndim is the dimension
            of the space, regularize should be True.

        -   `check_repeats: Determines whether calls to the GP realization will be checked against
        previous calls before evaluation.

    :SeeAlso: Mean, Covariance, BasisCovariance, observe, GP
    """

    # pickle support
    def __getstate__(self):
        return (self.M, self.C, self.x_sofar, self.f_sofar, self.check_repeats, False)

    def __setstate__(self, state):
        self.__init__(*state)

    def __init__(self, M, C, init_mesh = None, init_vals = None, check_repeats = True, regularize = True):

        self.M = M
        self.C = C

        # Make internal copies of M and C. Note that subsequent observations of M and C
        # will not affect self.
        M_internal = copy.copy(M)
        C_internal = copy.copy(C)
        M_internal.C = C_internal

        self.init_mesh = init_mesh
        self.init_vals = init_vals
        self.check_repeats = check_repeats
        self.regularize = regularize
        self.need_init_obs = True

        self.M_internal = M_internal
        self.C_internal = C_internal
        
        self.x_sofar = None
        self.f_sofar = None

    def _init_obs(self):
        # If initial values were specified on a mesh:
        if self.init_mesh is not None:

            if self.regularize:
                self.init_mesh = regularize_array(self.init_mesh)
                self.init_vals = self.init_vals.ravel()

            # Observe internal M and C with self's value on init_mesh.
            observe(self.M_internal,
                    self.C_internal,
                    obs_mesh=self.init_mesh,
                    obs_vals=self.init_vals,
                    obs_V=zeros(len(self.init_vals),dtype=float),
                    lintrans=None,
                    cross_validate = False)

            # Store init_mesh.
            if self.check_repeats:
                self.x_sofar = self.init_mesh
                self.f_sofar = self.init_vals
        
        self.need_init_obs = False
        

    def __call__(self, x, regularize=True):
        # TODO: check repeats for basis covariances too.
        
        # If initial values were passed in, observe on them.
        if self.need_init_obs:
            self._init_obs()

        # Record original shape of x and regularize it.
        orig_shape = shape(x)

        if len(orig_shape)>1:
            orig_shape = orig_shape[:-1]

        if regularize:
            if any(isnan(x)):
                raise ValueError, 'Input argument to Realization contains NaNs.'
            x = regularize_array(x)

        if x is self.x_sofar:
            return self.f_sofar

        if self.check_repeats:
            # use caching_call to save duplicate calls.
            f, self.x_sofar, self.f_sofar = caching_call(self.draw_vals, x, self.x_sofar, self.f_sofar)
        else:
            # Call to self.draw_vals.
            f = self.draw_vals(x)

        if regularize:
            return f.reshape(orig_shape)
        else:
            return f

    def draw_vals(self, x):

        # First observe the internal covariance on x.
        relevant_slice, obs_mesh_new, U, Uo_Cxo = self.C_internal.observe(x, zeros(x.shape[0]), output_type='r')

        # Then evaluate self's mean on x.
        M = self.M_internal(x, regularize=False, Uo_Cxo=Uo_Cxo)

        # Then draw new values for self(x).
        q = dot(U.T , normal(size = U.shape[0]))
        f = asarray((M.T+q)).ravel()

        # Then observe self's mean using the new values.
        self.M_internal.observe(self.C_internal, obs_mesh_new, f[relevant_slice])

        return f


class BasisRealization(StandardRealization):
    """
    f = BasisRealization(M, C[, init_mesh, init_vals, check_repeats = True, regularize = True])


    Returns a realization from a Gaussian process.


    :Arguments:

        -   `M`: A Gaussian process mean function.

        -   `C`: A BasisCovariance instance.

        -   `init_mesh`: An optional ndarray giving mesh at which f's initial value will be specified.

        -   `init_vals`: An ndarray giving the value of f over init_mesh.

        -   `regularize`: If init_mesh is not shaped as (n, ndim), where ndim is the dimension of the
            space, regularize should be True.

        -   `check_repeats: Determines whether calls to the GP realization will be checked against
            previous calls before evaluation.

    :SeeAlso: Mean, Covariance, BasisCovariance, observe, GP
    """

    def __init__(self, M, C, init_mesh = None, init_vals = None, regularize = True):

        StandardRealization.__init__(self, M, C, init_mesh, init_vals, False, regularize)
        self.coef_vals = asarray(dot(self.C_internal.coef_U.T,normal(size=self.C_internal.m)))
        
    def draw_vals(self, x):

        # If C is a BasisCovariance, just evaluate the basis over x, multiply by self's
        # values for the coefficients and add the mean.

        # TODO: For optimization, you could try a BasisMean class.
        # However, I don't think it would be very much faster.
        # You'd only save an evaluation of the underlying mean and
        # a vector addition.

        basis_x = self.C_internal.eval_basis(x)
        f = (self.M_internal(x, regularize=False, Uo_Cxo=basis_x) + dot(self.coef_vals, basis_x))

        return f
