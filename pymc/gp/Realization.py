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


# TODO: Make subclass FullRankRealization which takes the evaluation of its covariance
# on its input mesh as an argument. This can be passed in by GPNormal.
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

    def __init__(self, M, C, init_mesh = None, init_vals = None, check_repeats = True, regularize = True):

        # Make internal copies of M and C. Note that subsequent observations of M and C
        # will not affect self.
        M_internal = copy.copy(M)
        C_internal = copy.copy(C)
        M_internal.C = C_internal

        # If initial values were specified on a mesh:
        if init_mesh is not None:

            if regularize:
                init_mesh = regularize_array(init_mesh)
                init_vals = init_vals.ravel()

            # Observe internal M and C with self's value on init_mesh.
            observe(M_internal,
                    C_internal,
                    obs_mesh=init_mesh,
                    obs_vals=init_vals,
                    obs_V=zeros(len(init_vals),dtype=float),
                    lintrans=None,
                    cross_validate = False)

            # Store init_mesh.
            if check_repeats:
                self.x_sofar = init_mesh
                self.f_sofar = init_vals

        elif check_repeats:

            # Store init_mesh.
            self.x_sofar = None
            self.f_sofar = None

        self.check_repeats = check_repeats
        self.M_internal = M_internal
        self.C_internal = C_internal

    def __call__(self, x, regularize=True):

        # TODO: check repeats for basis covariances too.

        # TODO: Do the timing trick down through this method to see where the bottleneck is.

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

        # TODO: Optimization opportunity: Don't observe until next set of values are needed.
        # Get U straight from C_internal, M straight from M_internal, store them and draw values.
        # Next time values are needed, pass them back into C_internal and M_internal's observe
        # methods to make them faster.

        # First observe the internal covariance on x.
        relevant_slice, obs_mesh_new, U = self.C_internal.observe(x, zeros(x.shape[0]))

        # Then evaluate self's mean on x.
        M = self.M_internal(x, regularize=False)

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
        f = (self.M_internal(x, regularize=False) + dot(self.coef_vals, basis_x))

        return f
