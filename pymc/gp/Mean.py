# Copyright (c) Anand Patil, 2007

__docformat__ = 'reStructuredText'

from numpy import *
from .GPutils import regularize_array, trisolve

__all__ = ['Mean', 'zero_fn']


class Mean(object):

    """
    M = Mean(eval_fun, **params)


    A Gaussian process mean.


    :Arguments:

        -   `eval_fun`: A function that takes an argument x of shape  (n,ndim), where n is
            any integer and ndim is the dimensionality of  the space, or shape (n). In the
            latter case ndim should be assumed to be 1..

        -   `params`: Parameters to be passed to eval_fun


    :SeeAlso: Covariance, Realization, Observe
    """

    def __init__(self, eval_fun, **params):

        self.ndim = None
        self.observed = False
        self.obs_mesh = None
        self.base_mesh = None
        self.obs_vals = None
        self.obs_V = None
        self.reg_mat = None
        self.obs_len = None
        self.mean_under = None
        self.Uo = None
        self.obs_piv = None

        self.eval_fun = eval_fun
        self.params = params

    def observe(self, C, obs_mesh_new, obs_vals_new, mean_under=None):
        """
        Synchronizes self's observation status with C's.
        Values of observation are given by obs_vals.

        obs_mesh_new and obs_vals_new should already have
        been sliced, as Covariance.observe(..., output_type='o') does.
        """

        self.C = C
        self.obs_mesh = C.obs_mesh
        self.obs_len = C.obs_len

        self.Uo = C.Uo

        # Evaluate the underlying mean function on the new observation mesh.
        if mean_under is None:
            mean_under_new = C._mean_under_new(self, obs_mesh_new)
        else:
            mean_under_new = mean_under

        # If self hasn't been observed yet:
        if not self.observed:

            self.dev = (obs_vals_new - mean_under_new)

            self.reg_mat = C._unobs_reg(self)

        # If self has been observed already:
        elif len(obs_vals_new) > 0:

            # Rank of old observations.
            m_old = len(self.dev)

            # Deviation of new observation from mean without regard to old
            # observations.
            dev_new = (obs_vals_new - mean_under_new)

            # Again, basis covariances get special treatment.
            self.reg_mat = C._obs_reg(self, dev_new, m_old)

            # Stack deviations of old and new observations from unobserved
            # mean.
            self.dev = hstack((self.dev, dev_new))
        self.observed = True

    def __call__(self, x, observed=True, regularize=True, Uo_Cxo=None):

        # Record original shape of x and regularize it.
        orig_shape = shape(x)
        if len(orig_shape) > 1:
            orig_shape = orig_shape[:-1]

        if regularize:
            x = regularize_array(x)

        ndimx = x.shape[-1]
        lenx = x.shape[0]

        # Safety.
        if self.ndim is not None:
            if not self.ndim == ndimx:
                raise ValueError(
                    "The number of spatial dimensions of x does not match the number of spatial dimensions of the Mean instance's base mesh.")

        # Evaluate the unobserved mean
        M = self.eval_fun(x, **self.params).squeeze()

        # Condition. Basis covariances get special treatment. See documentation
        # for formulas.
        if self.observed and observed:
            M = self.C._obs_eval(self, M, x, Uo_Cxo)

        return M.reshape(orig_shape)


def zero_fn(x):
    "A function mapping any argument to zero."
    return zeros(x.shape[:-1])
