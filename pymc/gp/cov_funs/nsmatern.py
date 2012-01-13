from . import isotropic_cov_funs
import numpy as np


__all__ = ['nsmatern','nsmatern_diag','default_h']


def default_h(x):
    return np.ones(x.shape[:-1])

def nsmatern(C,x,y,diff_degree,amp=1.,scale=1.,h=default_h,cmin=0,cmax=-1,symm=False):
    """

    A covariance function. Remember, broadcasting for covariance functions works
    differently than for numpy universal functions. C(x,y) returns a matrix, and
    C(x) returns a vector.

    :Parameters:

        - `amp`: The pointwise standard deviation of f.

        - `scale`: The factor by which to scale the distance between points.
                 Large value implies long-range correlation.

        - `diff_degree`: A function that takes arrays and returns the degree
                         of differentiability at each location.
    
        - `h`: A function that takes arrays and returns the relative amplitude
               at each location.

        - `x and y` are arrays of points in Euclidean coordinates
          formatted as follows:

          [[x_{0,0} ... x_{0,ndim}],
           [x_{1,0} ... x_{1,ndim}],
           ...
           [x_{N,0} ... x_{N,ndim}]]

        - `symm` indicates whether x and y are references to
          the same array.

        - `cmin' and `cmax' indicate which columns to compute.
          These are used for multithreaded evaluation.

    :Reference: Pintore and Holmes, 2010, "Spatially adaptive non-stationary covariance functions
                via spatially adaptive spectra". Journal of the American Statistical Association.
                Forthcoming.
    """
    ddx, ddy = diff_degree(x), diff_degree(y)
    hx, hy = h(x), h(y)
    
    # for rkbesl
    nmax = np.floor(max(np.max(ddx), np.max(ddy)))
    
    # Compute covariance for this bit
    isotropic_cov_funs.nsmatrn(C,ddx,ddy,hx,hy,nmax,cmin,cmax,symm=symm)

    return C

def nsmatern_diag(x,diff_degree, amp=1., scale=1.,h=default_h):
    return (h(x)*amp)**2
