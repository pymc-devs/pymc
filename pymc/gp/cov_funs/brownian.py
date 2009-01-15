import isotropic_cov_funs
import numpy as np
from isotropic_cov_funs import symmetrize, imul
from copy import copy
from pymc import get_threadpool_size, map_noreturn

__all__ = ['brownian']

def brownian_targ(C,x,y,h,amp,cmin, cmax,symm):
    # Compute covariance for this bit
    if h==.5:
        isotropic_cov_funs.brownian(C,x,y,cmin=0,cmax=-1,symm=symm)
    else:
        isotropic_cov_funs.frac_brownian(C,x,y,h,cmin=0,cmax=-1,symm=symm)

    imul(C, amp*amp, cmin=cmin, cmax=cmax, symm=symm)
    # Possibly symmetrize this bit
    if symm:
        symmetrize(C, cmin=cmin, cmax=cmax)


def brownian(x,y,amp=1.,scale=1.,origin=None,h=.5,symm=None):
    """
    brownian(x,y,amp=1., scale=1.,h=.5,origin=None)

    Fractional n-dimensional brownian motion. h=.5 corresponds to standard
    Brownian motion.

    A covariance function. Remember, broadcasting for covariance functions works
    differently than for numpy universal functions. C(x,y) returns a matrix, and
    C(x) returns a vector.

    :Parameters:

        - `amp`: The pointwise standard deviation of f.

        - `scale`: The factor by which to scale the distance between points.
                 Large value implies long-range correlation.

        - `h': The fractional parameter.


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

    :Reference: http://en.wikipedia.org/wiki/Fractional_brownian_motion
    """
    # Thanks to Anne Archibald for handythread.py, the model for the
    # multithreaded call.

    if h<0 or h>1:
        raise ValueError, 'Parameter h must be between 0 and 1.'

    if amp<0. or scale<0.:
        raise ValueError, 'The amp and scale parameters must be positive.'

    if symm is None:
        symm = (x is y)

    # Figure out how to divide job up between threads.
    nx = x.shape[0]
    ny = y.shape[0]
    n_threads = min(get_threadpool_size(), nx*ny / 10000)

    if n_threads > 1:
        if not symm:
            bounds = np.linspace(0,ny,n_threads+1)
        else:
            bounds = np.array(np.sqrt(np.linspace(0,ny*ny,n_threads+1)),dtype=int)

    # Allocate the matrix
    C = np.asmatrix(np.empty((nx,ny),dtype=float,order='F'))
    if origin is not None:
        x = x-origin
        y = y-origin
    x = x / float(scale)
    y = y / float(scale)

    if n_threads <= 1:
        brownian_targ(C,x,y,h,amp,0,-1,symm)
    else:
        thread_args=[(C,x,y,h,amp,bounds[i],bounds[i+1],symm) for i in xrange(n_threads)]
        map_noreturn(brownian_targ, thread_args)

    return C

weiner = brownian

if __name__ == '__main__':
    import numpy
    import pymc
    from pylab import *
    N = 100
    x,y=numpy.meshgrid(linspace(.01,1,N), linspace(.01,1,N))
    z = numpy.empty((N,N,2))
    z[:,:,0]=x
    z[:,:,1]=y
    C = pymc.gp.FullRankCovariance(brownian, amp=1., scale=1., h=.1)
    M = pymc.gp.Mean(lambda x: numpy.zeros(x.shape[:-1]))
    f = pymc.gp.Realization(M,C)
    imshow(f(z))
