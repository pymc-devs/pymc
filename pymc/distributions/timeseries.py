from dist_math import *
from continuous import *

__all__ = ['AR1', 'GaussianRandomWalk']


@tensordist(continuous)
def AR1(k, tau_e):
    """
    Autoregressive process with 1 lag.

    Parameters
    ----------
    k : tensor
       effect of lagged value on current value
    tau_e : tensor
       precision for innovations
    """

    tau = tau_e * (1 - k ** 2)

    def logp(x):
        x_im1 = x[:-1]
        x_i = x[1:]
        boundary = Normal.dist(0, tau).logp

        innov_like = Normal.dist(k * x_im1, tau_e).logp(x_i)
        return boundary(x[0]) + sum(innov_like) + boundary(x[-1])

    mode = 0.

    return locals()


@tensordist(continuous)
def GaussianRandomWalk(tau =None, init=Flat.dist(), sd=None):
    """
    Random Walk with Normal innovations

    Parameters
    ----------
    tau : tensor
        tau > 0, innovation precision
    init : distribution
        distribution for initial value (Defaults to Flat())
    """

    def logp(x):
        x_im1 = x[:-1]
        x_i = x[1:]

        innov_like = Normal.dist(x_im1, tau, sd=sd).logp(x_i)
        return init.logp(x[0]) + sum(innov_like)

    mean = 0.

    return locals()
