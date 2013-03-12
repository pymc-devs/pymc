from dist_math import * 
from continuous import *

@quickclass
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

    support = 'continuous'

    tau = tau_e * (1-k**2)

    def logp(x): 
        x_im1 = x[:-1]
        x_i = x[1:]
        boundary = Normal(0, tau).logp

        innov_like = Normal(k * x_im1, tau_e).logp(x_i) 
        return boundary(x[0]) + sum(innov_like) + boundary(x[-1])

    mode = 0.
    

    return locals()
