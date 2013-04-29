# TODO Is this still relevant? If so, please document.
from dist_math import * 

def Normal_Summary(u, tau): 
    def like(mean, sd, n): 
        return switch( gt(tau,0),
                -tau/2 *(sd**2*(n-1) + n*(u - mean)**2) + n/2*log(.5*tau/pi),
                -inf)
    return like

