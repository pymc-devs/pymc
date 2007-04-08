from scipy import special
from numpy import dot, sqrt, pi, array, ndarray, zeros, matrix
from cov_utils import regularize_array

def Matern(x,y,diff_degree,amp,scale):
    """
    C=Matern(x,y,diff_degree,amp,scale)
    
    The isotropic Matern covariance function.
    
    :Arguments:
        - diff_degree: The degree of differentiability.
        - amp: The pointwise standard deviation.
        - scale: The factor by which the x-axis should effectively be stretched
    """
    symm=(x is y)

    x=regularize_array(x)
    y=regularize_array(y)

    lenx = x.shape[0]
    leny = y.shape[0]
    
    C = zeros((lenx,leny),dtype=float)
    
    prefac_scale = 1. / 2. ** (diff_degree-1.) / special.gamma(diff_degree)
    t_scale = 1./scale


    for i in xrange(lenx):
        if symm:

            C[i,i] = 1.

            for j in range(i+1, len(y)):

                t = abs(y[j]-x[i]) * t_scale
                if t == 0.:
                    C[i,j] = 1.
                else:
                    C[i,j] = t**diff_degree * special.kv (diff_degree, t) * prefac_scale
                C[j,i] = C[i,j]

        else:

            for j in xrange(leny):
                t = abs(y[j]-x[i]) * t_scale
                if t == 0.:
                    C[i,j] = 1.
                else:
                    C[i,j] = t**diff_degree * special.kv (diff_degree, t) * prefac_scale
                
    
    C *= amp**2
    return C.view(matrix)
            
