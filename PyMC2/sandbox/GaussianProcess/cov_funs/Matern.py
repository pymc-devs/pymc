from scipy import special
from numpy import dot, sqrt, pi, array, ndarray, zeros, matrix

def Matern(x,y,phi,nu,alpha):
    """
    C=Matern(x,y,nu,alpha)
    
    The isotropic Matern covariance function, as parametrized by Stein.
    The pointwise variance of draws is:
    sqrt(pi) * gamma(nu) / gamma(nu + .5) / alpha ** (2. * nu)    
    """
    C = zeros((len(x),len(x)),dtype=float)
    symm=(x is y)
    
    prefac = sqrt(pi) * special.gamma(nu) / special.gamma(nu+.5) / alpha ** (2. * nu)
    prefac_scale = 1. / 2. ** (nu-1.) / special.gamma(nu)


    for i in xrange(len(x)):
        if symm:

            C[i,i] = prefac / prefac_scale

            for j in range(i+1, len(y)):

                t = dot(y[j]-x[i], y[j]-x[i]);
                if t == 0.:
                    C[i,j] = prefac / prefac_scale
                else:
                    C[i,j] = prefac * (alpha * t)**nu * special.kv (nu, alpha * t)
                C[j,i] = C[i,j]

        else:

            for j in xrange(len(y)):
                t = dot(y[j]-x[i], y[j]-x[i]);
                if t == 0.:
                    C[i,j] = prefac / prefac_scale
                else:
                    C[i,j] = prefac * (alpha * t)**nu * special.kv (nu, alpha * t)
                
    
    C *= phi
    return C.view(matrix)
            
            
def NormalizedMatern(x,y,nu,scale,amp):
    """
    C=NormalizedMatern(x,y,nu,scale,amp)
    
    An alternative parametrization of the isotropic Matern covariance function.
    The pointwize variance of draws is 1 regardless of nu and alpha.
    """

    prefac = 1. / 2. ** (nu-1.) / special.gamma(nu)
    scalesq = scale * scale
    
    C = zeros((len(x),len(y)),dtype=float)

    symm = (x is y)

    for i in xrange(len(x)):

        if symm:
            C[i,i] = 1.
            
            for j in range(i+1, len(y)):
                t = dot(y[j]-x[i], y[j]-x[i]) /  scalesq;
                if (t==0):
                    C[i,j]=1.
                else:
                    C[i,j]=prefac * t ** nu * special.kv (nu, t)
                C[j,i]=C[i,j]
                
            
        else:
            for j in xrange(len(y)):

                t = dot(y[j]-x[i], y[j]-x[i]) /  scalesq;

                if (t==0):
                    C[i,j]=1.
                else:
                    C[i,j]=prefac * t ** nu * special.kv (nu, t)

    C *= amp            
    return C.view(matrix)
            