"""
A simple model to test the GUI.
Sample the parameters of a Generalized Extreme Value (GEV) distribution.
"""

from PyMC import data, parameter, Uniform, Beta, gev_like
from numpy import array

# Define the priors for the location (xi), scale (alpha) and shape (kappa) 
# parameters. 

xi = Uniform('xi', rseed=True, lower=0, upper=60, doc='Location parameter')

@parameter
def alpha(value=5):
    """Scale parameter"""
    return 1./value

kappa = Beta('kappa', rseed=True, alpha=5., beta=6., doc='Shape parameter')

annual_maxima = array([ 23.6,  11.9,  18.8,  13. ,  19.8,  19.8,  24.9,  21.6,  39.6,
        17.5,  17. ,  17.8,  30.5,  26.7,  17.8,  27.7,  27.2,  18.3,
        27.7,  40.4,  20.1,  19.8,  39.9,  25.9,  23.1,  48.5,  39.1,
        14. ,  15.2,  36.6,  24.4,  41.9,  17.8,  14.7,  12.2,  17.1,
        19.1,  30.2,  20.1,  22. ,  21.2,   9.9,  20.8,  44. ,  24.6,
        35. ,  14.8,  26.2,  18.2,  17.1,  27. ,  20.2,  20.8,  20.3,
        16.6,  17.2,  30.7,  18.6,  17.8,  24.5,  24.6,  35.8])
        
@data
def D(value=annual_maxima, location=xi, scale=alpha, shape=kappa):
   return gev_like(value, shape, location, scale)
   

