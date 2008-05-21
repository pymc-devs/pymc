# Copyright (c) Anand Patil, 2007

from numpy import ndarray, asarray, array, matrix, ones, diag, pi
from distances import euclidean, aniso_geo_rad, paniso_geo_rad
from distances import geographic as geo_rad
import inspect

__all__ = ['euclidean', 'geo_rad', 'geo_deg', 'aniso_geo_rad', 'aniso_geo_deg', 'partition_aniso_geo_rad', 'partition_aniso_geo_deg', 'apply_distance', 'covariance_function_bundle']

euclidean.__name__ = 'euclidean'
euclidean.__doc__ = """
    D = euclidean(x,y,symm=False)


    :Arguments:

        - `x and y` are arrays of points in Euclidean coordinates
          formatted as follows:
      
          [[x_{0,0} ... x_{0,ndim}],
           [x_{1,0} ... x_{1,ndim}],
           ...
           [x_{N,0} ... x_{N,ndim}]]

        - `symm` indicates whether x and y are references to
          the same array.

      
    Return value is a matrix D, where D[i,j] gives the Euclidean
    distance between the point x[i,:] and y[j,:].
"""

geo_rad.__name__ = 'geo_rad'
geo_rad.__doc__ = """
    D = geo_rad(x,y,symm=False)


    :Arguments:

        - `x and y` are arrays of points in geographic coordinates
          formatted as follows:

          [[lon_0, lat_0],
           [lon_1, lat_1],
           ...
           [lon_N, lat_N]]
 
          Latitudes and longitudes should be in radians.

        - `symm` indicates whether x and y are references to
          the same array.
      
    Return value is a matrix D, where D[i,j] gives the great-circle 
    distance between the point x[i,:] and y[j,:] on a sphere of unit
    radius.
"""

def geo_deg(x,y,symm=False):
    return geo_rad(x*pi/180., y*pi/180., symm)*180./pi
geo_deg.__doc__ = geo_rad.__doc__.replace('radian', 'degree')

aniso_geo_rad.extra_parameters = {'ecc': 'Eccentricity of level sets of distance', 'inc': 'Angle of inclination, in radians'}
aniso_geo_rad.__name__ = 'aniso_geo_rad'
aniso_geo_rad.__doc__ = """
    D = aniso_geo_rad(x,y,inc,ecc,symm=False)


    :Arguments:

        - `x and y` are arrays of points in geographic coordinates
          formatted as follows:

          [[lon_0, lat_0],
           [lon_1, lat_1],
           ...
           [lon_N, lat_N]]

          Latitudes and longitudes should be in radians.

        - `inc` gives the eccentricity of the elliptical level sets of distance.
        
        - `ecc` gives the angle of inclination of the elliptical level sets of 
          distance, in radians.

        - `symm` indicates whether x and y are references to
          the same array.
  
    Return value is a matrix D, where D[i,j] gives the great-circle 
    distance between the point x[i,:] and y[j,:] on a sphere of unit
    radius.
"""

def aniso_geo_deg(x,y,inc,ecc,symm=False):
    return aniso_geo_rad(x*pi/180., y*pi/180., inc*pi/180., ecc, symm)*180./pi
aniso_geo_deg.__doc__ = geo_deg.__doc__.replace('radian', 'degree')
aniso_geo_deg.extra_parameters = {'ecc': 'Eccentricity of level sets of distance', 'inc': 'Angle of inclination, in degrees'}

def partition_aniso_geo_rad(x,y,ctrs,scals,symm=False):
    return paniso_geo_rad(x,y,ctrs,scals,symm)
partition_aniso_geo_rad.extra_parameters = {'ctrs': 'Centers of angular bins, in radians','scals': 'Scales associated with each angular bin'}
partition_aniso_geo_rad.__doc__ = ""

def partition_aniso_geo_deg(x,y,ctrs,scals,symm=False):
    return paniso_geo_rad(x*pi/180., y*pi/180., ctrs*pi/180., scals, symm)*180./pi
partition_aniso_geo_deg.extra_parameters = {'ctrs': 'Centers of angular bins, in radians','scals': 'Scales associated with each angular bin'}
partition_aniso_geo_deg.__doc__ = ""

def regularize_array(A):
    """
    Takes an ndarray as an input.
    
    
    - If the array is one-dimensional, it's assumed to be an array of input values.
    
    - If the array is more than one-dimensional, its last index is assumed to curse
      over spatial dimension.
    
    
    Either way, the return value is at least two dimensional. A.shape[-1] gives the
    number of spatial dimensions.
    """
    if not isinstance(A,ndarray):
        A = array(A, dtype=float)
    else:
        A = asarray(A, dtype=float)
    
    if len(A.shape) <= 1:
        return A.reshape(-1,1)
        
    elif A.shape[-1]>1:
        return A.reshape(-1, A.shape[-1])
    
    else:
        return A
                
def apply_distance(cov_fun, distance_fun):
    """
    A wrapper for the Fortran covariance functions that 
    removes the need for worrying about the common arguments
    amp and scale, and that separates the distance-finding
    from the covariance-evaluating for less code duplication
    and easier nonstationary generalizations.
    """
    
    # Covariance_wrapper takes lots of time in the profiler, but it seems pretty 
    # efficient in that it spends most of its time in distance_fun, cov_fun and
    # the floating-point operations on C.
    def covariance_wrapper(x,y,amp=1.,scale=1.,*args,**kwargs):
        if amp<0. or scale<0.:
            raise ValueError, 'The amp and scale parameters must be positive.'
        symm = (x is y)
        kwargs['symm']=symm
        
        # x = regularize_array(x)
        if symm:
            y=x
        # else:
        #     y = regularize_array(y)

        # Split off the distance arguments
        distance_arg_dict = {}        
        if hasattr(distance_fun, 'extra_parameters'):
            for key in distance_fun.extra_parameters.iterkeys():
                if key in kwargs.keys():
                    distance_arg_dict[key] = kwargs.pop(key)

        # Form the distance matrix
        distance_arg_dict['symm']=symm
        C = distance_fun(x,y,**distance_arg_dict).view(matrix)
        C /= scale

        # Overwrite the distance matrix using a Fortran covariance function
        cov_fun(C,*args,**kwargs)
        C *= amp*amp        

        return C
    
        
    covariance_wrapper.__doc__ = cov_fun.__name__ + '.' + distance_fun.__name__+ covariance_wrapperdoc[0]
    
    # Add covariance parameters to function signature
    if hasattr(cov_fun, 'extra_parameters'):
        covariance_wrapper.extra_cov_params = cov_fun.extra_parameters
        for parameter in cov_fun.extra_parameters.iterkeys():
            covariance_wrapper.__doc__ += ', ' + parameter
    # Add distance parameters to function signature
    if hasattr(distance_fun,'extra_parameters'):
        covariance_wrapper.extra_distance_params = distance_fun.extra_parameters
        for parameter in distance_fun.extra_parameters.iterkeys():
            covariance_wrapper.__doc__ += ', ' + parameter
    # Document covariance parameters
    covariance_wrapper.__doc__ += covariance_wrapperdoc[1]
    if hasattr(cov_fun, 'extra_parameters'):
        for parameter in cov_fun.extra_parameters.iterkeys():
            covariance_wrapper.__doc__ += "\n\n    - " + parameter + ": " + cov_fun.extra_parameters[parameter]
    # Document distance parameters.
    if hasattr(distance_fun,'extra_parameters'):
        for parameter in distance_fun.extra_parameters.iterkeys():
            covariance_wrapper.__doc__ += "\n\n    - " + parameter + ": " + distance_fun.extra_parameters[parameter]
        
    covariance_wrapper.__doc__ += "\n\nDistances are computed using "+distance_fun.__name__+":\n\n"+distance_fun.__doc__
    
    # Expose distance and isotropic covariance function
    covariance_wrapper.distance_fun = distance_fun
    covariance_wrapper.raw_cov_fun = cov_fun

    
    return covariance_wrapper
        
# Common verbiage in the covariance functions' docstrings
covariance_wrapperdoc = ["(x,y",""", amp=1., scale=1.)

A covariance function. Remember, broadcasting for covariance functions works
differently than for numpy universal functions. C(x,y) returns a matrix, and 
C(x) returns a vector.


:Arguments:

    - `x, y`: Arrays on which to evaluate the covariance function.

    - `amp`: The pointwise standard deviation of f.

    - `scale`: The factor by which to scale the distance between points.
             Large value implies long-range correlation."""]

    
class covariance_function_bundle(object):
    """
    B = covariance_function_bundle(cov_fun)
    
    A bundle of related covariance functions that use the stationary, 
    isotropic covariance function cov_fun.
    
    Attributes:

        - `raw`: The raw covariance function, which overwrites a 
          distance matrix with a covariance matrix.

        - `euclidean`: The covariance function wrapped to use
          Euclidean coordinates in R^n, with amp and scale arguments.

        - `geo_rad`: The covariance function wrapped to use
          geographic coordinates (latitude and longitude) on the 
          surface of the sphere, with amp and scale arguments. 
          
          Angles are assumed to be in radians. Radius of sphere is
          assumed to be 1, but you can effectively change the radius
          using the 'scale' argument.
          
        - `geo_deg`: Like geo_rad, but angles are in degrees.

        - `aniso_geo_rad`: Like geo_rad, but the distance function takes extra
          parameters controlling the eccentricity and angle of inclination of
          the elliptical level sets of distance.
          
        - `aniso_geo_deg`: Like aniso_geo_rad, but angles are in degrees.
        
        - `nonstationary`: Not implemented yet.

    Method: 

        - `universal(distance_fun)`: Takes a function that computes a 
          distance matrix for points in some coordinate system and returns 
          the covariance function wrapped to use that coordinate system.
          
    :Arguments:

        - `cov_fun` should overwrite distance matrices with covariance 
          matrices in-place. In addition to the distance matrix, it should
          take an optional argument called 'symm' which indicates whether 
          the output matrix will be symmetric.
    """
    
    def __init__(self, cov_fun):
        
        self.raw = cov_fun
        self.universal(euclidean)
        self.universal(geo_rad)
        self.universal(geo_deg)
        self.universal(aniso_geo_rad)
        self.universal(aniso_geo_deg)
        self.universal(partition_aniso_geo_deg)
        self.universal(partition_aniso_geo_rad)
                
    def universal(self, distance_fun):
        """
        Takes a function that computes a distance matrix for 
        points in some coordinate system and returns self's 
        covariance function wrapped to use that distance function.
        
        
        Uses function apply_distance, which was used to produce
        self.euclidean and self.geographic and their docstrings.
        
        
        :Arguments:
        
            - `distance_fun`: Creates a distance matrix from two
              arrays of points, where the first index iterates
              over separate points and the second over coordinates.
              
              In addition to the arrays x and y, distance_fun should
              take an argument called symm which indicates whether
              x and y are the same array.


        :SeeAlso:
            - `apply_distance()`
        """
        
        new_fun = apply_distance(self.raw, distance_fun)
        try:
            setattr(self, distance_fun.__name__, new_fun)
        except:
            pass
        return new_fun
        
    
def nonstationary(cov_fun):
    """
    A decorator for an isotropic covariance function. Takes one 
    additional argument, kernel, which takes a value for input 
    argument x and returns a kernel matrix Sigma(x), which is square, 
    positive definite and  of the same rank as the dimension of x, as 
    in theorem 1 of the reference below.
    
    
    Christopher J. Paciorek,
    "Nonstationary Gaussian processes for regression and spatial modeling",
    PhD. Thesis, Carnegie Mellon University Department of Statistics,
    May 2003.
    
    
    TODO: Figure out what to do about the quadratic forms in different 
    coordinate systems.
    
    
    Note: No scale parameter is necessary, you can manipulate the kernels
    to get the same effect. The amp parameter is still necessary, though.
    
    WARNING: Not implemented yet.
    """
    def covariance_wrapper(x,y,kernel,amp=1.,*args,**kwargs):
        symm = (x is y)
        kwargs['symm']=symm
        
        x = regularize_array(x)
        if symm:
            y=x
        else:
            # if y_map is not None:
            #     y = y_map(y)
            y = regularize_array(y)
            
        ndim = x.shape[1]

        # Compute and store the kernels (note: you can do this in the loop with no loss
        # if symm=False.)
        kernels_x = zeros((len(x),ndim,ndim),dtype=float)
        for i in range(len(x)):
            kernels[i,:,:] = kernel(x[i,:])
        if not symm:
            kernels_y = zeros((len(y),ndim,ndim),dtype=float)
            for i in range(len(y)):
                kernels[i,:,:] = kernel(y[i,:])
        
        # Compute the distance matrix and the prefactors.
        C = zeros((len(x),len(y)),dtype=flooat)
        prefacs = ones(len(x), len(y), dtype=float) * 2. ** (ndim*.5) * amp
        
        for i in range(len(x)):
            kern_x = asmatrix(kernels_x[i,:,:])
            for j in range(len(y)):
                if symm:
                    kern_y = kern_x
                else:
                    kern_y = asmatrix(kernels_y[j,:,:])
                sum_kern = .5 * (kern_x + kern_y)
                dev = (x[i,:]-y[j,:])
                # Eventually just make this a half-loop if symm=True, of course.
                C[i,j] = (dev.T * solve(sum_kern, dev))
                
                # Eventually, make the prefactor zero if one of the kernels is lower
                # rank than the other. Also make it work if the kernels are both
                # not full rank.
                prefacs[i,j] *= (det(kern_x) * det(kern_y))**(.25) / det(sum_kern)**.5
        
        # Overwrite the distance matrix using a Fortran covariance function
        cov_fun(C,*args,**kwargs)
        C *= prefacs
        
        return C
    
    return covariance_wrapper
    