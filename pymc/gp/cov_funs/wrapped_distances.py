from distances import euclidean, aniso_geo_rad, paniso_geo_rad
from distances import geographic as geo_rad
import numpy as np

__all__ = ['euclidean', 'geo_rad', 'geo_deg', 'aniso_geo_rad', 'aniso_geo_deg', 'partition_aniso_geo_rad', 'partition_aniso_geo_deg']

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
    return geo_rad(x*np.pi/180., y*np.pi/180., symm)*180./np.pi
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
    return aniso_geo_rad(x*np.pi/180., y*np.pi/180., inc*np.pi/180., ecc, symm)*180./np.pi
aniso_geo_deg.__doc__ = geo_deg.__doc__.replace('radian', 'degree')
aniso_geo_deg.extra_parameters = {'ecc': 'Eccentricity of level sets of distance', 'inc': 'Angle of inclination, in degrees'}

def partition_aniso_geo_rad(x,y,ctrs,scals,symm=False):
    return paniso_geo_rad(x,y,ctrs,scals,symm)
partition_aniso_geo_rad.extra_parameters = {'ctrs': 'Centers of angular bins, in radians','scals': 'Scales associated with each angular bin'}
partition_aniso_geo_rad.__doc__ = ""

def partition_aniso_geo_deg(x,y,ctrs,scals,symm=False):
    return paniso_geo_rad(x*np.pi/180., y*np.pi/180., ctrs*np.pi/180., scals, symm)*180./np.pi
partition_aniso_geo_deg.extra_parameters = {'ctrs': 'Centers of angular bins, in radians','scals': 'Scales associated with each angular bin'}
partition_aniso_geo_deg.__doc__ = ""
