from csv import reader
from numpy import *
f = file('data.csv')
r = reader(f)
lons = []
lats = []
elevation = []
r.next()
for line in r:
    lons.append(float(line[2])/100.)
    lats.append(float(line[1])/100.)
    elevation.append(float(line[3]))
    
    
lons = array(lons,dtype=float)
lats = array(lats,dtype=float)
elevation = array(elevation,dtype=float)
    

f.close()
 
lat_low = lats.min()
lat_hi = lats.max()
lon_low = lons.min()
lon_hi = lons.max()

lat_mid = (lat_low + lat_hi) / 2.
lat_range = lat_hi - lat_low

lon_mid = (lon_low + lon_hi) / 2.
lon_range = lon_hi - lon_low

N = len(elevation)
obs_mesh = zeros((N,2),dtype=float)
obs_mesh[:,0] = lons
obs_mesh[:,1] = lats

