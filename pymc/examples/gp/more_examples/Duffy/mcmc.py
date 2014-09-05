# ============================================================
# = WARNING: This example is very computationally demanding! =
# ============================================================
# I have set the map resolutions to give nice-looking results, but I am using
# an 8-core, 3.0GHz Apple Mac Pro with 8GB of RAM and with environment variable 
# OMP_NUM_THREADS set to 8. If you are using a less powerful machine, you may 
# want to change the 'm' parameters
# below.

# The MCMC takes several hours on my machine. To make it run faster, thin the dataset.




from model import *
# from mpl_toolkits.basemap import Basemap
# from matplotlib import *
from pylab import *
from numpy import *
import model
import pymc as pm


data = csv2rec('duffy-jittered.csv')

# Convert from record array to dictionary
data = dict([(k,data[k]) for k in data.dtype.names])
# Create model. Use the HDF5 database backend, because the trace will take up a lot of memory.
DuffySampler = pm.MCMC(model.make_model(**data), db='hdf5', dbcomplevel=1, dbcomplib='zlib', dbname='duffydb.hdf5')

# ====================
# = Do the inference =
# ====================

# Use GPEvaluationGibbs step methods.
DuffySampler.use_step_method(pm.gp.GPEvaluationGibbs, DuffySampler.sp_sub_b, DuffySampler.V_b, DuffySampler.tilde_fb)
DuffySampler.use_step_method(pm.gp.GPEvaluationGibbs, DuffySampler.sp_sub_s, DuffySampler.V_s, DuffySampler.tilde_fs)
# Run the MCMC.
DuffySampler.isample(50000,10000,100)

n = len(DuffySampler.trace('V_b')[:])

# ==========================
# = Mean and variance maps =
# ==========================

import tables
covariate_raster = tables.open_file('africa.hdf5')
xplot = covariate_raster.root.lon[:]*pi/180.
yplot = covariate_raster.root.lat[:]*pi/180.

data = ma.masked_array(covariate_raster.root.data[:], mask = covariate_raster.root.mask[:])

where_unmasked = np.where(True-data.mask)
dpred = dstack(meshgrid(xplot,yplot))[::-1][where_unmasked]
africa = covariate_raster.root.data[:][where_unmasked]

Msurf = zeros(data.shape)
E2surf = zeros(data.shape)

# Get E[v] and E[v**2] over the entire posterior
for i in xrange(n):
    # Reset all variables to their values at frame i of the trace
    DuffySampler.remember(0,i)
    # Evaluate the observed mean
    store_africa_val(DuffySampler.sp_sub_b.M_obs.value, dpred, africa)
    Msurf_b, Vsurf_b = pm.gp.point_eval(DuffySampler.sp_sub_b.M_obs.value, DuffySampler.sp_sub_b.C_obs.value, dpred)
    Msurf_s, Vsurf_s = pm.gp.point_eval(DuffySampler.sp_sub_s.M_obs.value, DuffySampler.sp_sub_s.C_obs.value, dpred)
    Vsurf_b += DuffySampler.V_b.value
    Vsurf_s += DuffySampler.V_s.value
    
    freq_b = pm.invlogit(Msurf_b +pm.rnormal(0,1)*np.sqrt(Vsurf_b))
    freq_s = pm.invlogit(Msurf_s +pm.rnormal(0,1)*np.sqrt(Vsurf_s))
    
    samp_i = (freq_b*freq_s+(1-freq_b)*DuffySampler.p1.value)**2
    
    Msurf[where_unmasked] += samp_i/float(n)
    # Evaluate the observed covariance with one argument
    E2surf[where_unmasked] += samp_i**2/float(n)

# Get the posterior variance and standard deviation
Vsurf = E2surf - Msurf**2
SDsurf = sqrt(Vsurf)

Msurf = ma.masked_array(Msurf, mask=covariate_raster.root.mask[:])
SDsurf = ma.masked_array(SDsurf, mask=covariate_raster.root.mask[:])
covariate_raster.close()

# Plot mean and standard deviation surfaces
close('all')
imshow(Msurf[::-1,:], extent=np.array([xplot.min(),xplot.max(),yplot.min(),yplot.max()])*180./pi,interpolation='nearest')
plot(DuffySampler.lon,DuffySampler.lat,'r.',markersize=4)
axis(np.array([xplot.min(),xplot.max(),yplot.min(),yplot.max()])*180./pi)
title('Posterior predictive mean surface')
xlabel('Degrees longitude')
ylabel('Degrees latitude')
colorbar()
savefig('duffymean.pdf')

figure()
imshow(SDsurf[::-1,:], extent=np.array([xplot.min(),xplot.max(),yplot.min(),yplot.max()])*180./pi,interpolation='nearest')
plot(DuffySampler.lon,DuffySampler.lat,'r.',markersize=4)
axis(np.array([xplot.min(),xplot.max(),yplot.min(),yplot.max()])*180./pi)
title('Posterior predictive standard deviation surface')
xlabel('Degrees longitude')
ylabel('Degrees latitude')
colorbar()
savefig('duffyvar.pdf')
