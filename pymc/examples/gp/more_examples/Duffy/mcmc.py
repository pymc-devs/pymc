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
DuffySampler = pm.MCMC(model.make_model(**dict([(k,data[k]) for k in data.dtype.names])), db='hdf5', dbcomplevel=1, dbcomplib='zlib', dbname='duffydb.hdf5')

# ====================
# = Do the inference =
# ====================
# Use the HDF5 database backend, because the trace will take up a lot of memory.
DuffySampler.use_step_method(pm.gp.GPEvaluationGibbs, DuffySampler.sp_sub_b, DuffySampler.V_b, DuffySampler.eps_p_fb)
DuffySampler.use_step_method(pm.gp.GPEvaluationGibbs, DuffySampler.sp_sub_0, DuffySampler.V_0, DuffySampler.eps_p_f0)
DuffySampler.isample(50000,10000,100)

n = len(DuffySampler.trace('V_b')[:])


# ==========================
# = Mean and variance maps =
# ==========================

import tables
covariate_raster = tables.openFile('africa.hdf5')
xplot = covariate_raster.root.lon[:]*pi/180.
yplot = covariate_raster.root.lat[:]*pi/180.

data = ma.masked_array(covariate_raster.root.data[:], mask = covariate_raster.root.mask[:])
covariate_raster.close()

where_unmasked = np.where(True-data.mask)
dpred = dstack(meshgrid(xplot,yplot))[::-1][where_unmasked]

# This computation is O(m^2)
Msurf = zeros(data.shape)
E2surf = zeros(data.shape)

# Get E[v] and E[v**2] over the entire posterior
for i in xrange(n):
    # Reset all variables to their values at frame i of the trace
    DuffySampler.remember(0,i)
    # Evaluate the observed mean
    Msurf_b, Vsurf_b = pm.gp.point_eval(DuffySampler.sp_sub_b.M_obs.value, DuffySampler.sp_sub_b.C_obs.value, dpred)
    Msurf_0, Vsurf_0 = pm.gp.point_eval(DuffySampler.sp_sub_0.M_obs.value, DuffySampler.sp_sub_0.C_obs.value, dpred)
    
    freq_b = pm.invlogit(Msurf_b +pm.rnormal(0,1)*np.sqrt(Vsurf_b))
    freq_0 = pm.invlogit(Msurf_0 +pm.rnormal(0,1)*np.sqrt(Vsurf_0))
    
    samp_i = (freq_b*freq_0+(1-freq_b)*DuffySampler.p1.value)**2
    
    Msurf[where_unmasked] += samp_i/float(n)
    # Evaluate the observed covariance with one argument
    E2surf[where_unmasked] += samp_i**2/float(n)

# Get the posterior variance and standard deviation
Vsurf = E2surf - Msurf**2
SDsurf = sqrt(Vsurf)

Msurf = ma.masked_array(Msurf, mask=covariate_raster.root.mask[:])
SDsurf = ma.masked_array(SDsurf, mask=covariate_raster.root.mask[:])


# Plot mean and standard deviation surfaces
close('all')
imshow(Msurf[::-1,:], extent=[xplot.min(),xplot.max(),yplot.min(),yplot.max()],interpolation='nearest')
plot(DuffySampler.lon*pi/180.,DuffySampler.lat*pi/180.,'r.',markersize=4)
axis([xplot.min(),xplot.max(),yplot.min(),yplot.max()])
title('Posterior predictive mean surface')
colorbar()
savefig('duffymean.pdf')

figure()
imshow(SDsurf[::-1,:], extent=[xplot.min(),xplot.max(),yplot.min(),yplot.max()],interpolation='nearest')
plot(DuffySampler.lon*pi/180.,DuffySampler.lat*pi/180.,'r.',markersize=4)
axis([xplot.min(),xplot.max(),yplot.min(),yplot.max()])
title('Posterior predictive standard deviation surface')
colorbar()
savefig('duffyvar.pdf')


# # ====================
# # = Realization maps =
# # ====================
# 
# # Use thinner input arrays, this computation is O(m^6)!!
# m = 101
# xplot = linspace(x.min(),x.max(),m)
# yplot = linspace(y.min(),y.max(),m)
# dpred = dstack(meshgrid(yplot,xplot))
# 
# indices = random_integers(n,size=2)
# for j,i in enumerate(indices):
#     # Reset all variables to their values at frame i of the trace
#     DuffySampler.remember(0,i)
#     # Evaluate the Gaussian process realisation
#     R = DuffySampler.walker_v.f.value(dpred)
# 
#     # Plot the realization
#     figure()
#     imshow(R,extent=[x.min(),x.max(),y.min(),y.max()],interpolation='nearest')
#     plot(x,y,'r.',markersize=4)
#     axis([x.min(),x.max(),y.min(),y.max()])
#     title('Realization from the posterior predictive distribution')
#     colorbar()
#     savefig('elevdraw%i.pdf'%j)