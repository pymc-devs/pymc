# ============================================================
# = WARNING: This example is very computationally demanding! =
# ============================================================
# I have set the map resolutions to give nice-looking results, but I am using
# an 8-core, 3.0GHz Apple Mac Pro with 8GB of RAM and with environment variable 
# OMP_NUM_THREADS set to 8. If you are using a less powerful machine, you may 
# want to change the 'm' parameters
# below.

# The MCMC takes several hours on my machine. To make it run faster, thin the
# dataset in getdata.py




from model import *
# from mpl_toolkits.basemap import Basemap
# from matplotlib import *
from pylab import *
import model


# ====================
# = Do the inference =
# ====================
# Use the HDF5 database backend, because the trace will take up a lot of memory.
# You can use the 'ram' backend instead if you don't have PyTables installed, but
# you should thin the trace more.
WalkerSampler = MCMC(model, db='hdf5')
WalkerSampler.use_step_method(GPEvaluationGibbs, walker_v, V, d)
WalkerSampler.isample(50000,10000,100)

n = len(WalkerSampler.trace('V')[:])


# ==========================
# = Mean and variance maps =
# ==========================

# This computation is O(m^2)
m = 201
xplot = linspace(x.min(),x.max(),m)
yplot = linspace(y.min(),y.max(),m)
dplot = dstack(meshgrid(xplot,yplot))

Msurf = zeros(dplot.shape[:2])
E2surf = zeros(dplot.shape[:2])

# Get E[v] and E[v**2] over the entire posterior
for i in xrange(n):
    # Reset all variables to their values at frame i of the trace
    WalkerSampler.remember(i)
    # Evaluate the observed mean
    Msurf_i = WalkerSampler.walker_v.M_obs.value(dplot)/n
    Msurf += Msurf_i
    # Evaluate the observed covariance with one argument
    E2surf += (WalkerSampler.walker_v.C_obs.value(dplot) + Msurf_i**2)/n

# Get the posterior variance and standard deviation
Vsurf = E2surf - Msurf**2
SDsurf = sqrt(Vsurf)

# Plot mean and standard deviation surfaces
close('all')
imshow(Msurf, extent=[x.min(),x.max(),y.min(),y.max()],interpolation='nearest')
plot(x,y,'r.',markersize=4)
axis([x.min(),x.max(),y.min(),y.max()])
title('Posterior predictive mean surface')
colorbar()
savefig('elevmean.pdf')

figure()
imshow(SDsurf, extent=[x.min(),x.max(),y.min(),y.max()],interpolation='nearest')
plot(x,y,'r.',markersize=4)
axis([x.min(),x.max(),y.min(),y.max()])
title('Posterior predictive standard deviation surface')
colorbar()
savefig('elevvar.pdf')


# ====================
# = Realization maps =
# ====================

# Use thinner input arrays, this computation is O(m^6)!!
m = 101
xplot = linspace(x.min(),x.max(),m)
yplot = linspace(y.min(),y.max(),m)
dplot = dstack(meshgrid(yplot,xplot))

indices = random_integers(n,size=2)
for j,i in enumerate(indices):
    # Reset all variables to their values at frame i of the trace
    WalkerSampler.remember(i)
    # Evaluate the Gaussian process realisation
    R = WalkerSampler.walker_v.f.value(dplot)

    # Plot the realization
    figure()
    imshow(R,extent=[x.min(),x.max(),y.min(),y.max()],interpolation='nearest')
    plot(x,y,'r.',markersize=4)
    axis([x.min(),x.max(),y.min(),y.max()])
    title('Realization from the posterior predictive distribution')
    colorbar()
    savefig('elevdraw%i.pdf'%j)