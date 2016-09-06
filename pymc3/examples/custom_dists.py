# This model was presented by Jake Vanderplas in his blog post about
# comparing different MCMC packages
# http://jakevdp.github.io/blog/2014/06/14/frequentism-and-bayesianism-4-bayesian-in-python/
#
# While at the core it's just a linear regression, it's a nice
# illustration of using Jeffrey priors and custom density
# distributions in PyMC3.
#
# Adapted to PyMC3 by Thomas Wiecki

import matplotlib.pyplot as plt
import numpy as np

import pymc3
import theano.tensor as tt

np.random.seed(42)
theta_true = (25, 0.5)
xdata = 100 * np.random.random(20)
ydata = theta_true[0] + theta_true[1] * xdata

# add scatter to points
xdata = np.random.normal(xdata, 10)
ydata = np.random.normal(ydata, 10)
data = {'x': xdata, 'y': ydata}

with pymc3.Model() as model:
    alpha = pymc3.Uniform('intercept', -100, 100)
    # Create custom densities
    beta = pymc3.DensityDist('slope', lambda value: -
                             1.5 * tt.log(1 + value**2), testval=0)
    sigma = pymc3.DensityDist(
        'sigma', lambda value: -tt.log(tt.abs_(value)), testval=1)
    # Create likelihood
    like = pymc3.Normal('y_est', mu=alpha + beta *
                        xdata, sd=sigma, observed=ydata)

    start = pymc3.find_MAP()
    step = pymc3.NUTS(scaling=start)  # Instantiate sampler
    trace = pymc3.sample(10000, step, start=start)


#################################################
# Create some convenience routines for plotting
# All functions below written by Jake Vanderplas

def compute_sigma_level(trace1, trace2, nbins=20):
    """From a set of traces, bin by number of standard deviations"""
    L, xbins, ybins = np.histogram2d(trace1, trace2, nbins)
    L[L == 0] = 1E-16
    logL = np.log(L)

    shape = L.shape
    L = L.ravel()

    # obtain the indices to sort and unsort the flattened array
    i_sort = np.argsort(L)[::-1]
    i_unsort = np.argsort(i_sort)

    L_cumsum = L[i_sort].cumsum()
    L_cumsum /= L_cumsum[-1]

    xbins = 0.5 * (xbins[1:] + xbins[:-1])
    ybins = 0.5 * (ybins[1:] + ybins[:-1])

    return xbins, ybins, L_cumsum[i_unsort].reshape(shape)


def plot_MCMC_trace(ax, xdata, ydata, trace, scatter=False, **kwargs):
    """Plot traces and contours"""
    xbins, ybins, sigma = compute_sigma_level(trace[0], trace[1])
    ax.contour(xbins, ybins, sigma.T, levels=[0.683, 0.955], **kwargs)
    if scatter:
        ax.plot(trace[0], trace[1], ',k', alpha=0.1)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\beta$')


def plot_MCMC_model(ax, xdata, ydata, trace):
    """Plot the linear model and 2sigma contours"""
    ax.plot(xdata, ydata, 'ok')

    alpha, beta = trace[:2]
    xfit = np.linspace(-20, 120, 10)
    yfit = alpha[:, None] + beta[:, None] * xfit
    mu = yfit.mean(0)
    sig = 2 * yfit.std(0)

    ax.plot(xfit, mu, '-k')
    ax.fill_between(xfit, mu - sig, mu + sig, color='lightgray')

    ax.set_xlabel('x')
    ax.set_ylabel('y')


def plot_MCMC_results(xdata, ydata, trace, colors='k'):
    """Plot both the trace and the model together"""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    plot_MCMC_trace(ax[0], xdata, ydata, trace, True, colors=colors)
    plot_MCMC_model(ax[1], xdata, ydata, trace)

pymc3_trace = [trace['intercept'],
               trace['slope'],
               trace['sigma']]

plot_MCMC_results(xdata, ydata, pymc3_trace)
plt.show()
