import pymc as pm
import pymc.gp as gp
from pymc.gp.cov_funs import matern
import numpy as np
import matplotlib.pyplot as pl
import copy
from numpy.random import normal


def make_model(n_fmesh=11, fmesh_is_obsmesh=False):
    x = np.arange(-1., 1., .1)

    # Prior parameters of C
    nu = pm.Uniform('nu', 1., 3, value=1.5)
    phi = pm.Lognormal('phi', mu=.4, tau=1, value=1)
    theta = pm.Lognormal('theta', mu=.5, tau=1, value=1)

    # The covariance dtrm C is valued as a Covariance object.
    @pm.deterministic
    def C(eval_fun=gp.matern.euclidean,
          diff_degree=nu, amp=phi, scale=theta):
        return gp.NearlyFullRankCovariance(eval_fun, diff_degree=diff_degree, amp=amp, scale=scale)

    # Prior parameters of M
    a = pm.Normal('a', mu=1., tau=1., value=1)
    b = pm.Normal('b', mu=.5, tau=1., value=0)
    c = pm.Normal('c', mu=2., tau=1., value=0)

    # The mean M is valued as a Mean object.
    def linfun(x, a, b, c):
        return a * x ** 2 + b * x + c

    @pm.deterministic
    def M(eval_fun=linfun, a=a, b=b, c=c):
        return gp.Mean(eval_fun, a=a, b=b, c=c)

    # The actual observation locations
    actual_obs_locs = np.linspace(-.8, .8, 4)

    if fmesh_is_obsmesh:
        o = actual_obs_locs
        fmesh = o
    else:
        # The unknown observation locations
        o = pm.Normal('o', actual_obs_locs, 1000., value=actual_obs_locs)
        fmesh = np.linspace(-1, 1, n_fmesh)

    # The GP submodel
    sm = gp.GPSubmodel('sm', M, C, fmesh)

    # Observation variance
    V = pm.Lognormal('V', mu=-1, tau=1, value=.0001)
    observed_values = pm.rnormal(actual_obs_locs ** 2, 10000)

    # The data d is just array-valued. It's normally distributed about
    # GP.f(obs_x).
    d = pm.Normal(
        'd',
        mu=sm.f(o),
        tau=1. / V,
        value=observed_values,
        observed=True)

    return locals()
