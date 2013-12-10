import numpy as np
from numpy import inf
from pymc.tuning import scaling
from pymc.tuning import starting
from pymc import Model, Uniform, Normal, Beta, Binomial
from pymc.tests.checks import close_to
from . import models

a = np.array([-10, -.01, 0, 10, 1e300, -inf, inf])


def test_adjust_precision():
    a1 = scaling.adjust_precision(a)

    assert all((a1 > 0) & (a1 < 1e200))


def test_guess_scaling():

    start, model, _ = models.non_normal(n=5)
    a1 = scaling.guess_scaling(start, model=model)

    assert all((a1 > 0) & (a1 < 1e200))


def test_find_MAP_discrete():
    tol = 2.0**-11
    alpha = 4
    beta = 4
    n = 20
    yes = 15

    with Model() as model:
        p = Beta('p', alpha, beta)
        ss = Binomial('ss', n=n, p=p)
        s = Binomial('s', n=n, p=p, observed=yes)

        map_est1 = starting.find_MAP()
        map_est2 = starting.find_MAP(vars=model.vars)

    close_to(map_est1['p'], 0.6086956533498806, tol)
    assert 'ss' not in map_est1

    close_to(map_est2['p'], 0.695642178810167, tol)
    assert map_est2['ss'] == 14


def test_find_MAP():
    tol = 2.0**-11  # 16 bit machine epsilon, a low bar
    data = np.random.randn(100)
    # data should be roughly mean 0, std 1, but let's
    # normalize anyway to get it really close
    data = (data-np.mean(data))/np.std(data)

    with Model() as model:
        mu = Uniform('mu', -1, 1)
        sigma = Uniform('sigma', .5, 1.5)
        y = Normal('y', mu=mu, tau=sigma**-2, observed=data)

        # Test gradient minimization
        map_est1 = starting.find_MAP()
        # Test non-gradient minimization
        map_est2 = starting.find_MAP(fmin=starting.optimize.fmin_powell)

    close_to(map_est1['mu'], 0, tol)
    close_to(map_est1['sigma'], 1, tol)

    close_to(map_est2['mu'], 0, tol)
    close_to(map_est2['sigma'], 1, tol)
