from .checks import close_to
import numpy as np
from pymc3.tuning import starting
from pymc3 import Model, Uniform, Normal, Beta, Binomial, find_MAP, Point
from .models import simple_model, non_normal, simple_arbitrary_det
from .helpers import select_by_precision


def test_accuracy_normal():
    _, model, (mu, _) = simple_model()
    with model:
        newstart = find_MAP(Point(x=[-10.5, 100.5]))
        close_to(newstart['x'], [mu, mu], select_by_precision(float64=1e-5, float32=1E-4))


def test_accuracy_non_normal():
    _, model, (mu, _) = non_normal(4)
    with model:
        newstart = find_MAP(Point(x=[.5, .01, .95, .99]))
        close_to(newstart['x'], mu, select_by_precision(float64=1e-5, float32=1E-4))


def test_find_MAP_discrete():
    tol = 2.0**-11
    alpha = 4
    beta = 4
    n = 20
    yes = 15

    with Model() as model:
        p = Beta('p', alpha, beta)
        Binomial('ss', n=n, p=p)
        Binomial('s', n=n, p=p, observed=yes)

        map_est1 = starting.find_MAP()
        map_est2 = starting.find_MAP(vars=model.vars)

    close_to(map_est1['p'], 0.6086956533498806, tol)

    close_to(map_est2['p'], 0.695642178810167, tol)
    assert map_est2['ss'] == 14


def test_find_MAP_no_gradient():
    _, model = simple_arbitrary_det()
    with model:
        find_MAP()


def test_find_MAP():
    tol = 2.0**-11  # 16 bit machine epsilon, a low bar
    data = np.random.randn(100)
    # data should be roughly mean 0, std 1, but let's
    # normalize anyway to get it really close
    data = (data - np.mean(data)) / np.std(data)

    with Model():
        mu = Uniform('mu', -1, 1)
        sigma = Uniform('sigma', .5, 1.5)
        Normal('y', mu=mu, tau=sigma**-2, observed=data)

        # Test gradient minimization
        map_est1 = starting.find_MAP(progressbar=False)
        # Test non-gradient minimization
        map_est2 = starting.find_MAP(progressbar=False, method="Powell")

    close_to(map_est1['mu'], 0, tol)
    close_to(map_est1['sigma'], 1, tol)

    close_to(map_est2['mu'], 0, tol)
    close_to(map_est2['sigma'], 1, tol)
