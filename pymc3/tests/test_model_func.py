import pymc3 as pm
import numpy as np
from .checks import close_to
from .models import simple_model, mv_simple
from ..distributions import Normal


def test_lop():
    start, model, _ = simple_model()
    lp = model.fastlogp
    lp(start)


def test_dlogp():
    start, model, (mu, sig) = simple_model()
    dlogp = model.fastdlogp()
    close_to(dlogp(start), -(start['x'] - mu) / sig, 1. / sig / 100.)


def test_dlogp2():
    start, model, (_, sig) = mv_simple()
    H = np.linalg.inv(sig)
    d2logp = model.fastd2logp()
    close_to(d2logp(start), H, np.abs(H / 100.))


def test_deterministic():
    with pm.Model() as model:
        x = Normal('x', 0, 1)
        y = pm.Deterministic('y', x**2)

    assert model.y == y
    assert model['y'] == y
