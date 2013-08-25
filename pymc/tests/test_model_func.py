import pymc as pm
from models import *
from checks import *


def test_lop():
    start, model, _ = simple_model()

    lp = model.logpc

    lp(start)


def test_dlogp():
    start, model, (mu, sig) = simple_model()
    dlogp = model.dlogpc()

    close_to(dlogp(start), -(start['x'] - mu) / sig, 1. / sig / 100.)


def test_dlogp2():
    start, model, (mu, sig) = mv_simple()
    H = np.linalg.inv(sig)

    d2logp = model.d2logpc()

    close_to(d2logp(start), H, np.abs(H / 100.))


def test_named():
    with pm.Model() as model:
        x = Normal('x', 0,1)
        y = pm.named(x**2, 'y')

    assert model.y == y
    assert model['y'] == y
