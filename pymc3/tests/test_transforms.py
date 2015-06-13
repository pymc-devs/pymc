import pymc3 as pm
from .test_distributions import Simplex, Rplusbig, Unit
from .checks import *

def check_transform_identity(transform, domain):
    for val in domain.vals: 
        val2 = transform.backward(transform.forward(val)).tag.test_value
        close_to(val, val2, 1e-7)

def test_simplex():
    check_transform_identity(pm.simplextransform, Simplex(1))
    check_transform_identity(pm.simplextransform, Simplex(5))

def test_log():
    check_transform_identity(pm.logtransform, Rplusbig)

def test_logodds():
    check_transform_identity(pm.logtransform, Unit)


def test_interval():
    for a, b in [(-4, 5.5), (.1, .7), (-10, 4.3)]:
        domain = Unit * np.float64(b-a) + np.float64(a)
        check_transform_identity(pm.interval_transform(a,b), domain)
    


