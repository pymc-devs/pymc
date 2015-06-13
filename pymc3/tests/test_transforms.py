import pymc3 as pm
import theano
import theano.tensor as t 
from .test_distributions import Simplex, Rplusbig, Unit, R, Vector

from .checks import *
from ..theanof import jacobian

def check_transform_identity(transform, domain, constructor=t.dscalar, test=0):
    x = constructor('x')
    x.tag.test_value = test
    identity_f = theano.function([x], transform.backward(transform.forward(x)))

    for val in domain.vals: 
        close_to(val, identity_f(val), 1e-7)

def test_simplex():
    check_transform_identity(pm.simplextransform, Simplex(2), t.dvector, np.array([0,0]))
    check_transform_identity(pm.simplextransform, Simplex(4), t.dvector, np.array([0,0]))

def test_simplex_jacobian_det():
    y = t.dvector('y')
    y.tag.test_value = np.array([0, 0])
    x = pm.simplextransform.backward(y)[:-1]

    jac = t.log(t.nlinalg.det(jacobian(x, [y])))
    #ljd = log jacobian det 
    actual_ljd = theano.function([y], jac)

    computed_ljd = theano.function([y], pm.simplextransform.jacobian_det(y))
    
    for domain in [Vector(R, 1), Vector(R,3)]:
        for yval in domain.vals:
            close_to(
                actual_ljd(yval), 
                computed_ljd(yval), 1e-8)




def test_log():
    check_transform_identity(pm.logtransform, Rplusbig)

def test_logodds():
    check_transform_identity(pm.logtransform, Unit)


def test_interval():
    for a, b in [(-4, 5.5), (.1, .7), (-10, 4.3)]:
        domain = Unit * np.float64(b-a) + np.float64(a)
        check_transform_identity(pm.interval_transform(a,b), domain)
    


