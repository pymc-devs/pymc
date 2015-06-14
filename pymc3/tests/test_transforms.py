import pymc3 as pm
import theano
import theano.tensor as t 
from .test_distributions import Simplex, Rplusbig, Unit, R, Vector

from .checks import *
from ..theanof import jacobian

tol = 1e-7

def check_transform_identity(transform, domain, constructor=t.dscalar, test=0):
    x = constructor('x')
    x.tag.test_value = test
    identity_f = theano.function([x], transform.backward(transform.forward(x)))

    for val in domain.vals: 
        close_to(val, identity_f(val), tol)

def get_values(transform, domain=R, constructor=t.dscalar, test=0):
    x = constructor('x')
    x.tag.test_value = test
    f = theano.function([x], transform.backward(x))

    return np.array([f(val) for val in domain.vals])

def test_simplex():
    check_transform_identity(pm.simplextransform, Simplex(2), t.dvector, np.array([0,0]))
    check_transform_identity(pm.simplextransform, Simplex(4), t.dvector, np.array([0,0]))
    
def test_simplex_bounds():
    vals = get_values(pm.simplextransform, Vector(R, 2), t.dvector, np.array([0,0]))

    close_to(vals.sum(axis=1), 1, tol)
    close_to(vals > 0, True, tol)
    close_to(vals < 1, True, tol)

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
                computed_ljd(yval), tol)


def check_jacobian_det(transform, domain, constructor=t.dscalar, test=0):
    y = constructor('y')
    y.tag.test_value = test

    x = transform.backward(y)

    jac = t.log(t.nlinalg.det(jacobian(x, [y])))
    #ljd = log jacobian det 
    actual_ljd = theano.function([y], jac)

    computed_ljd = theano.function([y], transform.jacobian_det(y))

    for yval in domain.vals:
        close_to(
            actual_ljd(yval), 
            computed_ljd(yval), tol)

def test_log():
    check_transform_identity(pm.logtransform, Rplusbig)
    check_jacobian_det(pm.logtransform, Rplusbig) 
    check_jacobian_det(pm.logtransform, Vector(Rplusbig,2), t.dvector, [0,0]) 

    vals = get_values(pm.logtransform) 
    close_to(vals > 0, True, tol)

def test_logodds():
    check_transform_identity(pm.logoddstransform, Unit)
    check_jacobian_det(pm.logoddstransform, Unit)
    check_jacobian_det(pm.logoddstransform, Vector(Unit,2), t.dvector, [0,0]) 

    vals = get_values(pm.logoddstransform) 
    print (vals)
    close_to(vals > 0, True, tol)
    close_to(vals < 1, True, tol)

def test_interval():
    for a, b in [(-4, 5.5), (.1, .7), (-10, 4.3)]:
        domain = Unit * np.float64(b-a) + np.float64(a)
        trans = pm.interval_transform(a,b)
        check_transform_identity(trans, domain)
        check_jacobian_det(trans, domain)
    
        vals = get_values(trans) 
        close_to(vals > a, True, tol)
        close_to(vals < b, True, tol)
