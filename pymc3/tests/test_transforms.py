import pymc3.distributions.transforms as tr
import numpy as np
import theano
import theano.tensor as tt
from .test_distributions import Simplex, Rplusbig, Rminusbig, Unit, R, Vector, MultiSimplex

from .checks import close_to
from ..theanof import jacobian

tol = 1e-7


def check_transform_identity(transform, domain, constructor=tt.dscalar, test=0):
    x = constructor('x')
    x.tag.test_value = test
    identity_f = theano.function([x], transform.backward(transform.forward(x)))

    for val in domain.vals:
        close_to(val, identity_f(val), tol)


def check_vector_transform_identity(transform, domain):
    return check_transform_identity(transform, domain, tt.dvector, test=np.array([0, 0]))


def get_values(transform, domain=R, constructor=tt.dscalar, test=0):
    x = constructor('x')
    x.tag.test_value = test
    f = theano.function([x], transform.backward(x))

    return np.array([f(val) for val in domain.vals])


def test_simplex():
    check_vector_transform_identity(tr.stick_breaking, Simplex(2))
    check_vector_transform_identity(tr.stick_breaking, Simplex(4))
    check_transform_identity(tr.stick_breaking, MultiSimplex(
        3, 2), constructor=tt.dmatrix, test=np.zeros((2, 2)))


def test_simplex_bounds():
    vals = get_values(tr.stick_breaking, Vector(R, 2),
                      tt.dvector, np.array([0, 0]))

    close_to(vals.sum(axis=1), 1, tol)
    close_to(vals > 0, True, tol)
    close_to(vals < 1, True, tol)


def test_simplex_jacobian_det():
    check_jacobian_det(tr.stick_breaking, Vector(
        R, 2), tt.dvector, np.array([0, 0]), lambda x: x[:-1])


def test_sum_to_1():
    check_vector_transform_identity(tr.sum_to_1, Simplex(2))
    check_vector_transform_identity(tr.sum_to_1, Simplex(4))


def test_sum_to_1_jacobian_det():
    check_jacobian_det(tr.sum_to_1, Vector(Unit, 2),
                       tt.dvector, np.array([0, 0]), lambda x: x[:-1])


def check_jacobian_det(transform, domain,
                       constructor=tt.dscalar,
                       test=0,
                       make_comparable=None,
                       elemwise=False):
    y = constructor('y')
    y.tag.test_value = test

    x = transform.backward(y)
    if make_comparable:
        x = make_comparable(x)

    if not elemwise:
        jac = tt.log(tt.nlinalg.det(jacobian(x, [y])))
    else:
        jac = tt.log(tt.abs_(tt.diag(jacobian(x, [y]))))

    # ljd = log jacobian det
    actual_ljd = theano.function([y], jac)

    computed_ljd = theano.function([y], tt.as_tensor_variable(
        transform.jacobian_det(y)), on_unused_input='ignore')

    for yval in domain.vals:
        close_to(
            actual_ljd(yval),
            computed_ljd(yval), tol)


def test_log():
    check_transform_identity(tr.log, Rplusbig)
    check_jacobian_det(tr.log, Rplusbig, elemwise=True)
    check_jacobian_det(tr.log, Vector(Rplusbig, 2),
                       tt.dvector, [0, 0], elemwise=True)

    vals = get_values(tr.log)
    close_to(vals > 0, True, tol)


def test_logodds():
    check_transform_identity(tr.logodds, Unit)
    check_jacobian_det(tr.logodds, Unit, elemwise=True)
    check_jacobian_det(tr.logodds, Vector(Unit, 2),
                       tt.dvector, [.5, .5], elemwise=True)

    vals = get_values(tr.logodds)
    close_to(vals > 0, True, tol)
    close_to(vals < 1, True, tol)


def test_lowerbound():
    trans = tr.lowerbound(0.0)
    check_transform_identity(trans, Rplusbig)
    check_jacobian_det(trans, Rplusbig, elemwise=True)
    check_jacobian_det(trans, Vector(Rplusbig, 2),
                       tt.dvector, [0, 0], elemwise=True)

    vals = get_values(trans)
    close_to(vals > 0, True, tol)


def test_upperbound():
    trans = tr.upperbound(0.0)
    check_transform_identity(trans, Rminusbig)
    check_jacobian_det(trans, Rminusbig, elemwise=True)
    check_jacobian_det(trans, Vector(Rminusbig, 2),
                       tt.dvector, [-1, -1], elemwise=True)

    vals = get_values(trans)
    close_to(vals < 0, True, tol)


def test_interval():
    for a, b in [(-4, 5.5), (.1, .7), (-10, 4.3)]:
        domain = Unit * np.float64(b - a) + np.float64(a)
        trans = tr.interval(a, b)
        check_transform_identity(trans, domain)
        check_jacobian_det(trans, domain, elemwise=True)

        vals = get_values(trans)
        close_to(vals > a, True, tol)
        close_to(vals < b, True, tol)
