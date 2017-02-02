import functools
import numpy as np
import scipy.sparse
import theano.tensor as tt
import theano
import numpy.testing as npt

from pymc3.step_methods.hmc import quadpotential

from nose.tools import raises
from nose.plugins.skip import SkipTest


def require_sparse(f):
    @functools.wraps(f)
    def inner():
        if not quadpotential.chol_available:
            raise SkipTest("Test requires sksparse.cholmod")
        f()
    return inner


@raises(quadpotential.PositiveDefiniteError)
def test_elemwise_posdef():
    scaling = np.array([0, 2, 3])
    quadpotential.quad_potential(scaling, True, True)


@raises(quadpotential.PositiveDefiniteError)
def test_elemwise_posdef2():
    scaling = np.array([0, 2, 3])
    quadpotential.quad_potential(scaling, True, False)


def test_elemwise_velocity():
    scaling = np.array([1, 2, 3])
    x_ = np.ones_like(scaling)
    x = tt.vector()
    x.tag.test_value = x_
    pot = quadpotential.quad_potential(scaling, True, False)
    v = theano.function([x], pot.velocity(x))
    assert np.allclose(v(x_), scaling)
    pot = quadpotential.quad_potential(scaling, True, True)
    v = theano.function([x], pot.velocity(x))
    assert np.allclose(v(x_), 1. / scaling)


def test_elemwise_energy():
    scaling = np.array([1, 2, 3])
    x_ = np.ones_like(scaling)
    x = tt.vector()
    x.tag.test_value = x_
    pot = quadpotential.quad_potential(scaling, True, False)
    energy = theano.function([x], pot.energy(x))
    assert np.allclose(energy(x_), 0.5 * scaling.sum())
    pot = quadpotential.quad_potential(scaling, True, True)
    energy = theano.function([x], pot.energy(x))
    assert np.allclose(energy(x_), 0.5 * (1. / scaling).sum())


def test_equal_diag():
    np.random.seed(42)
    for _ in range(3):
        diag = np.random.rand(5)
        x_ = np.random.randn(5)
        x = tt.vector()
        x.tag.test_value = x_

        samples = np.random.randn(100000, 5) * np.sqrt(diag)

        pots = [
            quadpotential.quad_potential(diag, True, False),
            quadpotential.quad_potential(1. / diag, False, False),
            quadpotential.quad_potential(np.diag(diag), True, False),
            quadpotential.quad_potential(np.diag(1. / diag), False, False),
        ]
        if quadpotential.chol_available:
            diag_ = scipy.sparse.csc_matrix(np.diag(diag))
            pots.append(quadpotential.quad_potential(diag_, True, False))

        pots.append(quadpotential.QuadPotentialSample(samples, 0))

        v = np.diag(diag).dot(x_)
        e = x_.dot(np.diag(diag).dot(x_)) / 2
        for pot in pots[:-1]:
            v_function = theano.function([x], pot.velocity(x))
            e_function = theano.function([x], pot.energy(x))
            npt.assert_allclose(v_function(x_), v)
            npt.assert_allclose(e_function(x_), e)

        # QuadPotentialSample is less accurate
        for pot in pots[-1:]:
            v_function = theano.function([x], pot.velocity(x))
            e_function = theano.function([x], pot.energy(x))
            npt.assert_allclose(v_function(x_), v, rtol=0.2)
            npt.assert_allclose(e_function(x_), e, rtol=0.2)


def test_equal_dense():
    np.random.seed(42)
    for _ in range(3):
        cov = np.random.rand(5, 5)
        cov += cov.T
        cov += 10 * np.eye(5)
        inv = np.linalg.inv(cov)
        assert np.allclose(inv.dot(cov), np.eye(5))
        x_ = np.random.randn(5)
        x = tt.vector()
        x.tag.test_value = x_

        samples = np.random.multivariate_normal(np.zeros_like(x_), cov, 1000)

        pots = [
            quadpotential.quad_potential(cov, True, False),
            quadpotential.quad_potential(inv, False, False),
        ]
        if quadpotential.chol_available:
            pots.append(quadpotential.quad_potential(cov, True, False))

        pots.append(quadpotential.QuadPotentialSample(samples, 0))

        v = np.dot(cov, x_)
        e = 0.5 * x_.dot(v)
        for pot in pots[:-1]:
            v_function = theano.function([x], pot.velocity(x))
            e_function = theano.function([x], pot.energy(x))
            npt.assert_allclose(v_function(x_), v)
            npt.assert_allclose(e_function(x_), e)

        for pot in pots[:-1]:
            v_function = theano.function([x], pot.velocity(x))
            e_function = theano.function([x], pot.energy(x))
            npt.assert_allclose(v_function(x_), v, rtol=0.5)
            npt.assert_allclose(e_function(x_), e, rtol=0.5)


def test_random_diag():
    d = np.arange(10) + 1
    np.random.seed(42)

    samples = np.random.randn(1000, 10) * np.sqrt(d)

    pots = [
        quadpotential.quad_potential(d, True, False),
        quadpotential.quad_potential(1./d, False, False),
        quadpotential.quad_potential(np.diag(d), True, False),
        quadpotential.quad_potential(np.diag(1./d), False, False),
        quadpotential.QuadPotentialSample(samples, 0),
    ]
    if quadpotential.chol_available:
        d_ = scipy.sparse.csc_matrix(np.diag(d))
        pot = quadpotential.quad_potential(d, True, False)
        pots.append(pot)

    for pot in pots:
        vals = np.array([pot.random() for _ in range(1000)])
        assert np.allclose(vals.std(0), np.sqrt(1./d), atol=0.1)


def test_random_dense():
    np.random.seed(42)
    for _ in range(3):
        cov = np.random.rand(5, 5)
        cov += cov.T
        cov += 10 * np.eye(5)
        inv = np.linalg.inv(cov)
        assert np.allclose(inv.dot(cov), np.eye(5))

        samples = np.random.multivariate_normal(np.zeros(5), cov, 1000)

        pots = [
            quadpotential.QuadPotential(cov),
            quadpotential.QuadPotential_Inv(inv),
            quadpotential.QuadPotentialSample(samples, 0),
        ]
        if quadpotential.chol_available:
            pot = quadpotential.QuadPotential_Sparse(scipy.sparse.csc_matrix(cov))
            pots.append(pot)

        for pot in pots:
            cov_ = np.cov(np.array([pot.random() for _ in range(1000)]).T)
            assert np.allclose(cov_, inv, atol=0.1)
