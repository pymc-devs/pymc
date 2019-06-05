import numpy as np
import scipy.sparse

from pymc3.step_methods.hmc import quadpotential
import pymc3
from pymc3.theanof import floatX
import pytest
import numpy.testing as npt


def test_elemwise_posdef():
    scaling = np.array([0, 2, 3])
    with pytest.raises(quadpotential.PositiveDefiniteError):
        quadpotential.quad_potential(scaling, True)


def test_elemwise_velocity():
    scaling = np.array([1, 2, 3])
    x = floatX(np.ones_like(scaling))
    pot = quadpotential.quad_potential(scaling, True)
    v = pot.velocity(x)
    npt.assert_allclose(v, scaling)
    assert v.dtype == pot.dtype


def test_elemwise_energy():
    scaling = np.array([1, 2, 3])
    x = floatX(np.ones_like(scaling))
    pot = quadpotential.quad_potential(scaling, True)
    energy = pot.energy(x)
    npt.assert_allclose(energy, 0.5 * scaling.sum())


def test_equal_diag():
    np.random.seed(42)
    for _ in range(3):
        diag = np.random.rand(5)
        x = floatX(np.random.randn(5))
        pots = [
            quadpotential.quad_potential(diag, False),
            quadpotential.quad_potential(1. / diag, True),
            quadpotential.quad_potential(np.diag(diag), False),
            quadpotential.quad_potential(np.diag(1. / diag), True),
        ]
        if quadpotential.chol_available:
            diag_ = scipy.sparse.csc_matrix(np.diag(1. / diag))
            pots.append(quadpotential.quad_potential(diag_, True))

        v = np.diag(1. / diag).dot(x)
        e = x.dot(np.diag(1. / diag).dot(x)) / 2
        for pot in pots:
            v_ = pot.velocity(x)
            e_ = pot.energy(x)
            npt.assert_allclose(v_, v, rtol=1e-6)
            npt.assert_allclose(e_, e, rtol=1e-6)


def test_equal_dense():
    np.random.seed(42)
    for _ in range(3):
        cov = np.random.rand(5, 5)
        cov += cov.T
        cov += 10 * np.eye(5)
        inv = np.linalg.inv(cov)
        npt.assert_allclose(inv.dot(cov), np.eye(5), atol=1e-10)
        x = floatX(np.random.randn(5))
        pots = [
            quadpotential.quad_potential(cov, False),
            quadpotential.quad_potential(inv, True),
        ]
        if quadpotential.chol_available:
            pots.append(quadpotential.quad_potential(cov, False))

        v = np.linalg.solve(cov, x)
        e = 0.5 * x.dot(v)
        for pot in pots:
            v_ = pot.velocity(x)
            e_ = pot.energy(x)
            npt.assert_allclose(v_, v, rtol=1e-4)
            npt.assert_allclose(e_, e, rtol=1e-4)


def test_random_diag():
    d = np.arange(10) + 1
    np.random.seed(42)
    pots = [
        quadpotential.quad_potential(d, True),
        quadpotential.quad_potential(1./d, False),
        quadpotential.quad_potential(np.diag(d), True),
        quadpotential.quad_potential(np.diag(1./d), False),
    ]
    if quadpotential.chol_available:
        d_ = scipy.sparse.csc_matrix(np.diag(d))
        pot = quadpotential.quad_potential(d_, True)
        pots.append(pot)
    for pot in pots:
        vals = np.array([pot.random() for _ in range(1000)])
        npt.assert_allclose(vals.std(0), np.sqrt(1./d), atol=0.1)


def test_random_dense():
    np.random.seed(42)
    for _ in range(3):
        cov = np.random.rand(5, 5)
        cov += cov.T
        cov += 10 * np.eye(5)
        inv = np.linalg.inv(cov)
        assert np.allclose(inv.dot(cov), np.eye(5))

        pots = [
            quadpotential.QuadPotentialFull(cov),
            quadpotential.QuadPotentialFullInv(inv),
        ]
        if quadpotential.chol_available:
            pot = quadpotential.QuadPotential_Sparse(scipy.sparse.csc_matrix(cov))
            pots.append(pot)
        for pot in pots:
            cov_ = np.cov(np.array([pot.random() for _ in range(1000)]).T)
            assert np.allclose(cov_, inv, atol=0.1)


def test_user_potential():
    model = pymc3.Model()
    with model:
        pymc3.Normal("a", mu=0, sigma=1)

    # Work around missing nonlocal in python2
    called = []

    class Potential(quadpotential.QuadPotentialDiag):
        def energy(self, x, velocity=None):
            called.append(1)
            return super().energy(x, velocity)

    pot = Potential(floatX([1]))
    with model:
        step = pymc3.NUTS(potential=pot)
        pymc3.sample(10, init=None, step=step, chains=1)
    assert called
