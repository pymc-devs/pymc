import numpy as np
import pytest

from pymc import Bernoulli, Flat, HalfFlat, Normal, TruncatedNormal, Uniform
from pymc.distributions import HalfNormal
from pymc.distributions.shape_utils import rv_size_is_none
from pymc.initial_point import make_initial_point_fn
from pymc.model import Model


def test_rv_size_is_none():
    rv = Normal.dist(0, 1, size=None)
    assert rv_size_is_none(rv.owner.inputs[1])

    rv = Normal.dist(0, 1, size=1)
    assert not rv_size_is_none(rv.owner.inputs[1])

    size = Bernoulli.dist(0.5)
    rv = Normal.dist(0, 1, size=size)
    assert not rv_size_is_none(rv.owner.inputs[1])

    size = Normal.dist(0, 1).size
    rv = Normal.dist(0, 1, size=size)
    assert not rv_size_is_none(rv.owner.inputs[1])


def assert_moment_is_expected(model, expected):
    fn = make_initial_point_fn(
        model=model,
        return_transformed=False,
        default_strategy="moment",
    )
    result = fn(0)["x"]
    expected = np.asarray(expected)
    try:
        random_draw = model["x"].eval()
    except NotImplementedError:
        random_draw = result
    assert result.shape == expected.shape == random_draw.shape
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "size, expected",
    [
        (None, 0),
        (5, np.zeros(5)),
        ((2, 5), np.zeros((2, 5))),
    ],
)
def test_flat_moment(size, expected):
    with Model() as model:
        Flat("x", size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "size, expected",
    [
        (None, 1),
        (5, np.ones(5)),
        ((2, 5), np.ones((2, 5))),
    ],
)
def test_halfflat_moment(size, expected):
    with Model() as model:
        HalfFlat("x", size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "lower, upper, size, expected",
    [
        (-1, 1, None, 0),
        (-1, 1, 5, np.zeros(5)),
        (0, np.arange(1, 6), None, np.arange(1, 6) / 2),
        (0, np.arange(1, 6), (2, 5), np.full((2, 5), np.arange(1, 6) / 2)),
    ],
)
def test_uniform_moment(lower, upper, size, expected):
    with Model() as model:
        Uniform("x", lower=lower, upper=upper, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "mu, sigma, size, expected",
    [
        (0, 1, None, 0),
        (0, np.ones(5), None, np.zeros(5)),
        (np.arange(5), 1, None, np.arange(5)),
        (np.arange(5), np.arange(1, 6), (2, 5), np.full((2, 5), np.arange(5))),
    ],
)
def test_normal_moment(mu, sigma, size, expected):
    with Model() as model:
        Normal("x", mu=mu, sigma=sigma, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "sigma, size, expected",
    [
        (1, None, 1),
        (1, 5, np.ones(5)),
        (np.arange(5), None, np.arange(5)),
        (np.arange(5), (2, 5), np.full((2, 5), np.arange(5))),
    ],
)
def test_halfnormal_moment(sigma, size, expected):
    with Model() as model:
        HalfNormal("x", sigma=sigma, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.skip(reason="aeppl interval transform fails when both edges are None")
@pytest.mark.parametrize(
    "mu, sigma, lower, upper, size, expected",
    [
        (0.9, 1, -1, 1, None, 0),
        (0.9, 1, -np.inf, np.inf, 5, np.full(5, 0.9)),
        (np.arange(5), 1, None, 10, (2, 5), np.full((2, 5), 9)),
        (1, np.ones(5), -10, np.inf, None, np.full((2, 5), -9)),
    ],
)
def test_truncatednormal_moment(mu, sigma, lower, upper, size, expected):
    with Model() as model:
        TruncatedNormal("x", mu=mu, sigma=sigma, lower=lower, upper=upper, size=size)
    assert_moment_is_expected(model, expected)


@pytest.mark.parametrize(
    "p, size, expected",
    [
        (0.3, None, 0),
        (0.9, 5, np.ones(5)),
        (np.linspace(0, 1, 4), None, [0, 0, 1, 1]),
        (np.linspace(0, 1, 4), (2, 4), np.full((2, 4), [0, 0, 1, 1])),
    ],
)
def test_bernoulli_moment(p, size, expected):
    with Model() as model:
        Bernoulli("x", p=p, size=size)
    assert_moment_is_expected(model, expected)
