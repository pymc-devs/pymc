import numpy as np
import theano
import pymc3 as pm
import pytest


@pytest.fixture(scope="function", autouse=True)
def theano_config():
    config = theano.configparser.change_flags(compute_test_value='raise')
    with config:
        yield


@pytest.fixture(scope='function', autouse=True)
def exception_verbosity():
    config = theano.configparser.change_flags(
        exception_verbosity='high')
    with config:
        yield


@pytest.fixture(scope='function', autouse=False)
def strict_float32():
    if theano.config.floatX == 'float32':
        config = theano.configparser.change_flags(
            warn_float64='raise')
        with config:
            yield
    else:
        yield


@pytest.fixture('function', autouse=False)
def seeded_test():
    # TODO: use this instead of SeededTest
    np.random.seed(42)
    pm.set_tt_rng(42)
