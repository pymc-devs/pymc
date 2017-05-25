import theano
import pytest


@pytest.fixture(scope="session", autouse=True)
def theano_config():
    config = theano.configparser.change_flags(compute_test_value='raise')
    with config:
        yield


@pytest.fixture(scope='function')
def strict_float32():
    config = theano.configparser.change_flags(
        warn_float64='raise',
        floatX='float32')
    with config:
        yield
