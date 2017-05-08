import theano
import pytest


@pytest.fixture(scope="session", autouse=True)
def theano_config():
    config = theano.configparser.change_flags(compute_test_value='raise')
    with config:
        yield
