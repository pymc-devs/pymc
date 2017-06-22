import theano
import numpy as np
import pytest


class DataSampler(object):
    """
    Not for users
    """
    def __init__(self, data, batchsize=50, random_seed=42, dtype='floatX'):
        self.dtype = theano.config.floatX if dtype == 'floatX' else dtype
        self.rng = np.random.RandomState(random_seed)
        self.data = data
        self.n = batchsize

    def __iter__(self):
        return self

    def __next__(self):
        idx = (self.rng
               .uniform(size=self.n,
                        low=0.0,
                        high=self.data.shape[0] - 1e-16)
               .astype('int64'))
        return np.asarray(self.data[idx], self.dtype)

    next = __next__


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


@pytest.fixture('session', params=[
    np.random.uniform(size=(1000, 10))
])
def datagen(request):
    return DataSampler(request.param)
