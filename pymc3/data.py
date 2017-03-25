import pkgutil
import io
from copy import copy
import numpy as np
import theano.tensor as tt
import theano
import pymc3 as pm

__all__ = [
    'get_data_file',
    'GeneratorAdapter',
    'DataSampler'
]


def get_data_file(pkg, path):
    """Returns a file object for a package data file.

    Parameters
    ----------
    pkg : str
        dotted package hierarchy. e.g. "pymc3.examples"
    path : str 
        file path within package. e.g. "data/wells.dat"
    Returns 
    -------
    BytesIO of the data
    """

    return io.BytesIO(pkgutil.get_data(pkg, path))


class GenTensorVariable(tt.TensorVariable):
    def __init__(self, op, type, name=None):
        super(GenTensorVariable, self).__init__(type=type, name=name)
        self.op = op

    def set_gen(self, gen):
        self.op.set_gen(gen)

    def set_default(self, value):
        self.op.set_default(value)

    def clone(self):
        cp = self.__class__(self.op, self.type, self.name)
        cp.tag = copy(self.tag)
        return cp


class GeneratorAdapter(object):
    """
    Helper class that helps to infer data type of generator with looking
    at the first item, preserving the order of the resulting generator
    """

    def make_variable(self, gop, name=None):
        var = GenTensorVariable(gop, self.tensortype, name)
        var.tag.test_value = self.test_value
        return var

    def __init__(self, generator):
        if not pm.vartypes.isgenerator(generator):
            raise TypeError('Object should be generator like')
        self.test_value = copy(next(generator))
        # make pickling potentially possible
        self._yielded_test_value = False
        self.gen = generator
        self.tensortype = tt.TensorType(
            self.test_value.dtype,
            ((False, ) * self.test_value.ndim))

    # python3 generator
    def __next__(self):
        if not self._yielded_test_value:
            self._yielded_test_value = True
            return self.test_value
        else:
            return copy(next(self.gen))

    # python2 generator
    next = __next__

    def __iter__(self):
        return self

    def __eq__(self, other):
        return id(self) == id(other)

    def __hash__(self):
        return hash(id(self))


class DataSampler(object):
    """
    Convenient picklable data sampler for minibatch inference.

    This generator can be used for passing to pm.generator
    creating picklable theano computational grapf

    Parameters
    ----------
    data : array like
    batchsize : sample size over zero axis
    seed : int for numpy random generator
    dtype : str representing dtype

    Usage
    -----
    >>> import pickle
    >>> from functools import partial
    >>> np.random.seed(42) # reproducibility
    >>> pm.set_tt_rng(42)
    >>> data = np.random.normal(size=(1000,)) + 10
    >>> minibatches = DataSampler(data, batchsize=50)
    >>> with pm.Model():
    ...     sd = pm.Uniform('sd', 0, 10)
    ...     mu = pm.Normal('mu', sd=10)
    ...     obs_norm = pm.Normal('obs_norm', mu=mu, sd=sd,
    ...                          observed=minibatches,
    ...                          total_size=data.shape[0])
    ...     adam = partial(pm.adam, learning_rate=.8) # easy problem
    ...     approx = pm.fit(10000, method='advi', obj_optimizer=adam)
    >>> new = pickle.loads(pickle.dumps(approx))
    >>> new #doctest: +ELLIPSIS
    <pymc3.variational.approximations.MeanField object at 0x...>
    >>> new.sample_vp(draws=1000)['mu'].mean()
    10.08339999101371
    >>> new.sample_vp(draws=1000)['sd'].mean()
    1.2178044136104513
    """
    def __init__(self, data, batchsize=50, seed=42, dtype='floatX'):
        self.dtype = theano.config.floatX if dtype == 'floatX' else dtype
        self.rng = np.random.RandomState(seed)
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
