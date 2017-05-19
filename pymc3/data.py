from copy import copy
import io
import os
import pkgutil

import numpy as np
import pymc3 as pm
import theano.tensor as tt
import theano

__all__ = [
    'get_data',
    'GeneratorAdapter',
    'DataSampler',
    'Minibatch'
]


def get_data(filename):
    """Returns a BytesIO object for a package data file.

    Parameters
    ----------
    filename : str
        file to load
    Returns
    -------
    BytesIO of the data
    """
    data_pkg = 'pymc3.examples'
    return io.BytesIO(pkgutil.get_data(data_pkg, os.path.join('data', filename)))


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
        self.test_value = pm.smartfloatX(copy(next(generator)))
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
            return pm.smartfloatX(copy(next(self.gen)))

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
    random_seed : int for numpy random generator
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
    >>> new.sample(draws=1000)['mu'].mean()
    10.08339999101371
    >>> new.sample(draws=1000)['sd'].mean()
    1.2178044136104513
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


class Minibatch(object):
    """Multidimensional minibatch

    Parameters
    ----------
    data : ndarray
        initial data
    batch_size : int or List[tuple(size, random_seed)]
        batch size for inference, random seed is needed 
        for child random generators
    in_memory_size : int or List[int,slice,Ellipsis]
        data size for storing in theano.shared
    random_seed : int
        random seed that is used for 1d random slice
    update_shared_f : callable
        gets in_memory_shape and returns np.ndarray

    Attributes
    ----------
    shared : shared tensor
        Used for storing data
    minibatch : minibatch tensor
        Used for training
    """
    def __init__(self, data, batch_size, in_memory_size=None, random_seed=42, update_shared_f=None):
        self._random_seed = random_seed
        in_memory_slc = self._to_slices(in_memory_size)
        self.data = data
        self.batch_size = batch_size
        self.shared = theano.shared(data[in_memory_slc])
        self.in_memory_shape = self.shared.get_value().shape
        self.update_shared_f = update_shared_f
        self.random_slc = self._to_random_slices(self.in_memory_shape, batch_size)
        self.minibatch = self.shared[self.random_slc]

    def rslice(self, total, size, seed):
        if size is None:
            return slice(None)
        elif isinstance(size, int):
            return (pm.tt_rng(seed)
                    .uniform(size=(size, ), low=0.0, high=total - 1e-16)
                    .astype('int64'))
        else:
            raise TypeError('Unrecognized size type, %r' % size)

    @staticmethod
    def _to_slices(user_size):
        if user_size is None:
            return [Ellipsis]
        elif isinstance(user_size, int):
            return slice(None, user_size)
        elif isinstance(user_size, (list, tuple)):
            slc = list()
            for i in user_size:
                if isinstance(i, int):
                    slc.append(i)
                elif i is None:
                    slc.append(slice(None))
                elif i is Ellipsis:
                    slc.append(Ellipsis)
                elif isinstance(i, slice):
                    slc.append(i)
                else:
                    raise TypeError('Unrecognized size type, %r' % user_size)
            return slc
        else:
            raise TypeError('Unrecognized size type, %r' % user_size)

    def _to_random_slices(self, in_memory_shape, batch_size):
        if batch_size is None:
            return [Ellipsis]
        elif isinstance(batch_size, int):
            slc = [self.rslice(in_memory_shape[0], batch_size, self._random_seed)]
        elif isinstance(batch_size, (list, tuple)):
            def check(t):
                if t is Ellipsis or t is None:
                    return True
                else:
                    if not isinstance(t, (tuple, list)):
                        return False
                    else:
                        if not len(t) == 2:
                            return False
                        else:
                            return isinstance(t[0], int) and isinstance(t[1], int)

            if not all(check(t) for t in batch_size):
                raise TypeError('Unrecognized `batch_size` type, expected '
                                'int or List[tuple(size, random_seed)] where '
                                'size and random seed are both ints')
            shape = in_memory_shape
            if Ellipsis in batch_size:
                sep = batch_size.index(Ellipsis)
                begin = batch_size[:sep]
                end = batch_size[sep + 1:]
                if Ellipsis in end:
                    raise ValueError('Double Ellipsis in `batch_size` is restricted')
                mid = [Ellipsis]
            else:
                begin = batch_size
                end = []
                mid = []
            if (len(begin) + len(end)) > len(in_memory_shape):
                raise ValueError('Length of `batch_size` is too big, '
                                 'number of ints is bigger that ndim')
            if len(end) > 0:
                shp_end = shape[-len(end):]
            else:
                shp_end = np.asarray([])
            shp_begin = shape[:len(begin)]
            slc_begin = [self.rslice(shp_begin[i], bs, s) for i, (bs, s) in enumerate(begin)]
            slc_end = [self.rslice(shp_end[i], bs, s) for i, (bs, s) in enumerate(end)]
            slc = slc_begin + mid + slc_end
            slc = slc
        else:
            raise TypeError('Unrecognized size type, %r' % batch_size)
        return pm.theanof.ix_(*slc)

    def refresh(self):
        self.shared.set_value(self.update_shared_f(self.in_memory_shape))

    def set_value(self, value):
        self.shared.set_value(value)

    def __repr__(self):
        return '<Minibatch of %s from memory of shape %s>' % (self.batch_size, list(self.in_memory_shape))
