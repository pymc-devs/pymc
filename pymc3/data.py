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


class Minibatch(tt.TensorVariable):
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
    @theano.configparser.change_flags(compute_test_value='raise')
    def __init__(self, data, batch_size=128, in_memory_size=None,
                 random_seed=42, update_shared_f=None,
                 broadcastable=None, name='Minibatch'):
        data = pm.smartfloatX(np.asarray(data))
        self._random_seed = random_seed
        in_memory_slc = self._to_slices(in_memory_size)
        self.batch_size = batch_size
        self.shared = theano.shared(data[in_memory_slc])
        self.update_shared_f = update_shared_f
        self.random_slc = self._to_random_slices(self.shared.shape, batch_size)
        minibatch = self.shared[self.random_slc]
        if broadcastable is None:
            broadcastable = (False, ) * minibatch.ndim
        minibatch = tt.patternbroadcast(minibatch, broadcastable)
        self.minibatch = minibatch
        super(Minibatch, self).__init__(
            self.minibatch.type, None, None, name=name)
        theano.Apply(
            theano.compile.view_op,
            inputs=[self.minibatch], outputs=[self])
        self.tag.test_value = copy(self.minibatch.tag.test_value)

    @staticmethod
    def rslice(total, size, seed):
        if size is None:
            return slice(None)
        elif isinstance(size, int):
            return (pm.tt_rng(seed)
                    .uniform(size=(size, ), low=0.0, high=pm.floatX(total) - 1e-16)
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
                                'size and random seed are both ints, got %r' %
                                batch_size)
            shape = in_memory_shape
            if Ellipsis in batch_size:
                sep = batch_size.index(Ellipsis)
                begin = batch_size[:sep]
                end = batch_size[sep + 1:]
                if Ellipsis in end:
                    raise ValueError('Double Ellipsis in `batch_size` is restricted, got %r' %
                                     batch_size)
                if len(end) > 0:
                    shp_mid = shape[sep:-len(end)]
                    mid = [tt.arange(s) for s in shp_mid]
                else:
                    mid = []
            else:
                begin = batch_size
                end = []
                mid = []
            if (len(begin) + len(end)) > len(in_memory_shape.eval()):
                raise ValueError('Length of `batch_size` is too big, '
                                 'number of ints is bigger that ndim, got %r'
                                 % batch_size)
            if len(end) > 0:
                shp_end = shape[-len(end):]
            else:
                shp_end = np.asarray([])
            shp_begin = shape[:len(begin)]
            slc_begin = [self.rslice(shp_begin[i], t[0], t[1])
                         if t is not None else tt.arange(shp_begin[i])
                         for i, t in enumerate(begin)]
            slc_end = [self.rslice(shp_end[i], t[0], t[1])
                       if t is not None else tt.arange(shp_end[i])
                       for i, t in enumerate(end)]
            slc = slc_begin + mid + slc_end
            slc = slc
        else:
            raise TypeError('Unrecognized size type, %r' % batch_size)
        return pm.theanof.ix_(*slc)

    def update_shared(self):
        self.set_value(np.asarray(self.update_shared_f(), self.dtype))

    def set_value(self, value):
        self.shared.set_value(np.asarray(value, self.dtype))

    def clone(self):
        ret = self.type()
        ret.name = self.name
        ret.tag = copy(self.tag)
        return ret
