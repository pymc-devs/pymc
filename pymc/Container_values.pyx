from Node import Variable, ContainerBase
from copy import copy
from numpy import ndarray, array, zeros, shape, arange, where

cdef extern from "numpy/ndarrayobject.h":
    void* PyArray_DATA(object obj)

cdef class LCValue:
    """
    l = LCValue(container)
    l.run()

    run() method fills in value of ListContainer.
    """
    cdef void **val_obj, **nonval_obj
    cdef int *val_ind, *nonval_ind
    cdef object _value
    cdef int n_val, n_nonval

    def __init__(self, container):
        
        cdef int i
        self._value = container._value
        self.val_ind = <int*> PyArray_DATA(container.val_ind)
        self.val_obj = <void**> PyArray_DATA(container.val_obj)
        self.n_val = container.n_val
        self.nonval_ind = <int*> PyArray_DATA(container.nonval_ind)
        self.nonval_obj = <void**> PyArray_DATA(container.nonval_obj)
        self.n_nonval = container.n_nonval
                    
    def run(self):
        cdef int i
        for i from 0 <= i < self.n_val:
            self._value[self.val_ind[i]] = (<object> self.val_obj[i]).value
        for i from 0 <= i < self.n_nonval:
            self._value[self.nonval_ind[i]] = (<object> self.nonval_obj[i])


cdef class DCValue:
    """
    d = DCValue(container)
    d.run()

    run() method replaces DictContainers'keys corresponding to variables and containers
    with their values.
    """
    cdef void **val_obj, **val_keys, **nonval_obj, **nonval_keys
    cdef object _value
    cdef int n_val, n_nonval
    def __init__(self, container):
        self._value = container._value
        self.val_keys = <void**> PyArray_DATA(container.val_keys)
        self.val_obj = <void**> PyArray_DATA(container.val_obj)
        self.n_val = container.n_val
        self.nonval_keys = <void**> PyArray_DATA(container.nonval_keys)
        self.nonval_obj = <void**> PyArray_DATA(container.nonval_obj)
        self.n_nonval = container.n_nonval

    def run(self):
        cdef int i
        cdef object key
        for i from 0 <= i < self.n_val:
            key = <object> self.val_keys[i]
            self._value[key] = (<object> self.val_obj[i]).value
        for i from 0 <= i < self.n_nonval:
            key = <object> self.nonval_keys[i]
            self._value[key] = (<object> self.nonval_obj[i])

cdef class OCValue:
    """
    d = OCValue(container)
    d.run()

    run() method fills in value of container.
    """
    cdef void **val_obj, **val_keys, **nonval_obj, **nonval_keys
    cdef object _value
    cdef int n_val, n_nonval
    def __init__(self, container):
        self._value = container._value.__dict__
        self.val_keys = <void**> PyArray_DATA(container._dict_container.val_keys)
        self.val_obj = <void**> PyArray_DATA(container._dict_container.val_obj)
        self.n_val = container._dict_container.n_val
        self.nonval_keys = <void**> PyArray_DATA(container._dict_container.nonval_keys)
        self.nonval_obj = <void**> PyArray_DATA(container._dict_container.nonval_obj)
        self.n_nonval = container._dict_container.n_nonval
    def run(self):
        cdef int i
        cdef object key
        for i from 0 <= i < self.n_val:
            key = <object> self.val_keys[i]
            self._value[key] = (<object> self.val_obj[i]).value
        for i from 0 <= i < self.n_nonval:
            key = <object> self.nonval_keys[i]
            self._value[key] = (<object> self.nonval_obj[i])

cdef class ACValue:
    """
    A = ACValue(container)
    A.run()

    run() method fills in value of ArrayContainer.
    """
    cdef void **val_obj, **nonval_obj
    cdef int *val_ind, *nonval_ind
    cdef object _ravelledvalue
    cdef int n_val, n_nonval
    def __init__(self, container):
        self._ravelledvalue = container._ravelledvalue
        self.val_ind = <int*> PyArray_DATA(container.val_ind)
        self.val_obj = <void**> PyArray_DATA(container.val_obj)
        self.n_val = container.n_val
        self.nonval_ind = <int*> PyArray_DATA(container.nonval_ind)
        self.nonval_obj = <void**> PyArray_DATA(container.nonval_obj)
        self.n_nonval = container.n_nonval
    def run(self):
        cdef int i
        for i from 0 <= i < self.n_val:
            self._ravelledvalue[self.val_ind[i]] = (<object> self.val_obj[i]).value
        for i from 0 <= i < self.n_nonval:
            self._ravelledvalue[self.nonval_ind[i]] = (<object> self.nonval_obj[i])
