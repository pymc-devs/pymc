from Node import Variable, ContainerBase
from copy import copy
from numpy import ndarray, array, zeros, shape, arange, where

cdef extern from "numpy/ndarrayobject.h":
    void* PyArray_DATA(object obj)

def TCValue(container):
    """
    Fills in a tuple container's value
    
    :SeeAlso: TupleContainer    
    """
    cdef int i
    cdef object _value, isval
    
    isval = container.isval
    _value = tuple()
    
    for i from 0 <= i < len(container):
        if isval[i]:
            _value = _value + (container[i].value,)
        else:
            _value = _value + (container[i],)
                
    return _value


def LCValue(container):
    """
    Fills in a list container's value.
    
    :SeeAlso: ListContainer
    """
    cdef int i, ind
    cdef object _value, val_ind, nonval_ind
    
    val_ind = container.val_ind
    nonval_ind = container.nonval_ind
    _value = container._value
    
    for i from 0 <= i < container.n_val:
        ind = val_ind[i]
        _value[ind] = container[ind].value
    for i from 0 <= i < container.n_nonval:
        ind = nonval_ind[i]
        _value[ind] = container[ind]

def DCValue(container):
    """
    Fills in a dictionary container's value.
    
    :SeeAlso: DictContainer
    """
    cdef int i
    cdef object _value, val_keys, nonval_keys, key
    
    val_keys = container.val_keys
    nonval_keys = container.nonval_keys
    _value = container._value
    
    for i from 0 <= i < container.n_val:
        key = val_keys[i]
        _value[key] = container[key].value
    for i from 0 <= i < container.n_nonval:
        key = nonval_keys[i]
        _value[key] = container[key]

def OCValue(container):
    """
    Fills in an object container's value.
    
    :SeeAlso: ObjectContainer
    """
    cdef int i
    cdef object _value, val_keys, nonval_keys, key, _dict_container
    
    _dict_container = container._dict_container
    val_keys = _dict_container.val_keys
    nonval_keys = _dict_container.nonval_keys
    _value = container._value.__dict__
    
    
    for i from 0 <= i < _dict_container.n_val:
        key = val_keys[i]
        _value[key] = _dict_container[key].value
    for i from 0 <= i < _dict_container.n_nonval:
        key = nonval_keys[i]
        _value[key] = _dict_container[key]
    
def ACValue(container):
    """
    Fills in an array container's value.
    
    :SeeAlso: ArrayContainer
    """

    cdef int i
    cdef long ind
    
    val_ind = container.val_ind
    nonval_ind = container.nonval_ind
    
    ravelledvalue = container._ravelledvalue
    ravelleddata = container._ravelleddata
    
    for i from 0 <= i < container.n_val:
        ind = val_ind[i]
        ravelledvalue[ind] = ravelleddata[ind].value

    for i from 0 <= i < container.n_nonval:
        ind = nonval_ind[i]
        ravelledvalue[ind] = ravelleddata[ind]
    