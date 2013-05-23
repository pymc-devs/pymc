'''
Created on Jan 20, 2011

@author: jsalvatier
'''
import numpy as np
from collections import defaultdict

_sts_memory = defaultdict(dict)


def sum_to_shape(key1, key2, value, sum_shape):

    try:
        axes, lx = _sts_memory[key1][key2]

    except KeyError:

        value_shape = np.array(np.shape(value))

        sum_shape_expanded = np.zeros(value_shape.size)
        sum_shape_expanded[0:len(sum_shape)] += np.array(sum_shape)

        axes = np.where(sum_shape_expanded != value_shape)[0]
        lx = np.size(axes)

        _sts_memory[key1][key2] = (axes, lx)

    if lx > 0:
        return np.apply_over_axes(np.sum, value, axes)

    else:
        return value
