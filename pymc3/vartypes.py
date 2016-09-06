import sys

__all__ = ['bool_types', 'int_types', 'float_types', 'complex_types', 'continuous_types',
           'discrete_types', 'default_type', 'typefilter']

bool_types = set(['int8'])

int_types = set(['int8',
                 'int16',
                 'int32',
                 'int64',
                 'uint8',
                 'uint16',
                 'uint32',
                 'uint64'])
float_types = set(['float32',
                   'float64'])
complex_types = set(['complex64',
                     'complex128'])
continuous_types = float_types | complex_types
discrete_types = bool_types | int_types

default_type = {'discrete': 'int64',
                'continuous': 'float64'}

if sys.version_info[0] == 3:
    string_types = str
else:
    string_types = basestring


def typefilter(vars, types):
    # Returns variables of type `types` from `vars`
    return [v for v in vars if v.dtype in types]
