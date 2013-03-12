
bool_types = set(['int8'])
   
int_types = set(['int8',
            'int16' ,   
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

default_type = {'discrete' :    'int64', 
                'continuous':   'float64'}

def typefilter(vars, types):
    return filter(lambda v: v.dtype in types, vars)
