from numpy import obj2sctype, ndarray
from numpy import bool_
from numpy import byte, short, intc, int_, longlong, intp
from numpy import ubyte, ushort, uintc, uint, ulonglong, uintp
from numpy import single, float_, longfloat
from numpy import csingle, complex_, clongfloat

# These are only used for membership tests, but things break if they are sets
# rather than lists. TK, Jan 2012.
integer_dtypes = [int, uint, byte, short, intc, int_, longlong, intp, ubyte, ushort, uintc, uint, ulonglong, uintp]
try:
    integer_dtypes.append(long)
except NameError:
    pass       # long is just int for Python 3
float_dtypes = [float, single, float_, longfloat]
complex_dtypes = [complex, csingle, complex_, clongfloat]
bool_dtypes = [bool, bool_]

def check_type(stochastic):
    """
    type, shape = check_type(stochastic)

    Checks the type of a stochastic's value. Output value 'type' may be
    bool, int, float, or complex. Nonnative numpy dtypes are lumped into
    these categories. Output value 'shape' is () if the stochastic's value
    is scalar, or a nontrivial tuple otherwise.
    """
    val = stochastic.value
    
    if val.__class__ in bool_dtypes:
        return bool, ()
    elif val.__class__ in integer_dtypes:
        return int, ()
    elif val.__class__ in float_dtypes:
        return float, ()
    elif val.__class__ in complex_dtypes:
        return complex, ()
    elif isinstance(val, ndarray):
        if obj2sctype(val) in bool_dtypes:
            return bool, val.shape
        elif obj2sctype(val) in integer_dtypes:
            return int, val.shape
        elif obj2sctype(val) in float_dtypes:
            return float, val.shape
        elif obj2sctype(val) in complex_dtypes:
            return complex, val.shape
        else:
            return 'object', val.shape
    else:
        return 'object', ()
    
continuous_types = [float, complex]

def is_continuous(stochastic):
    dtype, shape = check_type(stochastic)
    if dtype in continuous_types:
        return True
    else:
        return False
    
