import sys
import os

# Default backend: theano.
_BACKEND = 'theano'

# Check environment for backend
if 'PYMC_SYMB_BACKEND' in os.environ:
    _backend = os.environ['PYMC_SYMB_BACKEND']
    _BACKEND = _backend

if _BACKEND == 'theano':
    sys.stderr.write('Using Theano backend.\n')
    from .backends_theano import *

elif _BACKEND == 'tensorflow':
    sys.stderr.write('Using TensorFlow backend.\n')
    from .backends_tf import *


def backend():
    """Publicly accessible method
    for determining the current backend.
    # Returns
        String, the name of the backend pymc is currently using.
    # Example
    ```python
        >>> pymc3.backend_symbolic.backend()
        'theano'
    ```
    """
    return _BACKEND
