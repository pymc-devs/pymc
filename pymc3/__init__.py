# pylint: disable=wildcard-import
__version__ = "3.1.rc2"

from .blocking import *
from .distributions import *
from .external import *
from .glm import *
from . import gp
from .math import logsumexp, logit, invlogit, expand_packed_triangular
from .model import *
from .stats import *
from .sampling import *
from .step_methods import *
from .theanof import *
from .tuning import *
from .variational import *
from .vartypes import *
from . import sampling

from .debug import *

from .diagnostics import *
from .backends.tracetab import *

from .plots import *

from .tests import test

from .data import *

import logging
_log = logging.getLogger('pymc3')
if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    _log.addHandler(handler)

def tqdm(*args, **kwargs):
    try:
        from tqdm import tqdm_notebook
        return tqdm_notebook(*args, **kwargs)
    except:
        from tqdm import tqdm
        return tqdm(*args, **kwargs)

def trange(*args, **kwargs):
    try:
        from tqdm import tnrange
        return tnrange(*args, **kwargs)
    except:
        from tqdm import trange
        return trange(*args, **kwargs)


