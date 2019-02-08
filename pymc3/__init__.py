# pylint: disable=wildcard-import
__version__ = "3.6"

from .blocking import *
from .distributions import *
from .glm import *
from . import gp
from .math import logaddexp, logsumexp, logit, invlogit, expand_packed_triangular, probit, invprobit
from .model import *
from .model_graph import model_to_graphviz
from .stats import *
from .sampling import *
from .step_methods import *
from .theanof import *
from .tuning import *
from .variational import *
from .vartypes import *
from .exceptions import *
from . import sampling

from .diagnostics import *
from .backends.tracetab import *
from .backends import save_trace, load_trace

from .plots import *
from .tests import test

from .data import *

import logging
_log = logging.getLogger('pymc3')
if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler()
        _log.addHandler(handler)
