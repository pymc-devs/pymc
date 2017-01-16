<<<<<<< 8c87f738da31e2985230434ecff6e25b14c05e91
__version__ = "3.0"
=======
__version__ = "3.0.rc2"
<<<<<<< HEAD
>>>>>>> initial gp commit
=======
print "using gp modified version"
>>>>>>> 947d0c2edad1f378090b082ea4093ba1d5b403e1

from .blocking import *
from .distributions import *
from .math import logsumexp, logit, invlogit
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

from . import glm
from . import gp
from .data import *

import logging
_log = logging.getLogger('pymc3')
if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    _log.addHandler(handler)
