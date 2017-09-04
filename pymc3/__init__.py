# pylint: disable=wildcard-import
__version__ = "3.1"

from .blocking import *
from .distributions import *
from .external import *
from .glm import *
from . import gp
from .math import logaddexp, logsumexp, logit, invlogit, expand_packed_triangular, probit, invprobit
from .model import *
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

import pymc3
import pymc3.plots
# deprecate plotting imports to prepare to remove matplotlib as dependency
autocorrplot = pymc3.util.deprecate(
    'autocorrplot', pymc3.plots.autocorrplot, 'pymc3.plots', version='3.2')
compareplot = pymc3.util.deprecate(
    'compareplot', pymc3.plots.compareplot, 'pymc3.plots', version='3.2')
forestplot = pymc3.util.deprecate(
    'forestplot', pymc3.plots.forestplot, 'pymc3.plots', version='3.2')
kdeplot = pymc3.util.deprecate(
    'kdeplot', pymc3.plots.kdeplot, 'pymc3.plots', version='3.2')
plot_posterior = pymc3.util.deprecate(
    'plot_posterior', pymc3.plots.plot_posterior, 'pymc3.plots', version='3.2')
plot_posterior_predictive_glm = pymc3.util.deprecate(
    'plot_posterior_predictive_glm', pymc3.plots.plot_posterior_predictive_glm, 'pymc3.plots', version='3.2')
traceplot = pymc3.util.deprecate(
    'traceplot', pymc3.plots.traceplot, 'pymc3.plots', version='3.2')
energyplot = pymc3.util.deprecate(
    'energyplot', pymc3.plots.energyplot, 'pymc3.plots', version='3.2')

from .tests import test

from .data import *

import logging
_log = logging.getLogger('pymc3')
if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    _log.addHandler(handler)
