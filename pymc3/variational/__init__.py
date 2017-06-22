from .advi import advi, sample_vp
from .advi_minibatch import advi_minibatch

# commonly used
from . import updates
from .updates import (
    sgd,
    apply_momentum,
    momentum,
    apply_nesterov_momentum,
    adagrad_window,
    nesterov_momentum,
    adagrad,
    rmsprop,
    adadelta,
    adam,
    adamax,
    norm_constraint,
    total_norm_constraint
)

from . import inference
from .inference import (
    ADVI,
    FullRankADVI,
    SVGD,
    ASVGD,
    Inference,
    fit
)

from . import approximations
from .approximations import (
    MeanField,
    FullRank,
    Empirical,
    sample_approx
)

# special
from .stein import Stein
from . import operators
from . import test_functions
from . import opvi
from . import callbacks
