from .advi import advi, sample_vp
from .advi_minibatch import advi_minibatch

from .updates import (
    sgd,
    apply_momentum,
    momentum,
    apply_nesterov_momentum,
    nesterov_momentum,
    adagrad,
    adagrad_window,
    rmsprop,
    adadelta,
    adam,
    adamax,
    norm_constraint,
    total_norm_constraint,
)

from .inference import (
    ADVI,
    FullRankADVI,
    SVGD,
    fit,
)
from .approximations import (
    Empirical,
    FullRank,
    MeanField,
    sample_approx
)

from . import approximations
from . import operators
from . import test_functions
from . import opvi
from . import updates
from . import inference
from . import callbacks
