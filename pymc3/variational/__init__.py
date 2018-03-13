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
    NFVI,
    Inference,
    KLqp,
    ImplicitGradient,
    fit
)

from . import approximations
from .approximations import (
    MeanField,
    FullRank,
    Empirical,
    NormalizingFlow,
    sample_approx
)
from . import opvi
from .opvi import (
    Group,
    Approximation
)

# special
from .stein import Stein
from . import flows
from . import operators
from . import test_functions
from . import callbacks
