#   Copyright 2020 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

# commonly used
from pymc3.variational import (
    approximations,
    callbacks,
    flows,
    inference,
    operators,
    opvi,
    test_functions,
    updates,
)
from pymc3.variational.approximations import (
    Empirical,
    FullRank,
    MeanField,
    NormalizingFlow,
    sample_approx,
)
from pymc3.variational.inference import (
    ADVI,
    ASVGD,
    NFVI,
    SVGD,
    FullRankADVI,
    ImplicitGradient,
    Inference,
    KLqp,
    fit,
)
from pymc3.variational.opvi import Approximation, Group

# special
from pymc3.variational.stein import Stein
from pymc3.variational.updates import (
    adadelta,
    adagrad,
    adagrad_window,
    adam,
    adamax,
    apply_momentum,
    apply_nesterov_momentum,
    momentum,
    nesterov_momentum,
    norm_constraint,
    rmsprop,
    sgd,
    total_norm_constraint,
)
