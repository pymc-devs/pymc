#   Copyright 2024 The PyMC Developers
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

from pymc.step_methods.compound import BlockedStep, CompoundStep
from pymc.step_methods.hmc import NUTS, HamiltonianMC
from pymc.step_methods.metropolis import (
    BinaryGibbsMetropolis,
    BinaryMetropolis,
    CategoricalGibbsMetropolis,
    CauchyProposal,
    DEMetropolis,
    DEMetropolisZ,
    LaplaceProposal,
    Metropolis,
    MultivariateNormalProposal,
    NormalProposal,
    PoissonProposal,
    UniformProposal,
)
from pymc.step_methods.slicer import Slice

# Other step methods can be added by appending to this list
STEP_METHODS: list[type[BlockedStep]] = [
    NUTS,
    HamiltonianMC,
    Metropolis,
    BinaryMetropolis,
    BinaryGibbsMetropolis,
    Slice,
    CategoricalGibbsMetropolis,
]
