#   Copyright 2025 - present The PyMC Developers
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


def __init__():
    """Make PyMC aware of the xtensor functionality."""
    import pytensor.xtensor

    from pytensor.compile import optdb
    from pytensor.xtensor.vectorization import XRV

    from pymc.initial_point import initial_point_rewrites_db
    from pymc.logprob.abstract import MeasurableOp
    from pymc.logprob.rewriting import logprob_rewrites_db

    # Make PyMC aware of xtensor functionality
    MeasurableOp.register(XRV)
    logprob_rewrites_db.register(
        "pre_lower_xtensor", optdb.query("+lower_xtensor"), "basic", position=0.1
    )
    logprob_rewrites_db.register(
        "post_lower_xtensor", optdb.query("+lower_xtensor"), "cleanup", position=5.1
    )
    initial_point_rewrites_db.register(
        "lower_xtensor", optdb.query("+lower_xtensor"), "basic", position=0.1
    )


__init__()
del __init__

from pytensor.tensor import TensorLike
from pytensor.xtensor import as_xtensor, broadcast, concat, dot, full_like, ones_like, zeros_like
from pytensor.xtensor.basic import tensor_from_xtensor
from pytensor.xtensor.type import XTensorVariable
from xarray import DataArray

XTensorLike = TensorLike | XTensorVariable | DataArray
del DataArray, TensorLike

from pymc.dims import math
from pymc.dims.distributions import *
from pymc.dims.model import Data, Deterministic, Potential
