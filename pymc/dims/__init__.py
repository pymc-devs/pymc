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
    """Make PyMC aware of the xtensor functionality.

    This should be done eagerly once development matures.
    """
    import datetime
    import warnings

    from pytensor.compile import optdb

    from pymc.initial_point import initial_point_rewrites_db
    from pymc.logprob.abstract import MeasurableOp
    from pymc.logprob.rewriting import logprob_rewrites_db

    # Filter PyTensor xtensor warning, we emmit our own warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        import pytensor.xtensor

        from pytensor.xtensor.vectorization import XRV

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

    # TODO: Better model of probability of bugs
    day_of_conception = datetime.date(2025, 6, 17)
    day_of_last_bug = datetime.date(2025, 6, 30)
    today = datetime.date.today()
    days_with_bugs = (day_of_last_bug - day_of_conception).days
    days_without_bugs = (today - day_of_last_bug).days
    p = 1 - (days_without_bugs / (days_without_bugs + days_with_bugs + 10))
    if p > 0.05:
        warnings.warn(
            f"The `pymc.dims` module is experimental and may contain critical bugs (p={p:.3f}).\n"
            "Please report any issues you encounter at https://github.com/pymc-devs/pymc/issues.\n"
            "API changes are expected in future releases.\n",
            UserWarning,
            stacklevel=2,
        )


__init__()
del __init__

from pytensor.xtensor import as_xtensor, broadcast, concat, dot, full_like, ones_like, zeros_like
from pytensor.xtensor.basic import tensor_from_xtensor

from pymc.dims import math
from pymc.dims.distributions import *
from pymc.dims.model import Data, Deterministic, Potential
