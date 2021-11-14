#   Copyright 2021 The PyMC Developers
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

from typing import Union

import numpy as np

from theano.tensor import TensorVariable

from pymc3.distributions.distribution import Distribution
from pymc3.model import Factor


def logp(
    rv: Union[Factor, Distribution], value: Union[TensorVariable, np.ndarray]
) -> Union[TensorVariable, np.ndarray]:
    """
    Calculate log-probability of a distribution at specified value.

    This function is a limited functionality backported version of PyMC >=4.0 like capabilities.

    Parameters
    ----------
    value : numeric
        Value(s) for which log-probability is calculated. If the log-probabilities for multiple
        values are desired the values must be provided in a numpy array or theano tensor

    Returns
    -------
    logp : TensorVariable or np.ndarray
    """
    return rv.logp(value)
