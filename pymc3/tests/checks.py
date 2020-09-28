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

import numpy as np


def close_to(x, v, bound, name="value"):
    assert np.all(np.logical_or(np.abs(x - v) < bound, x == v)), (
        name + " out of bounds: " + repr(x) + ", " + repr(v) + ", " + repr(bound)
    )


def close_to_logical(x, v, bound, name="value"):
    assert np.all(np.logical_or(np.abs(np.bitwise_xor(x, v)) < bound, x == v)), (
        name + " out of bounds: " + repr(x) + ", " + repr(v) + ", " + repr(bound)
    )
