#   Copyright 2024 - present The PyMC Developers
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
"""
Contains tools used to perform inference on ordinary differential equations.

Due to the nature of the model (as well as included solvers), ODE solution may perform slowly.
Another library based on PyMC--sunode--has implemented Adams' method and BDF (backward differentation formula) using the very fast SUNDIALS suite of ODE and PDE solvers.
It is much faster than the ``pm.ode`` implementation.
More information about ``sunode`` is available at: https://github.com/aseyboldt/sunode.
"""
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

from pymc.ode import utils
from pymc.ode.ode import DifferentialEquation
