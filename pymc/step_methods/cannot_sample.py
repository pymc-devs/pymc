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
from pymc.step_methods.arraystep import ArrayStep


class CannotSampleRV(ArrayStep):
    """
    A step method that raises an error when sampling a latent Multinomial variable.
    """

    name = "cannot_sample_rv"

    def __init__(self, vars, **kwargs):
        # Remove keys that ArrayStep.__init__ does not accept.
        kwargs.pop("model", None)
        kwargs.pop("initial_point", None)
        kwargs.pop("compile_kwargs", None)
        self.vars = vars
        super().__init__(vars=vars, fs=[], **kwargs)

    def astep(self, q0):
        # This method is required by the abstract base class.
        raise ValueError("Latent Multinomial variables are not supported")
