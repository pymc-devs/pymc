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

import pickle
import traceback

import cloudpickle

from pymc.tests.models import simple_model


class TestPickling:
    def setup_method(self):
        _, self.model, _ = simple_model()

    def test_model_roundtrip(self):
        m = self.model
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            try:
                s = cloudpickle.dumps(m, proto)
                cloudpickle.loads(s)
            except Exception:
                raise AssertionError(
                    "Exception while trying roundtrip with pickle protocol %d:\n" % proto
                    + "".join(traceback.format_exc())
                )
