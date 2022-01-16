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

from pymc.tests.models import simple_model


class TestProfile:
    def setup_method(self):
        _, self.model, _ = simple_model()

    def test_profile_model(self):
        assert self.model.profile(self.model.logpt()).fct_call_time > 0

    def test_profile_variable(self):
        rv = self.model.basic_RVs[0]
        assert self.model.profile(self.model.logpt(vars=[rv], sum=False)).fct_call_time

    def test_profile_count(self):
        count = 1005
        assert self.model.profile(self.model.logpt(), n=count).fct_callcount == count
