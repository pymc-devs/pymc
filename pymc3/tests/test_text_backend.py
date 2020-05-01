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

import pymc3 as pm
from pymc3.tests import backend_fixtures as bf
from pymc3.backends import ndarray, text
import pytest
import theano


class TestTextSampling:
    name = 'text-db'

    def test_supports_sampler_stats(self):
        with pm.Model():
            pm.Normal("mu", mu=0, sigma=1, shape=2)
            db = text.Text(self.name)
            pm.sample(20, tune=10, init=None, trace=db, cores=2)

    def test_supports_sampler_stats_diverging(self):
        with pm.Model():
            pm.Normal("mu", mu=0, sigma=1, shape=2)
            pm.sample(20, tune=10, init=None, trace='text', cores=1)

    def teardown_method(self):
        bf.remove_file_or_directory(self.name)


class TestText0dSampling(bf.SamplingTestCase):
    backend = text.Text
    name = 'text-db'
    shape = ()


class TestText1dSampling(bf.SamplingTestCase):
    backend = text.Text
    name = 'text-db'
    shape = 2


class TestText2dSampling(bf.SamplingTestCase):
    backend = text.Text
    name = 'text-db'
    shape = (2, 3)


@pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32")
class TestText0dSelection(bf.SelectionTestCase):
    backend = text.Text
    name = 'text-db'
    shape = ()


class TestText1dSelection(bf.SelectionTestCase):
    backend = text.Text
    name = 'text-db'
    shape = 2


class TestText2dSelection(bf.SelectionTestCase):
    backend = text.Text
    name = 'text-db'
    shape = (2, 3)


class TestTextDumpLoad(bf.DumpLoadTestCase):
    backend = text.Text
    load_func = staticmethod(text.load)
    name = 'text-db'
    shape = (2, 3)


class TestTextDumpLoadWithPartialChain(bf.DumpLoadTestCase):
    backend = text.Text
    load_func = staticmethod(text.load)
    name = 'text-db'
    shape = (2, 3)
    write_partial_chain = True


@pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32")
class TestTextDumpFunction(bf.BackendEqualityTestCase):
    backend0 = backend1 = ndarray.NDArray
    name0 = None
    name1 = 'text-db'
    shape = (2, 3)

    @classmethod
    def setup_class(cls):
        super().setup_class()
        text.dump(cls.name1, cls.mtrace1)
        with cls.model:
            cls.mtrace1 = text.load(cls.name1)


class TestNDArrayTextEquality(bf.BackendEqualityTestCase):
    backend0 = ndarray.NDArray
    name0 = None
    backend1 = text.Text
    name1 = 'text-db'
    shape = (2, 3)
