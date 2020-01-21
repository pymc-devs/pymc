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

import os
from pymc3.tests import backend_fixtures as bf
from pymc3.backends import ndarray, sqlite
import tempfile
import pytest
import theano

DBNAME = os.path.join(tempfile.gettempdir(), 'test.db')


@pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32 due to inf issues")
class TestSQlite0dSampling(bf.SamplingTestCase):
    backend = sqlite.SQLite
    name = DBNAME
    shape = ()


@pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32")
class TestSQlite1dSampling(bf.SamplingTestCase):
    backend = sqlite.SQLite
    name = DBNAME
    shape = 2


@pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32 due to inf issues")
class TestSQlite2dSampling(bf.SamplingTestCase):
    backend = sqlite.SQLite
    name = DBNAME
    shape = (2, 3)


@pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32 due to inf issues")
class TestSQLite0dSelection(bf.SelectionTestCase):
    backend = sqlite.SQLite
    name = DBNAME
    shape = ()


@pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32")
class TestSQLite1dSelection(bf.SelectionTestCase):
    backend = sqlite.SQLite
    name = DBNAME
    shape = 2


@pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32")
class TestSQLite2dSelection(bf.SelectionTestCase):
    backend = sqlite.SQLite
    name = DBNAME
    shape = (2, 3)


@pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32 due to inf issues")
class TestSQLiteDumpLoad(bf.DumpLoadTestCase):
    backend = sqlite.SQLite
    load_func = staticmethod(sqlite.load)
    name = DBNAME
    shape = (2, 3)


@pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32 due to inf issues")
class TestNDArraySqliteEquality(bf.BackendEqualityTestCase):
    backend0 = ndarray.NDArray
    name0 = None
    backend1 = sqlite.SQLite
    name1 = DBNAME
    shape = (2, 3)
