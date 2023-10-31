#   Copyright 2023 The PyMC Developers
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
import threading
import traceback
import unittest
import warnings

from unittest.mock import MagicMock, patch

import arviz as az
import cloudpickle
import numpy as np
import numpy.ma as ma
import numpy.testing as npt
import pytensor
import pytensor.sparse as sparse
import pytensor.tensor as pt
import pytest
import scipy.sparse as sps
import scipy.stats as st

from pytensor.graph import graph_inputs
from pytensor.raise_op import Assert, assert_op
from pytensor.tensor import TensorVariable
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.sharedvar import ScalarSharedVariable
from pytensor.tensor.variable import TensorConstant

import pymc as pm

from pymc import Deterministic, Potential
from pymc.blocking import DictToArrayBijection, RaveledVars
from pymc.distributions import Normal, transforms
from pymc.distributions.distribution import PartialObservedRV
from pymc.distributions.transforms import log, simplex
from pymc.exceptions import ImputationWarning, ShapeError, ShapeWarning
from pymc.logprob.basic import transformed_conditional_logp
from pymc.logprob.transforms import IntervalTransform
from pymc.model import Point, ValueGradFunction, modelcontext
from pymc.util import _FutureWarningValidatingScratchpad
from pymc.variational.minibatch_rv import MinibatchRandomVariable
from tests.models import simple_model


class NewModel(pm.Model):
    def __init__(self, name="", model=None):
        super().__init__(name, model)
        assert pm.modelcontext(None) is self
        # 1) init variables with Var method
        self.register_rv(pm.Normal.dist(), "v1")
        self.v2 = pm.Normal("v2", mu=0, sigma=1)
        # 2) Potentials and Deterministic variables with method too
        # be sure that names will not overlap with other same models
        pm.Deterministic("d", pt.constant(1))
        pm.Potential("p", pt.constant(1))


class DocstringModel(pm.Model):
    def __init__(self, mean=0, sigma=1, name="", model=None):
        super().__init__(name, model)
        self.register_rv(Normal.dist(mu=mean, sigma=sigma), "v1")
        Normal("v2", mu=mean, sigma=sigma)
        Normal("v3", mu=mean, sigma=Normal("sigma", mu=10, sigma=1, initval=1.0))
        Deterministic("v3_sq", self.v3**2)
        Potential("p1", pt.constant(1))


class TestBaseModel:
    def test_setattr_properly_works(self):
        with pm.Model() as model:
            pm.Normal("v1")
            assert len(model.value_vars) == 1
            with pm.Model("sub") as submodel:
                submodel.register_rv(pm.Normal.dist(), "v1")
                assert hasattr(submodel, "v1")
                assert len(submodel.value_vars) == 1
            assert len(model.value_vars) == 2
            with submodel:
                submodel.register_rv(pm.Normal.dist(), "v2")
                assert hasattr(submodel, "v2")
                assert len(submodel.value_vars) == 2
            assert len(model.value_vars) == 3

    def test_context_passes_vars_to_parent_model(self):
        with pm.Model() as model:
            assert pm.model.modelcontext(None) == model
            assert pm.Model.get_context() == model
            # a set of variables is created
            nm = NewModel()
            assert pm.Model.get_context() == model
            # another set of variables are created but with prefix 'another'
            usermodel2 = NewModel(name="another")
            assert pm.Model.get_context() == model
            assert usermodel2._parent == model
            # you can enter in a context with submodel
            with usermodel2:
                usermodel2.register_rv(pm.Normal.dist(), "v3")
                pm.Normal("v4")
                # this variable is created in parent model too
        assert "another::v2" in model.named_vars
        assert "another::v3" in model.named_vars
        assert "another::v3" in usermodel2.named_vars
        assert "another::v4" in model.named_vars
        assert "another::v4" in usermodel2.named_vars
        assert hasattr(usermodel2, "v3")
        assert hasattr(usermodel2, "v2")
        assert hasattr(usermodel2, "v4")
        # When you create a class based model you should follow some rules
        with model:
            m = NewModel("one_more")
        assert m.d is model["one_more::d"]
        assert m["d"] is model["one_more::d"]
        assert m["one_more::d"] is model["one_more::d"]


class TestNested:
    def test_nest_context_works(self):
        with pm.Model() as m:
            new = NewModel()
            with new:
                assert pm.modelcontext(None) is new
            assert pm.modelcontext(None) is m
        assert "v1" in m.named_vars
        assert "v2" in m.named_vars

    def test_named_context(self):
        with pm.Model() as m:
            NewModel(name="new")
        assert "new::v1" in m.named_vars
        assert "new::v2" in m.named_vars

    def test_docstring_example1(self):
        usage1 = DocstringModel()
        assert "v1" in usage1.named_vars
        assert "v2" in usage1.named_vars
        assert "v3" in usage1.named_vars
        assert "v3_sq" in usage1.named_vars
        assert len(usage1.potentials), 1

    def test_docstring_example2(self):
        with pm.Model() as model:
            DocstringModel(name="prefix")
        assert "prefix::v1" in model.named_vars
        assert "prefix::v2" in model.named_vars
        assert "prefix::v3" in model.named_vars
        assert "prefix::v3_sq" in model.named_vars
        assert len(model.potentials), 1

    def test_duplicates_detection(self):
        with pm.Model():
            DocstringModel(name="prefix")
            with pytest.raises(ValueError):
                DocstringModel(name="prefix")

    def test_model_root(self):
        with pm.Model() as model:
            assert model is model.root
            with pm.Model() as sub:
                assert model is sub.root

    def test_prefix_add_uses_separator(self):
        with pm.Model("foo"):
            foobar = pm.Normal("foobar")
            assert foobar.name == "foo::foobar"

    def test_nested_named_model_repeated(self):
        with pm.Model("sub") as model:
            b = pm.Normal("var")
            with pm.Model("sub"):
                b = pm.Normal("var")
        assert {"sub::var", "sub::sub::var"} == set(model.named_vars.keys())

    def test_nested_named_model(self):
        with pm.Model("sub1") as model:
            b = pm.Normal("var")
            with pm.Model("sub2"):
                b = pm.Normal("var")
        assert {"sub1::var", "sub1::sub2::var"} == set(model.named_vars.keys())

    def test_nested_model_to_netcdf(self, tmp_path):
        with pm.Model("scope") as model:
            b = pm.Normal("var")
            trace = pm.sample(100, tune=0)
        az.to_netcdf(trace, tmp_path / "trace.nc")
        trace1 = az.from_netcdf(tmp_path / "trace.nc")
        assert "scope::var" in trace1.posterior

    def test_bad_name(self):
        with pm.Model() as model:
            with pytest.raises(KeyError):
                b = pm.Normal("var::")
        with pytest.raises(KeyError):
            with pm.Model("scope::") as model:
                b = pm.Normal("v")


class TestObserved:
    def test_observed_rv_fail(self):
        with pytest.raises(TypeError):
            with pm.Model():
                x = Normal("x")
                Normal("n", observed=x)

    def test_observed_type(self):
        X_ = pm.floatX(np.random.randn(100, 5))
        X = pm.floatX(pytensor.shared(X_))
        with pm.Model():
            x1 = pm.Normal("x1", observed=X_)
            x2 = pm.Normal("x2", observed=X)

        assert x1.type.dtype == X.type.dtype
        assert x2.type.dtype == X.type.dtype


def test_duplicate_vars():
    with pytest.raises(ValueError) as err:
        with pm.Model():
            pm.Normal("a")
            pm.Normal("a")
    err.match("already exists")

    with pytest.raises(ValueError) as err:
        with pm.Model():
            pm.Normal("a")
            pm.Normal("a", transform=transforms.log)
    err.match("already exists")

    with pytest.raises(ValueError) as err:
        with pm.Model():
            a = pm.Normal("a")
            pm.Potential("a", a**2)
    err.match("already exists")

    with pytest.raises(ValueError) as err:
        with pm.Model():
            pm.Binomial("a", 10, 0.5)
            pm.Normal("a", transform=transforms.log)
    err.match("already exists")


def test_empty_observed():
    pd = pytest.importorskip("pandas")
    data = pd.DataFrame(np.ones((2, 3)) / 3)
    data.values[:] = np.nan
    with pm.Model():
        a = pm.Normal("a", observed=data)
        assert not hasattr(a.tag, "observations")


class TestValueGradFunction(unittest.TestCase):
    def test_no_extra(self):
        a = pt.vector("a")
        a.tag.test_value = np.zeros(3, dtype=a.dtype)
        f_grad = ValueGradFunction([a.sum()], [a], {}, mode="FAST_COMPILE")
        assert f_grad._extra_vars == []

    def test_invalid_type(self):
        a = pt.ivector("a")
        a.tag.test_value = np.zeros(3, dtype=a.dtype)
        a.dshape = (3,)
        a.dsize = 3
        with pytest.raises(TypeError) as err:
            ValueGradFunction([a.sum()], [a], {}, mode="FAST_COMPILE")
        err.match("Invalid dtype")

    def setUp(self):
        extra1 = pt.iscalar("extra1")
        extra1_ = np.array(0, dtype=extra1.dtype)
        extra1.dshape = tuple()
        extra1.dsize = 1

        val1 = pt.vector("val1")
        val1_ = np.zeros(3, dtype=val1.dtype)
        val1.dshape = (3,)
        val1.dsize = 3

        val2 = pt.matrix("val2")
        val2_ = np.zeros((2, 3), dtype=val2.dtype)
        val2.dshape = (2, 3)
        val2.dsize = 6

        self.val1, self.val1_ = val1, val1_
        self.val2, self.val2_ = val2, val2_
        self.extra1, self.extra1_ = extra1, extra1_

        self.cost = extra1 * val1.sum() + val2.sum()

        self.f_grad = ValueGradFunction(
            [self.cost], [val1, val2], {extra1: extra1_}, mode="FAST_COMPILE"
        )

    def test_extra_not_set(self):
        with pytest.raises(ValueError) as err:
            self.f_grad.get_extra_values()
        err.match("Extra values are not set")

        with pytest.raises(ValueError) as err:
            size = self.val1_.size + self.val2_.size
            self.f_grad(np.zeros(size, dtype=self.f_grad.dtype))
        err.match("Extra values are not set")

    def test_grad(self):
        self.f_grad.set_extra_values({"extra1": 5})
        size = self.val1_.size + self.val2_.size
        array = RaveledVars(
            np.ones(size, dtype=self.f_grad.dtype),
            (
                ("val1", self.val1_.shape, self.val1_.dtype),
                ("val2", self.val2_.shape, self.val2_.dtype),
            ),
        )
        val, grad = self.f_grad(array)
        assert val == 21
        npt.assert_allclose(grad, [5, 5, 5, 1, 1, 1, 1, 1, 1])

    @pytest.mark.xfail(reason="Test not refactored for v4")
    def test_edge_case(self):
        # Edge case discovered in #2948
        ndim = 3
        with pm.Model() as m:
            pm.LogNormal(
                "sigma", mu=np.zeros(ndim), tau=np.ones(ndim), shape=ndim
            )  # variance for the correlation matrix
            pm.HalfCauchy("nu", beta=10)
            step = pm.NUTS()

        func = step._logp_dlogp_func
        initial_point = m.initial_point()
        func.set_extra_values(initial_point)
        q = func.dict_to_array(initial_point)
        logp, dlogp = func(q)
        assert logp.size == 1
        assert dlogp.size == 4
        npt.assert_allclose(dlogp, 0.0, atol=1e-5)

    def test_missing_data(self):
        # Originally from a case described in #3122
        X = np.random.binomial(1, 0.5, 10)
        X[0] = -1  # masked a single value
        X = np.ma.masked_values(X, value=-1)
        with pm.Model() as m:
            x1 = pm.Uniform("x1", 0.0, 1.0)
            with pytest.warns(ImputationWarning):
                x2 = pm.Bernoulli("x2", x1, observed=X)

        gf = m.logp_dlogp_function()
        gf._extra_are_set = True

        assert m["x2_unobserved"].type == gf._extra_vars_shared["x2_unobserved"].type

        # The dtype of the merged observed/missing deterministic should match the RV dtype
        assert m.deterministics[0].type.dtype == x2.type.dtype

        point = m.initial_point(random_seed=None).copy()
        del point["x2_unobserved"]

        res = [gf(DictToArrayBijection.map(Point(point, model=m))) for i in range(5)]

        # Assert that all the elements of res are equal
        assert res[1:] == res[:-1]

    def test_pytensor_switch_broadcast_edge_cases_1(self):
        # Tests against two subtle issues related to a previous bug in Theano
        # where `tt.switch` would not always broadcast tensors with single
        # values https://github.com/pymc-devs/pytensor/issues/270

        # Known issue 1: https://github.com/pymc-devs/pymc/issues/4389
        data = pm.floatX(np.zeros(10))
        with pm.Model() as m:
            p = pm.Beta("p", 1, 1)
            obs = pm.Bernoulli("obs", p=p, observed=data)

        npt.assert_allclose(
            m.compile_logp(obs)({"p_logodds__": pm.floatX(np.array(0.0))}),
            np.log(0.5) * 10,
        )

    def test_pytensor_switch_broadcast_edge_cases_2(self):
        # Known issue 2: https://github.com/pymc-devs/pymc/issues/4417
        # fmt: off
        data = np.array([
            1.35202174, -0.83690274, 1.11175166, 1.29000367, 0.21282749,
            0.84430966, 0.24841369, 0.81803141, 0.20550244, -0.45016253,
        ])
        # fmt: on
        with pm.Model() as m:
            mu = pm.Normal("mu", 0, 5)
            obs = pm.TruncatedNormal("obs", mu=mu, sigma=1, lower=-1, upper=2, observed=data)

        npt.assert_allclose(m.compile_dlogp(mu)({"mu": 0}), 2.499424682024436, rtol=1e-5)


def test_multiple_observed_rv():
    "Test previously buggy multi-observed RV comparison code."
    y1_data = np.random.randn(10)
    y2_data = np.random.randn(100)
    with pm.Model() as model:
        mu = pm.Normal("mu")
        x = pm.CustomDist(  # pylint: disable=unused-variable
            "x", mu, logp=lambda value, mu: pm.Normal.logp(value, mu, 1.0), observed=0.1
        )
    assert not model["x"] == model["mu"]
    assert model["x"] == model["x"]
    assert model["x"] in model.observed_RVs
    assert model["x"] not in model.value_vars


def test_tempered_logp_dlogp():
    with pm.Model() as model:
        pm.Normal("x")
        pm.Normal("y", observed=1)
        pm.Potential("z", pt.constant(-1.0, dtype=pytensor.config.floatX))

    func = model.logp_dlogp_function()
    func.set_extra_values({})

    func_temp = model.logp_dlogp_function(tempered=True)
    func_temp.set_extra_values({})

    func_nograd = model.logp_dlogp_function(compute_grads=False)
    func_nograd.set_extra_values({})

    func_temp_nograd = model.logp_dlogp_function(tempered=True, compute_grads=False)
    func_temp_nograd.set_extra_values({})

    x = np.ones(1, dtype=func.dtype)
    npt.assert_allclose(func(x)[0], func_temp(x)[0])
    npt.assert_allclose(func(x)[1], func_temp(x)[1])

    npt.assert_allclose(func_nograd(x), func(x)[0])
    npt.assert_allclose(func_temp_nograd(x), func(x)[0])

    func_temp.set_weights(np.array([0.0], dtype=func.dtype))
    func_temp_nograd.set_weights(np.array([0.0], dtype=func.dtype))
    npt.assert_allclose(func(x)[0], 2 * func_temp(x)[0] - 1)
    npt.assert_allclose(func(x)[1], func_temp(x)[1])

    npt.assert_allclose(func_nograd(x), func(x)[0])
    npt.assert_allclose(func_temp_nograd(x), func_temp(x)[0])

    func_temp.set_weights(np.array([0.5], dtype=func.dtype))
    func_temp_nograd.set_weights(np.array([0.5], dtype=func.dtype))
    npt.assert_allclose(func(x)[0], 4 / 3 * (func_temp(x)[0] - 1 / 4))
    npt.assert_allclose(func(x)[1], func_temp(x)[1])

    npt.assert_allclose(func_nograd(x), func(x)[0])
    npt.assert_allclose(func_temp_nograd(x), func_temp(x)[0])


class TestPickling:
    def test_model_pickle(self, tmpdir):
        """Tests that PyMC models are pickleable"""
        with pm.Model() as model:
            x = pm.Normal("x")
            pm.Normal("y", observed=1)

        cloudpickle.loads(cloudpickle.dumps(model))

    def test_model_pickle_deterministic(self, tmpdir):
        """Tests that PyMC models are pickleable"""
        with pm.Model() as model:
            x = pm.Normal("x")
            z = pm.Normal("z")
            pm.Deterministic("w", x / z)
            pm.Normal("y", observed=1)

        cloudpickle.loads(cloudpickle.dumps(model))

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


def test_model_value_vars():
    with pm.Model() as model:
        a = pm.Normal("a")
        pm.Normal("x", a)

    value_vars = model.value_vars
    assert len(value_vars) == 2
    assert set(value_vars) == set(pm.inputvars(model.logp()))


def test_model_var_maps():
    with pm.Model() as model:
        a = pm.Uniform("a")
        x = pm.Normal("x", a)

    assert set(model.rvs_to_values.keys()) == {a, x}
    a_value = model.rvs_to_values[a]
    x_value = model.rvs_to_values[x]
    assert a_value.owner is None
    assert x_value.owner is None
    assert model.values_to_rvs == {a_value: a, x_value: x}

    assert set(model.rvs_to_transforms.keys()) == {a, x}
    assert isinstance(model.rvs_to_transforms[a], IntervalTransform)
    assert model.rvs_to_transforms[x] is None


def test_make_obs_var():
    """
    Check returned values for `data` given known inputs to `as_tensor()`.

    Note that ndarrays should return a TensorConstant and sparse inputs
    should return a Sparse PyTensor object.
    """
    # Create the various inputs to the function
    input_name = "testing_inputs"
    sparse_input = sps.csr_matrix(np.eye(3))
    dense_input = np.arange(9).reshape((3, 3))
    masked_array_input = ma.array(dense_input, mask=(np.mod(dense_input, 2) == 0))

    # Create a fake model and fake distribution to be used for the test
    fake_model = pm.Model()
    with fake_model:
        fake_distribution = pm.Normal.dist(mu=0, sigma=1, size=(3, 3))
        # Create the testval attribute simply for the sake of model testing
        fake_distribution.name = input_name

    # The function requires data and RV dimensionality to be compatible
    with pytest.raises(ShapeError, match="Dimensionality of data and RV don't match."):
        fake_model.make_obs_var(fake_distribution, np.ones((3, 3, 1)), None, None, None)

    # Check function behavior using the various inputs
    # dense, sparse: Ensure that the missing values are appropriately set to None
    # masked: a deterministic variable is returned

    dense_output = fake_model.make_obs_var(fake_distribution, dense_input, None, None, None)
    assert dense_output == fake_distribution
    assert isinstance(fake_model.rvs_to_values[dense_output], TensorConstant)
    del fake_model.named_vars[fake_distribution.name]

    sparse_output = fake_model.make_obs_var(fake_distribution, sparse_input, None, None, None)
    assert sparse_output == fake_distribution
    assert sparse.basic._is_sparse_variable(fake_model.rvs_to_values[sparse_output])
    del fake_model.named_vars[fake_distribution.name]

    # Here the RandomVariable is split into observed/imputed and a Deterministic is returned
    with pytest.warns(ImputationWarning):
        masked_output = fake_model.make_obs_var(
            fake_distribution, masked_array_input, None, None, None
        )
    assert masked_output != fake_distribution
    assert not isinstance(masked_output, RandomVariable)
    # Ensure it has missing values
    assert {"testing_inputs_unobserved"} == {v.name for v in fake_model.value_vars}
    assert {"testing_inputs", "testing_inputs_observed"} == {
        v.name for v in fake_model.observed_RVs
    }
    del fake_model.named_vars[fake_distribution.name]

    # Test that setting total_size returns a MinibatchRandomVariable
    scaled_outputs = fake_model.make_obs_var(
        fake_distribution, dense_input, None, None, total_size=100
    )
    assert scaled_outputs != fake_distribution
    assert isinstance(scaled_outputs.owner.op, MinibatchRandomVariable)
    del fake_model.named_vars[fake_distribution.name]


def test_initial_point():
    with pm.Model() as model:
        a = pm.Uniform("a")
        x = pm.Normal("x", a)

    b_initval = np.array(0.3, dtype=pytensor.config.floatX)

    with pytest.warns(FutureWarning), model:
        b = pm.Uniform("b", testval=b_initval)

    b_initval_trans = model.rvs_to_transforms[b].forward(b_initval, *b.owner.inputs).eval()

    y_initval = np.array(-2.4, dtype=pytensor.config.floatX)

    with model:
        y = pm.Normal("y", initval=y_initval)

    assert a in model.rvs_to_initial_values
    assert x in model.rvs_to_initial_values
    assert model.rvs_to_initial_values[b] == b_initval
    assert model.initial_point(0)["b_interval__"] == b_initval_trans
    assert model.rvs_to_initial_values[y] == y_initval


def test_point_logps():
    with pm.Model() as model:
        a = pm.Uniform("a")
        pm.Normal("x", a)

    logp_vals = model.point_logps()

    assert "x" in logp_vals.keys()
    assert "a" in logp_vals.keys()


def test_point_logps_potential():
    with pm.Model() as model:
        x = pm.Flat("x", initval=1)
        y = pm.Potential("y", x * 2)

    logps = model.point_logps()
    assert np.isclose(logps["y"], 2)


class TestShapeEvaluation:
    def test_eval_rv_shapes(self):
        with pm.Model(
            coords={
                "city": ["Sydney", "Las Vegas", "Düsseldorf"],
            }
        ) as pmodel:
            pm.MutableData("budget", [1, 2, 3, 4], dims="year")
            pm.Normal("untransformed", size=(1, 2))
            pm.Uniform("transformed", size=(7,))
            obs = pm.Uniform("observed", size=(3,), observed=[0.1, 0.2, 0.3])
            pm.LogNormal("lognorm", mu=pt.log(obs))
            pm.Normal("from_dims", dims=("city", "year"))
        shapes = pmodel.eval_rv_shapes()
        assert shapes["untransformed"] == (1, 2)
        assert shapes["transformed"] == (7,)
        assert shapes["transformed_interval__"] == (7,)
        assert shapes["lognorm"] == (3,)
        assert shapes["lognorm_log__"] == (3,)
        assert shapes["from_dims"] == (3, 4)


class TestCheckStartVals:
    def test_valid_start_point(self):
        with pm.Model() as model:
            a = pm.Uniform("a", lower=0.0, upper=1.0)
            b = pm.Uniform("b", lower=2.0, upper=3.0)

        start = {
            "a_interval__": model.rvs_to_transforms[a].forward(0.3, *a.owner.inputs).eval(),
            "b_interval__": model.rvs_to_transforms[b].forward(2.1, *b.owner.inputs).eval(),
        }
        model.check_start_vals(start)

    def test_invalid_start_point(self):
        with pm.Model() as model:
            a = pm.Uniform("a", lower=0.0, upper=1.0)
            b = pm.Uniform("b", lower=2.0, upper=3.0)

        start = {
            "a_interval__": np.nan,
            "b_interval__": model.rvs_to_transforms[b].forward(2.1, *b.owner.inputs).eval(),
        }
        with pytest.raises(pm.exceptions.SamplingError):
            model.check_start_vals(start)

    def test_invalid_variable_name(self):
        with pm.Model() as model:
            a = pm.Uniform("a", lower=0.0, upper=1.0)
            b = pm.Uniform("b", lower=2.0, upper=3.0)

        start = {
            "a_interval__": model.rvs_to_transforms[a].forward(0.3, *a.owner.inputs).eval(),
            "b_interval__": model.rvs_to_transforms[b].forward(2.1, *b.owner.inputs).eval(),
            "c": 1.0,
        }
        with pytest.raises(KeyError):
            model.check_start_vals(start)


def test_set_initval():
    # Make sure the dependencies between variables are maintained when
    # generating initial values
    rng = np.random.RandomState(392)

    with pm.Model() as model:
        eta = pm.Uniform("eta", 1.0, 2.0, size=(1, 1))
        mu = pm.Normal("mu", sigma=eta, initval=[[100]])
        alpha = pm.HalfNormal("alpha", initval=100)
        value = pm.NegativeBinomial("value", mu=mu, alpha=alpha)

    assert np.array_equal(model.initial_values[mu], np.array([[100.0]]))
    np.testing.assert_array_equal(model.initial_values[alpha], np.array(100))
    assert model.initial_values[value] is None

    # `Flat` cannot be sampled, so let's make sure that doesn't break initial
    # value computations
    with pm.Model() as model:
        x = pm.Flat("x")
        y = pm.Normal("y", x, 1)

    assert y in model.initial_values


def test_datalogp_multiple_shapes():
    with pm.Model() as m:
        x = pm.Normal("x", 0, 1)
        z1 = pm.Potential("z1", x)
        z2 = pm.Potential("z2", pt.full((1, 3), x))
        y1 = pm.Normal("y1", x, 1, observed=np.array([1]))
        y2 = pm.Normal("y2", x, 1, observed=np.array([1, 2]))
        y3 = pm.Normal("y3", x, 1, observed=np.array([1, 2, 3]))

    # This would raise a TypeError, see #4803 and #4804
    x_val = m.rvs_to_values[x]
    m.datalogp.eval({x_val: 0})


def test_nested_model_coords():
    with pm.Model(name="m1", coords=dict(dim1=range(2))) as m1:
        a = pm.Normal("a", dims="dim1")
        with pm.Model(name="m2", coords=dict(dim2=range(4))) as m2:
            b = pm.Normal("b", dims="dim1")
            m1.add_coord("dim3", range(4))
            c = pm.HalfNormal("c", dims="dim3")
            d = pm.Normal("d", b, c, dims="dim2")
        e = pm.Normal("e", a[None] + d[:, None], dims=("dim2", "dim1"))
    assert m1.coords is m2.coords
    assert m1.dim_lengths is m2.dim_lengths
    assert set(m2.named_vars_to_dims) < set(m1.named_vars_to_dims)


def test_shapeerror_from_set_data_dimensionality():
    with pm.Model() as pmodel:
        pm.MutableData("m", np.ones((3,)), dims="one")
        with pytest.raises(ValueError, match="must have 1 dimensions"):
            pmodel.set_data("m", np.ones((3, 4)))


def test_shapeerror_from_resize_immutable_dim_from_RV():
    """
    Trying to resize an immutable dimension should raise a ShapeError.
    Even if the variable being updated is a SharedVariable and has other
    dimensions that are mutable.
    """
    with pm.Model(coords={"fixed": range(3)}) as pmodel:
        pm.Normal("a", mu=[1, 2, 3], dims="fixed")
        assert isinstance(pmodel.dim_lengths["fixed"], TensorVariable)

        pm.MutableData("m", [[1, 2, 3]], dims=("one", "fixed"))

        # This is fine because the "fixed" dim is not resized
        pmodel.set_data("m", [[1, 2, 3], [3, 4, 5]])

    msg = "The 'm' variable already had 3 coord values defined for its fixed dimension"
    with pytest.raises(ValueError, match=msg):
        # Can't work because the "fixed" dimension is linked to a
        # TensorVariable with constant shape.
        # Note that the new data tries to change both dimensions
        pmodel.set_data("m", [[1, 2], [3, 4]])


def test_shapeerror_from_resize_immutable_dim_from_coords():
    with pm.Model(coords={"immutable": [1, 2]}) as pmodel:
        assert isinstance(pmodel.dim_lengths["immutable"], TensorConstant)
        pm.MutableData("m", [1, 2], dims="immutable")
        # Data can be changed
        pmodel.set_data("m", [3, 4])

    with pytest.raises(ShapeError, match="`TensorConstant` stores its length"):
        # But the length is linked to a TensorConstant
        pmodel.set_data("m", [1, 2, 3], coords=dict(immutable=[1, 2, 3]))


def test_valueerror_from_resize_without_coords_update():
    """
    Resizing a mutable dimension that had coords,
    without passing new coords raises a ValueError.
    """
    with pm.Model() as pmodel:
        pmodel.add_coord("shared", [1, 2, 3], mutable=True)
        pm.MutableData("m", [1, 2, 3], dims=("shared"))
        with pytest.raises(ValueError, match="'m' variable already had 3"):
            # tries to resize m but without passing coords so raise ValueError
            pm.set_data({"m": [1, 2, 3, 4]})


def test_coords_and_constantdata_create_immutable_dims():
    """
    When created from `pm.Model(coords=...)` or `pm.ConstantData`
    a dimension should be resizable.
    """
    with pm.Model(coords={"group": ["A", "B"]}) as m:
        x = pm.ConstantData("x", [0], dims="feature")
        y = pm.Normal("y", x, 1, dims=("group", "feature"))
    assert isinstance(m._dim_lengths["feature"], TensorConstant)
    assert isinstance(m._dim_lengths["group"], TensorConstant)
    assert x.eval().shape == (1,)
    assert y.eval().shape == (2, 1)


def test_add_coord_mutable_kwarg():
    """
    Checks resulting tensor type depending on mutable kwarg in add_coord.
    """
    with pm.Model() as m:
        m.add_coord("fixed", values=[1], mutable=False)
        m.add_coord("mutable1", values=[1, 2], mutable=True)
        assert isinstance(m._dim_lengths["fixed"], TensorConstant)
        assert isinstance(m._dim_lengths["mutable1"], ScalarSharedVariable)
        pm.MutableData("mdata", np.ones((1, 2, 3)), dims=("fixed", "mutable1", "mutable2"))
        assert isinstance(m._dim_lengths["mutable2"], TensorVariable)


def test_set_dim():
    """Test the conscious re-sizing of dims created through add_coord()."""
    with pm.Model() as pmodel:
        pmodel.add_coord("fdim", mutable=False, length=1)
        pmodel.add_coord("mdim", mutable=True, length=2)
        a = pm.Normal("a", dims="mdim")
    assert a.eval().shape == (2,)

    with pytest.raises(ValueError, match="is immutable"):
        pmodel.set_dim("fdim", 3)

    pmodel.set_dim("mdim", 3)
    assert a.eval().shape == (3,)


def test_set_dim_with_coords():
    """Test the conscious re-sizing of dims created through add_coord() with coord value."""
    with pm.Model() as pmodel:
        pmodel.add_coord("mdim", mutable=True, length=2, values=["A", "B"])
        a = pm.Normal("a", dims="mdim")
    assert len(pmodel.coords["mdim"]) == 2

    with pytest.raises(ValueError, match="has coord values"):
        pmodel.set_dim("mdim", new_length=3)

    with pytest.raises(ShapeError, match="does not match"):
        pmodel.set_dim("mdim", new_length=3, coord_values=["A", "B"])

    pmodel.set_dim("mdim", 3, ["A", "B", "C"])
    assert a.eval().shape == (3,)
    assert pmodel.coords["mdim"] == ("A", "B", "C")


def test_add_named_variable_checks_dim_name():
    with pm.Model() as pmodel:
        rv = pm.Normal.dist(mu=[1, 2])

        # Checks that vars are named
        with pytest.raises(ValueError, match="is unnamed"):
            pmodel.add_named_variable(rv)
        rv.name = "nomnom"

        # Coords must be available already
        with pytest.raises(ValueError, match="not specified in `coords`"):
            pmodel.add_named_variable(rv, dims="nomnom")
        pmodel.add_coord("nomnom", [1, 2])

        # No name collisions
        with pytest.raises(ValueError, match="same name as"):
            pmodel.add_named_variable(rv, dims="nomnom")

        # This should work (regression test against #6335)
        rv2 = rv[:, None]
        rv2.name = "yumyum"
        pmodel.add_named_variable(rv2, dims=("nomnom", None))


def test_dims_type_check():
    with pm.Model(coords={"a": range(5)}) as m:
        with pytest.raises(TypeError, match="Dims must be string"):
            x = pm.Normal("x", shape=(10, 5), dims=(None, "a"))


def test_none_coords_autonumbering():
    with pm.Model() as m:
        m.add_coord(name="a", values=None, length=3)
        m.add_coord(name="b", values=range(5))
        x = pm.Normal("x", dims=("a", "b"))
        prior = pm.sample_prior_predictive(samples=2).prior
    assert prior["x"].shape == (1, 2, 3, 5)
    assert list(prior.coords["a"].values) == list(range(3))
    assert list(prior.coords["b"].values) == list(range(5))


def test_set_data_indirect_resize():
    with pm.Model() as pmodel:
        pmodel.add_coord("mdim", mutable=True, length=2)
        pm.MutableData("mdata", [1, 2], dims="mdim")

    # First resize the dimension.
    pmodel.dim_lengths["mdim"].set_value(3)
    # Then change the data.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pmodel.set_data("mdata", [1, 2, 3])

    # Now the other way around.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pmodel.set_data("mdata", [1, 2, 3, 4])


def test_set_data_warns_on_resize_of_dims_defined_by_other_mutabledata():
    with pm.Model() as pmodel:
        pm.MutableData("m1", [1, 2], dims="mutable")
        pm.MutableData("m2", [3, 4], dims="mutable")

        # Resizing the non-defining variable first gives a warning
        with pytest.warns(ShapeWarning, match="by another variable"):
            pmodel.set_data("m2", [4, 5, 6])
            pmodel.set_data("m1", [1, 2, 3])

        # Resizing the definint variable first is silent
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pmodel.set_data("m1", [1, 2])
            pmodel.set_data("m2", [3, 4])


def test_set_data_indirect_resize_with_coords():
    with pm.Model() as pmodel:
        pmodel.add_coord("mdim", ["A", "B"], mutable=True, length=2)
        pm.MutableData("mdata", [1, 2], dims="mdim")

    assert pmodel.coords["mdim"] == ("A", "B")

    # First resize the dimension.
    pmodel.set_dim("mdim", 3, ["A", "B", "C"])
    assert pmodel.coords["mdim"] == ("A", "B", "C")
    # Then change the data.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pmodel.set_data("mdata", [1, 2, 3])

    # Now the other way around.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pmodel.set_data("mdata", [1, 2, 3, 4], coords=dict(mdim=["A", "B", "C", "D"]))
    assert pmodel.coords["mdim"] == ("A", "B", "C", "D")

    # This time with incorrectly sized coord values
    with pytest.raises(ShapeError, match="new coordinate values"):
        pmodel.set_data("mdata", [1, 2], coords=dict(mdim=[1, 2, 3]))


def test_set_data_constant_shape_error():
    with pm.Model() as pmodel:
        x = pm.Normal("x", size=7)
        pmodel.add_coord("weekday", length=x.shape[0])
        pm.MutableData("y", np.arange(7), dims="weekday")

    msg = "because the dimension was initialized from 'x'"
    with pytest.raises(ShapeError, match=msg):
        pmodel.set_data("y", np.arange(10))


def test_model_deprecation_warning():
    with pm.Model() as m:
        x = pm.Normal("x", 0, 1, size=2)
        y = pm.LogNormal("y", 0, 1, size=2)

    with pytest.warns(FutureWarning):
        m.disc_vars

    with pytest.warns(FutureWarning):
        m.cont_vars


@pytest.mark.parametrize("jacobian", [True, False])
def test_model_logp(jacobian):
    with pm.Model() as m:
        x = pm.Normal("x", 0, 1, size=2)
        y = pm.LogNormal("y", 0, 1, size=2)

    test_vals = np.array([0.0, 1.0])

    expected_x_logp = st.norm().logpdf(test_vals)
    expected_y_logp = expected_x_logp.copy()
    if not jacobian:
        expected_y_logp -= np.array([0.0, 1.0])

    x_logp, y_logp = m.compile_logp(sum=False, jacobian=jacobian)(
        {"x": test_vals, "y_log__": test_vals}
    )
    assert np.all(np.isclose(x_logp, expected_x_logp))
    assert np.all(np.isclose(y_logp, expected_y_logp))

    x_logp2 = m.compile_logp(vars=[x], sum=False, jacobian=jacobian)({"x": test_vals})
    assert np.all(np.isclose(x_logp2, expected_x_logp))

    y_logp2 = m.compile_logp(vars=[y], sum=False, jacobian=jacobian)({"y_log__": test_vals})
    assert np.all(np.isclose(y_logp2, expected_y_logp))

    logp_sum = m.compile_logp(sum=True, jacobian=jacobian)({"x": test_vals, "y_log__": test_vals})
    assert np.isclose(logp_sum, expected_x_logp.sum() + expected_y_logp.sum())


@pytest.mark.parametrize("jacobian", [True, False])
def test_model_dlogp(jacobian):
    with pm.Model() as m:
        x = pm.Normal("x", 0, 1, size=2)
        y = pm.LogNormal("y", 0, 1, size=2)

    test_vals = np.array([0.0, -1.0])
    state = {"x": test_vals, "y_log__": test_vals}

    expected_x_dlogp = expected_y_dlogp = np.array([0.0, 1.0])
    if not jacobian:
        expected_y_dlogp = np.array([-1.0, 0.0])

    dlogps = m.compile_dlogp(jacobian=jacobian)(state)
    assert np.all(np.isclose(dlogps[:2], expected_x_dlogp))
    assert np.all(np.isclose(dlogps[2:], expected_y_dlogp))

    x_dlogp2 = m.compile_dlogp(vars=[x], jacobian=jacobian)(state)
    assert np.all(np.isclose(x_dlogp2, expected_x_dlogp))

    y_dlogp2 = m.compile_dlogp(vars=[y], jacobian=jacobian)(state)
    assert np.all(np.isclose(y_dlogp2, expected_y_dlogp))


@pytest.mark.parametrize("jacobian", [True, False])
def test_model_d2logp(jacobian):
    with pm.Model() as m:
        x = pm.Normal("x", 0, 1, size=2)
        y = pm.LogNormal("y", 0, 1, size=2)

    test_vals = np.array([0.0, -1.0])
    state = {"x": test_vals, "y_log__": test_vals}

    expected_x_d2logp = expected_y_d2logp = np.eye(2)

    dlogps = m.compile_d2logp(jacobian=jacobian)(state)
    assert np.all(np.isclose(dlogps[:2, :2], expected_x_d2logp))
    assert np.all(np.isclose(dlogps[2:, 2:], expected_y_d2logp))

    x_dlogp2 = m.compile_d2logp(vars=[x], jacobian=jacobian)(state)
    assert np.all(np.isclose(x_dlogp2, expected_x_d2logp))

    y_dlogp2 = m.compile_d2logp(vars=[y], jacobian=jacobian)(state)
    assert np.all(np.isclose(y_dlogp2, expected_y_d2logp))


def test_deterministic():
    with pm.Model() as model:
        x = pm.Normal("x", 0, 1)
        y = pm.Deterministic("y", x**2)

    assert model.y == y
    assert model["y"] == y


def test_determinsitic_with_dims():
    """
    Test to check the passing of dims to the potential
    """
    with pm.Model(coords={"observed": range(10)}) as model:
        x = pm.Normal("x", 0, 1)
        y = pm.Deterministic("y", x**2, dims=("observed",))
    assert model.named_vars_to_dims == {"y": ("observed",)}


def test_potential_with_dims():
    """
    Test to check the passing of dims to the potential
    """
    with pm.Model(coords={"observed": range(10)}) as model:
        x = pm.Normal("x", 0, 1)
        y = pm.Potential("y", x**2, dims=("observed",))
    assert model.named_vars_to_dims == {"y": ("observed",)}


def test_empty_model_representation():
    assert pm.Model().str_repr() == ""


def test_compile_fn():
    with pm.Model() as m:
        x = pm.Normal("x", 0, 1, size=2)
        y = pm.LogNormal("y", 0, 1, size=2)

    test_vals = np.array([0.0, -1.0])
    state = {"x": test_vals, "y": test_vals}

    with m:
        func = pm.compile_fn(x + y, inputs=[x, y])
        result_compute = func(state)

    func = m.compile_fn(x + y, inputs=[x, y])
    result_expect = func(state)

    np.testing.assert_allclose(result_compute, result_expect)


def test_model_pytensor_config():
    assert pytensor.config.mode != "JAX"
    with pm.Model(pytensor_config=dict(mode="JAX")) as model:
        assert pytensor.config.mode == "JAX"
    assert pytensor.config.mode != "JAX"


def test_model_parent_set_programmatically():
    with pm.Model() as model:
        x = pm.Normal("x")

    with pm.Model(model=model):
        y = pm.Normal("y")

    assert "y" in model.named_vars


class TestModelContext:
    def test_thread_safety(self):
        """Regression test for issue #1552: Thread safety of model context manager

        This test creates two threads that attempt to construct two
        unrelated models at the same time.
        For repeatable testing, the two threads are synchronised such
        that thread A enters the context manager first, then B,
        then A attempts to declare a variable while B is still in the context manager.
        """
        aInCtxt, bInCtxt, aDone = (threading.Event() for _ in range(3))
        modelA = pm.Model()
        modelB = pm.Model()

        def make_model_a():
            with modelA:
                aInCtxt.set()
                bInCtxt.wait()
                Normal("a", 0, 1)
            aDone.set()

        def make_model_b():
            aInCtxt.wait()
            with modelB:
                bInCtxt.set()
                aDone.wait()
                Normal("b", 0, 1)

        threadA = threading.Thread(target=make_model_a)
        threadB = threading.Thread(target=make_model_b)
        threadA.start()
        threadB.start()
        threadA.join()
        threadB.join()
        # now let's see which model got which variable
        # previous to #1555, the variables would be swapped:
        # - B enters it's model context after A, but before a is declared -> a goes into B
        # - A leaves it's model context before B attempts to declare b. A's context manager
        #   takes B from the stack, such that b ends up in model A
        assert (
            list(modelA.named_vars),
            list(modelB.named_vars),
        ) == (["a"], ["b"])


def test_mixed_contexts():
    modelA = pm.Model()
    modelB = pm.Model()
    with pytest.raises((ValueError, TypeError)):
        modelcontext(None)
    with modelA:
        with modelB:
            assert pm.Model.get_context() == modelB
            assert modelcontext(None) == modelB
        assert pm.Model.get_context() == modelA
        assert modelcontext(None) == modelA
    assert pm.Model.get_context(error_if_none=False) is None
    with pytest.raises(TypeError):
        pm.Model.get_context(error_if_none=True)
    with pytest.raises((ValueError, TypeError)):
        modelcontext(None)


class TestProfile:
    def setup_method(self):
        _, self.model, _ = simple_model()

    def test_profile_model(self):
        assert self.model.profile(self.model.logp(), n=3000).fct_call_time > 0

    def test_profile_variable(self):
        rv = self.model.basic_RVs[0]
        assert self.model.profile(self.model.logp(vars=[rv], sum=False), n=3000).fct_call_time > 0

    def test_profile_count(self):
        count = 1005
        assert self.model.profile(self.model.logp(), n=count).fct_callcount == count


@pytest.fixture(params=["masked", "pandas"])
def missing_data(request):
    if request.param == "masked":
        return np.ma.masked_values([1, 2, -1, 4, -1], value=-1)
    else:
        # request.param == "pandas"
        pd = pytest.importorskip("pandas")
        return pd.DataFrame([1, 2, np.nan, 4, np.nan])


class TestImputationMissingData:
    "Test for Missing Data imputation"

    def test_missing_basic(self, missing_data):
        with pm.Model() as model:
            x = pm.Normal("x", 1, 1)
            with pytest.warns(ImputationWarning):
                _ = pm.Normal("y", x, 1, observed=missing_data)

        assert "y_unobserved" in model.named_vars

        test_point = model.initial_point()
        assert not np.isnan(model.compile_logp()(test_point))

        with model:
            ipr = pm.sample_prior_predictive()
        assert {"x", "y"} <= set(ipr.prior.keys())

    def test_missing_with_predictors(self):
        predictors = np.array([0.5, 1, 0.5, 2, 0.3])
        data = np.ma.masked_values([1, 2, -1, 4, -1], value=-1)
        with pm.Model() as model:
            x = pm.Normal("x", 1, 1)
            with pytest.warns(ImputationWarning):
                y = pm.Normal("y", x * predictors, 1, observed=data)

        assert "y_unobserved" in model.named_vars

        test_point = model.initial_point()
        assert not np.isnan(model.compile_logp()(test_point))

        with model:
            ipr = pm.sample_prior_predictive()
        assert {"x", "y"} <= set(ipr.prior.keys())

    def test_missing_dual_observations(self):
        with pm.Model() as model:
            obs1 = np.ma.masked_values([1, 2, -1, 4, -1], value=-1)
            obs2 = np.ma.masked_values([-1, -1, 6, -1, 8], value=-1)
            beta1 = pm.Normal("beta1", 1, 1)
            beta2 = pm.Normal("beta2", 2, 1)
            latent = pm.Normal("theta", size=5)
            with pytest.warns(ImputationWarning):
                ovar1 = pm.Normal("o1", mu=beta1 * latent, observed=obs1)
            with pytest.warns(ImputationWarning):
                ovar2 = pm.Normal("o2", mu=beta2 * latent, observed=obs2)

            prior_trace = pm.sample_prior_predictive(return_inferencedata=False)
            assert {"beta1", "beta2", "theta", "o1", "o2"} <= set(prior_trace.keys())
            # TODO: Assert something
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
                trace = pm.sample(chains=1, tune=5, draws=50)

    def test_interval_missing_observations(self):
        rng = np.random.default_rng(1198)

        with pm.Model() as model:
            obs1 = np.ma.masked_values([1, 2, -1, 4, -1], value=-1)
            obs2 = np.ma.masked_values([-1, -1, 6, -1, 8], value=-1)

            with pytest.warns(ImputationWarning):
                theta1 = pm.Uniform("theta1", 0, 5, observed=obs1)
            with pytest.warns(ImputationWarning):
                theta2 = pm.Normal("theta2", mu=theta1, observed=obs2)

            assert isinstance(
                model.rvs_to_transforms[model["theta1_unobserved"]], IntervalTransform
            )
            assert model.rvs_to_transforms[model["theta1_observed"]] is None

            prior_trace = pm.sample_prior_predictive(random_seed=rng, return_inferencedata=False)
            assert set(prior_trace.keys()) == {
                "theta1",
                "theta1_observed",
                "theta1_unobserved",
                "theta2",
                "theta2_observed",
                "theta2_unobserved",
            }

            # Make sure the observed + missing combined deterministics have the
            # same shape as the original observations vectors
            assert prior_trace["theta1"].shape[-1] == obs1.shape[0]
            assert prior_trace["theta2"].shape[-1] == obs2.shape[0]

            # Make sure that the observed values are newly generated samples
            assert np.all(np.var(prior_trace["theta1_observed"], 0) > 0.0)
            assert np.all(np.var(prior_trace["theta2_observed"], 0) > 0.0)

            # Make sure the missing parts of the combined deterministic matches the
            # sampled missing and observed variable values
            assert (
                np.mean(prior_trace["theta1"][:, obs1.mask] - prior_trace["theta1_unobserved"])
                == 0.0
            )
            assert (
                np.mean(prior_trace["theta1"][:, ~obs1.mask] - prior_trace["theta1_observed"])
                == 0.0
            )
            assert (
                np.mean(prior_trace["theta2"][:, obs2.mask] - prior_trace["theta2_unobserved"])
                == 0.0
            )
            assert (
                np.mean(prior_trace["theta2"][:, ~obs2.mask] - prior_trace["theta2_observed"])
                == 0.0
            )

            trace = pm.sample(
                chains=1,
                draws=50,
                compute_convergence_checks=False,
                return_inferencedata=False,
                random_seed=rng,
            )
            assert set(trace.varnames) == {
                "theta1",
                "theta1_unobserved",
                "theta1_unobserved_interval__",
                "theta2",
                "theta2_unobserved",
            }

            # Make sure that the missing values are newly generated samples and that
            # the observed and deterministic match
            assert np.all(0 < trace["theta1_unobserved"].mean(0))
            assert np.all(0 < trace["theta2_unobserved"].mean(0))
            assert np.isclose(
                np.mean(trace["theta1"][:, obs1.mask] - trace["theta1_unobserved"]), 0
            )
            assert np.isclose(
                np.mean(trace["theta2"][:, obs2.mask] - trace["theta2_unobserved"]), 0
            )

            # Make sure that the observed values are unchanged
            assert np.allclose(np.var(trace["theta1"][:, ~obs1.mask], 0), 0.0)
            assert np.allclose(np.var(trace["theta2"][:, ~obs2.mask], 0), 0.0)
            np.testing.assert_array_equal(trace["theta1"][0][~obs1.mask], obs1[~obs1.mask])
            np.testing.assert_array_equal(trace["theta2"][0][~obs2.mask], obs1[~obs2.mask])

            pp_idata = pm.sample_posterior_predictive(trace, random_seed=rng)
            pp_trace = pp_idata.posterior_predictive.stack(sample=["chain", "draw"]).transpose(
                "sample", ...
            )
            assert set(pp_trace.keys()) == {
                "theta1",
                "theta1_observed",
                "theta2",
                "theta2_observed",
            }

            # Make sure that the observed values are newly generated samples and that
            # the observed and deterministic match
            assert np.all(np.var(pp_trace["theta1"], 0) > 0.0)
            assert np.all(np.var(pp_trace["theta2"], 0) > 0.0)
            assert np.isclose(
                np.mean(pp_trace["theta1"][:, ~obs1.mask] - pp_trace["theta1_observed"]), 0
            )
            assert np.isclose(
                np.mean(pp_trace["theta2"][:, ~obs2.mask] - pp_trace["theta2_observed"]), 0
            )

    def test_missing_logp1(self):
        with pm.Model(check_bounds=False) as m1:
            x = pm.Gamma("x", 1, 1, size=4)

        logp_val = m1.compile_logp()({"x_log__": np.array([0, 0, 0, 0])})
        assert logp_val == -4.0

        with pm.Model(check_bounds=False) as m2:
            with pytest.warns(ImputationWarning):
                x = pm.Gamma("x", 1, 1, observed=[1, 1, 1, np.nan])

        logp_val = m2.compile_logp()({"x_unobserved_log__": np.array([0])})
        assert logp_val == -4.0

    def test_missing_logp2(self):
        with pm.Model() as m:
            theta1 = pm.Normal("theta1", 0, 5, observed=[0, 1, 2, 3, 4])
            theta2 = pm.Normal("theta2", mu=theta1, observed=[0, 1, 2, 3, 4])
        m_logp = m.compile_logp()({})

        with pm.Model() as m_missing:
            with pytest.warns(ImputationWarning):
                theta1 = pm.Normal("theta1", 0, 5, observed=np.array([0, 1, np.nan, 3, np.nan]))
                theta2 = pm.Normal(
                    "theta2", mu=theta1, observed=np.array([np.nan, np.nan, 2, np.nan, 4])
                )
        m_missing_logp = m_missing.compile_logp()(
            {"theta1_unobserved": [2, 4], "theta2_unobserved": [0, 1, 3]}
        )

        assert m_logp == m_missing_logp

    def test_missing_multivariate_separable(self):
        with pm.Model() as m_miss:
            with pytest.warns(ImputationWarning):
                x = pm.Dirichlet(
                    "x",
                    a=[1, 2, 3],
                    observed=np.array([[0.3, 0.3, 0.4], [np.nan, np.nan, np.nan]]),
                )
        assert (m_miss["x_unobserved"].owner.op, pm.Dirichlet)
        assert (m_miss["x_observed"].owner.op, pm.Dirichlet)

        with pm.Model() as m_unobs:
            x = pm.Dirichlet("x", a=[1, 2, 3], shape=(1, 3))

        inp_vals = simplex.forward(np.array([[0.3, 0.3, 0.4]])).eval()
        np.testing.assert_allclose(
            m_miss.compile_logp(jacobian=False)({"x_unobserved_simplex__": inp_vals}),
            m_unobs.compile_logp(jacobian=False)({"x_simplex__": inp_vals}) * 2,
        )

    def test_missing_multivariate_unseparable(self):
        with pm.Model() as m_miss:
            with pytest.warns(ImputationWarning):
                x = pm.Dirichlet(
                    "x",
                    a=[1, 2, 3],
                    observed=np.array([[0.3, 0.3, np.nan], [np.nan, np.nan, 0.4]]),
                )

        assert isinstance(m_miss["x_unobserved"].owner.op, PartialObservedRV)
        assert isinstance(m_miss["x_observed"].owner.op, PartialObservedRV)

        inp_values = np.array([0.3, 0.3, 0.4])
        np.testing.assert_allclose(
            m_miss.compile_logp()({"x_unobserved": [0.4, 0.3, 0.3]}),
            st.dirichlet.logpdf(inp_values, [1, 2, 3]) * 2,
        )

    def test_missing_vector_parameter(self):
        with pm.Model() as m:
            with pytest.warns(ImputationWarning):
                x = pm.Normal(
                    "x",
                    np.array([-10, 10]),
                    0.1,
                    observed=np.array([[np.nan, 10], [-10, np.nan], [np.nan, np.nan]]),
                )
        x_draws = x.eval()
        assert x_draws.shape == (3, 2)
        assert np.all(x_draws[:, 0] < 0)
        assert np.all(x_draws[:, 1] > 0)
        assert np.isclose(
            m.compile_logp()({"x_unobserved": np.array([-10, 10, -10, 10])}),
            st.norm(scale=0.1).logpdf(0) * 6,
        )

    def test_missing_symmetric(self):
        """Check that logp works when partially observed variable have equal observed and
        unobserved dimensions.

        This would fail in a previous implementation because the two variables would be
        equivalent and one of them would be discarded during MergeOptimization while
        building the logp graph
        """
        with pm.Model() as m:
            with pytest.warns(ImputationWarning):
                x = pm.Gamma("x", alpha=3, beta=10, observed=np.array([1, np.nan]))

        x_obs_rv = m["x_observed"]
        x_obs_vv = m.rvs_to_values[x_obs_rv]

        x_unobs_rv = m["x_unobserved"]
        x_unobs_vv = m.rvs_to_values[x_unobs_rv]

        logp = transformed_conditional_logp(
            [x_obs_rv, x_unobs_rv],
            rvs_to_values={x_obs_rv: x_obs_vv, x_unobs_rv: x_unobs_vv},
            rvs_to_transforms={},
        )
        logp_inputs = list(graph_inputs(logp))
        assert x_obs_vv in logp_inputs
        assert x_unobs_vv in logp_inputs

    def test_dims(self):
        """Test that we don't propagate dims to the subcomponents of a partially
        observed RV

        See https://github.com/pymc-devs/pymc/issues/6177
        """
        data = np.array([np.nan] * 3 + [0] * 7)
        with pm.Model(coords={"observed": range(10)}) as model:
            with pytest.warns(ImputationWarning):
                x = pm.Normal("x", observed=data, dims=("observed",))
        assert model.named_vars_to_dims == {"x": ("observed",)}

    def test_symbolic_random_variable(self):
        data = np.array([np.nan] * 3 + [0] * 7)
        with pm.Model() as model:
            with pytest.warns(ImputationWarning):
                x = pm.Censored(
                    "x",
                    pm.Normal.dist(),
                    lower=0,
                    upper=10,
                    observed=data,
                )
        np.testing.assert_almost_equal(
            model.compile_logp()({"x_unobserved": [0] * 3}),
            st.norm.logcdf(0) * 10,
        )


class TestShared:
    def test_deterministic(self):
        with pm.Model() as model:
            data_values = np.array([0.5, 0.4, 5, 2])
            X = pytensor.shared(np.asarray(data_values, dtype=pytensor.config.floatX), borrow=True)
            pm.Normal("y", 0, 1, observed=X)
            assert np.all(
                np.isclose(model.compile_logp(sum=False)({}), st.norm().logpdf(data_values))
            )


def test_tag_future_warning_model():
    # Test no unexpected warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        model = pm.Model()

        x = pt.random.normal()
        x.tag.something_else = "5"
        x.tag.test_value = 0
        assert not isinstance(x.tag, _FutureWarningValidatingScratchpad)

        # Test that model changes the tag type, but copies existing contents
        x = model.register_rv(x, name="x", transform=log)
        assert isinstance(x.tag, _FutureWarningValidatingScratchpad)
        assert x.tag.something_else == "5"
        assert x.tag.test_value == 0

        # Test expected warnings
        with pytest.warns(FutureWarning, match="model.rvs_to_values"):
            x_value = x.tag.value_var

        assert isinstance(x_value.tag, _FutureWarningValidatingScratchpad)
        with pytest.warns(FutureWarning, match="model.rvs_to_transforms"):
            transform = x_value.tag.transform
        assert transform is log

        with pytest.raises(AttributeError):
            x.tag.observations

        # Cloning a node will keep the same tag type and contents
        y = x.owner.clone().default_output()
        assert y is not x
        assert y.tag is not x.tag
        assert isinstance(y.tag, _FutureWarningValidatingScratchpad)
        y = model.register_rv(y, name="y", observed=5)
        assert isinstance(y.tag, _FutureWarningValidatingScratchpad)

        # Test expected warnings
        with pytest.warns(FutureWarning, match="model.rvs_to_values"):
            y_value = y.tag.value_var
        with pytest.warns(FutureWarning, match="model.rvs_to_values"):
            y_obs = y.tag.observations
        assert y_value is y_obs
        assert y_value.eval() == 5

        assert isinstance(y_value.tag, _FutureWarningValidatingScratchpad)


class TestModelDebug:
    @pytest.mark.parametrize("fn", ("logp", "dlogp", "random"))
    def test_no_problems(self, fn, capfd):
        with pm.Model() as m:
            x = pm.Normal("x", [1, -1, 1])
        m.debug(fn=fn)

        out, _ = capfd.readouterr()
        assert out == "point={'x': array([ 1., -1.,  1.])}\n\nNo problems found\n"

    @pytest.mark.parametrize("fn", ("logp", "dlogp", "random"))
    def test_invalid_parameter(self, fn, capfd):
        with pm.Model() as m:
            x = pm.Normal("x", [1, -1, 1])
            y = pm.HalfNormal("y", tau=x)
        m.debug(fn=fn)

        out, _ = capfd.readouterr()
        if fn == "dlogp":
            # var dlogp is 0 or 1 without a likelihood
            assert "No problems found" in out
        else:
            assert "The parameters evaluate to:\n0: 0.0\n1: [ 1. -1.  1.]" in out
            if fn == "logp":
                assert "This does not respect one of the following constraints: sigma > 0" in out
            else:
                assert (
                    "The variable y random method raised the following exception: Domain error in arguments."
                    in out
                )

    @pytest.mark.parametrize("verbose", (True, False))
    @pytest.mark.parametrize("fn", ("logp", "dlogp", "random"))
    def test_invalid_parameter_cant_be_evaluated(self, fn, verbose, capfd):
        with pm.Model() as m:
            x = pm.Normal("x", [1, 1, 1])
            sigma = Assert(msg="x > 0")(pm.math.abs(x), (x > 0).all())
            y = pm.HalfNormal("y", sigma=sigma)
        m.debug(point={"x": [-1, -1, -1], "y_log__": [0, 0, 0]}, fn=fn, verbose=verbose)

        out, _ = capfd.readouterr()
        assert "{'x': [-1, -1, -1], 'y_log__': [0, 0, 0]}" in out
        assert "The parameters of the variable y cannot be evaluated: x > 0" in out
        verbose_str = "Apply node that caused the error:" in out
        assert verbose_str if verbose else not verbose_str

    def test_invalid_value(self, capfd):
        with pm.Model() as m:
            x = pm.Normal("x", [1, -1, 1])
            y = pm.HalfNormal("y", tau=pm.math.abs(x), initval=[-1, 1, -1], transform=None)
        m.debug()

        out, _ = capfd.readouterr()
        assert "The parameters of the variable y evaluate to:\n0: array(0., dtype=float32)\n1: array([1., 1., 1.])]"
        assert "Some of the values of variable y are associated with a non-finite logp" in out
        assert "value = -1.0 -> logp = -inf" in out

    def test_invalid_observed_value(self, capfd):
        with pm.Model() as m:
            theta = pm.Uniform("theta", lower=0, upper=1)
            y = pm.Uniform("y", lower=0, upper=theta, observed=[0.49, 0.27, 0.53, 0.19])
        m.debug()

        out, _ = capfd.readouterr()
        assert "The parameters of the variable y evaluate to:\n0: 0.0\n1: 0.5"
        assert (
            "Some of the observed values of variable y are associated with a non-finite logp" in out
        )
        assert "value = 0.53 -> logp = -inf" in out


def test_model_logp_fast_compile():
    # Issue #5618
    with pm.Model() as m:
        pm.Dirichlet("a", np.ones(3))

    with pytensor.config.change_flags(mode="FAST_COMPILE"):
        assert m.point_logps() == {"a": -1.5}


class TestModelGraphs:
    @staticmethod
    def school_model(J: int) -> pm.Model:
        y = np.array([28, 8, -3, 7, -1, 1, 18, 12])
        sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18])
        with pm.Model(coords={"school": np.arange(J)}) as schools:
            eta = pm.Normal("eta", 0, 1, dims="school")
            mu = pm.Normal("mu", 0, sigma=1e6)
            tau = pm.HalfCauchy("tau", 25)
            theta = mu + tau * eta
            pm.Normal("obs", theta, sigma=sigma, observed=y, dims="school")
        return schools

    @pytest.mark.parametrize(
        argnames="var_names", argvalues=[None, ["mu", "tau"]], ids=["all", "subset"]
    )
    def test_graphviz_call_function(self, var_names) -> None:
        model = self.school_model(J=8)
        with patch("pymc.model.core.model_to_graphviz") as mock_model_to_graphviz:
            model.to_graphviz(var_names=var_names)
            mock_model_to_graphviz.assert_called_once_with(
                model=model, var_names=var_names, formatting="plain"
            )
