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
import unittest

from functools import reduce

import aesara
import aesara.sparse as sparse
import aesara.tensor as at
import arviz as az
import cloudpickle
import numpy as np
import numpy.ma as ma
import numpy.testing as npt
import pytest
import scipy.sparse as sps
import scipy.stats as st

from aesara.tensor.random.op import RandomVariable
from aesara.tensor.var import TensorConstant

import pymc as pm

from pymc import Deterministic, Potential
from pymc.blocking import DictToArrayBijection, RaveledVars
from pymc.distributions import Normal, transforms
from pymc.exceptions import ShapeError
from pymc.model import Point, ValueGradFunction
from pymc.tests.helpers import SeededTest


class NewModel(pm.Model):
    def __init__(self, name="", model=None):
        super().__init__(name, model)
        assert pm.modelcontext(None) is self
        # 1) init variables with Var method
        self.register_rv(pm.Normal.dist(), "v1")
        self.v2 = pm.Normal("v2", mu=0, sigma=1)
        # 2) Potentials and Deterministic variables with method too
        # be sure that names will not overlap with other same models
        pm.Deterministic("d", at.constant(1))
        pm.Potential("p", at.constant(1))


class DocstringModel(pm.Model):
    def __init__(self, mean=0, sigma=1, name="", model=None):
        super().__init__(name, model)
        self.register_rv(Normal.dist(mu=mean, sigma=sigma), "v1")
        Normal("v2", mu=mean, sigma=sigma)
        Normal("v3", mu=mean, sigma=Normal("sigma", mu=10, sigma=1, initval=1.0))
        Deterministic("v3_sq", self.v3**2)
        Potential("p1", at.constant(1))


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
        X = pm.floatX(aesara.shared(X_))
        with pm.Model():
            x1 = pm.Normal("x1", observed=X_)
            x2 = pm.Normal("x2", observed=X)

        assert x1.type == X.type
        assert x2.type == X.type


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
        a = at.vector("a")
        a.tag.test_value = np.zeros(3, dtype=a.dtype)
        f_grad = ValueGradFunction([a.sum()], [a], {}, mode="FAST_COMPILE")
        assert f_grad._extra_vars == []

    def test_invalid_type(self):
        a = at.ivector("a")
        a.tag.test_value = np.zeros(3, dtype=a.dtype)
        a.dshape = (3,)
        a.dsize = 3
        with pytest.raises(TypeError) as err:
            ValueGradFunction([a.sum()], [a], {}, mode="FAST_COMPILE")
        err.match("Invalid dtype")

    def setUp(self):
        extra1 = at.iscalar("extra1")
        extra1_ = np.array(0, dtype=extra1.dtype)
        extra1.dshape = tuple()
        extra1.dsize = 1

        val1 = at.vector("val1")
        val1_ = np.zeros(3, dtype=val1.dtype)
        val1.dshape = (3,)
        val1.dsize = 3

        val2 = at.matrix("val2")
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
        initial_point = m.compute_initial_point()
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
            x2 = pm.Bernoulli("x2", x1, observed=X)

        gf = m.logp_dlogp_function()
        gf._extra_are_set = True

        assert m["x2_missing"].type == gf._extra_vars_shared["x2_missing"].type

        pnt = m.test_point.copy()
        del pnt["x2_missing"]

        res = [gf(DictToArrayBijection.map(Point(pnt, model=m))) for i in range(5)]

        assert reduce(lambda x, y: np.array_equal(x, y) and y, res) is not False

    def test_aesara_switch_broadcast_edge_cases_1(self):
        # Tests against two subtle issues related to a previous bug in Theano
        # where `tt.switch` would not always broadcast tensors with single
        # values https://github.com/pymc-devs/aesara/issues/270

        # Known issue 1: https://github.com/pymc-devs/pymc/issues/4389
        data = pm.floatX(np.zeros(10))
        with pm.Model() as m:
            p = pm.Beta("p", 1, 1)
            obs = pm.Bernoulli("obs", p=p, observed=data)

        npt.assert_allclose(
            m.compile_logp(obs)({"p_logodds__": pm.floatX(np.array(0.0))}),
            np.log(0.5) * 10,
        )

    def test_aesara_switch_broadcast_edge_cases_2(self):
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
        x = pm.DensityDist(  # pylint: disable=unused-variable
            "x", mu, logp=lambda value, mu: pm.Normal.logp(value, mu, 1.0), observed=0.1
        )
    assert not model["x"] == model["mu"]
    assert model["x"] == model["x"]
    assert model["x"] in model.observed_RVs
    assert not model["x"] in model.value_vars


def test_tempered_logp_dlogp():
    with pm.Model() as model:
        pm.Normal("x")
        pm.Normal("y", observed=1)
        pm.Potential("z", at.constant(-1.0, dtype=aesara.config.floatX))

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


def test_model_pickle(tmpdir):
    """Tests that PyMC models are pickleable"""
    with pm.Model() as model:
        x = pm.Normal("x")
        pm.Normal("y", observed=1)

    cloudpickle.loads(cloudpickle.dumps(model))


def test_model_pickle_deterministic(tmpdir):
    """Tests that PyMC models are pickleable"""
    with pm.Model() as model:
        x = pm.Normal("x")
        z = pm.Normal("z")
        pm.Deterministic("w", x / z)
        pm.Normal("y", observed=1)

    cloudpickle.loads(cloudpickle.dumps(model))


def test_model_vars():
    with pm.Model() as model:
        a = pm.Normal("a")
        pm.Normal("x", a)

    with pytest.warns(FutureWarning):
        old_vars = model.vars

    assert old_vars == model.value_vars


def test_model_var_maps():
    with pm.Model() as model:
        a = pm.Uniform("a")
        x = pm.Normal("x", a)

    assert model.rvs_to_values == {a: a.tag.value_var, x: x.tag.value_var}
    assert model.values_to_rvs == {a.tag.value_var: a, x.tag.value_var: x}


def test_make_obs_var():
    """
    Check returned values for `data` given known inputs to `as_tensor()`.

    Note that ndarrays should return a TensorConstant and sparse inputs
    should return a Sparse Aesara object.
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
        fake_model.make_obs_var(fake_distribution, np.ones((3, 3, 1)), None, None)

    # Check function behavior using the various inputs
    # dense, sparse: Ensure that the missing values are appropriately set to None
    # masked: a deterministic variable is returned

    dense_output = fake_model.make_obs_var(fake_distribution, dense_input, None, None)
    assert dense_output == fake_distribution
    assert isinstance(dense_output.tag.observations, TensorConstant)
    del fake_model.named_vars[fake_distribution.name]

    sparse_output = fake_model.make_obs_var(fake_distribution, sparse_input, None, None)
    assert sparse_output == fake_distribution
    assert sparse.basic._is_sparse_variable(sparse_output.tag.observations)
    del fake_model.named_vars[fake_distribution.name]

    # Here the RandomVariable is split into observed/imputed and a Deterministic is returned
    masked_output = fake_model.make_obs_var(fake_distribution, masked_array_input, None, None)
    assert masked_output != fake_distribution
    assert not isinstance(masked_output, RandomVariable)
    # Ensure it has missing values
    assert {"testing_inputs_missing"} == {v.name for v in fake_model.vars}
    assert {"testing_inputs", "testing_inputs_observed"} == {
        v.name for v in fake_model.observed_RVs
    }


def test_initial_point():

    with pm.Model() as model:
        a = pm.Uniform("a")
        x = pm.Normal("x", a)

    with pytest.warns(FutureWarning):
        initial_point = model.test_point

    assert all(var.name in initial_point for var in model.value_vars)

    b_initval = np.array(0.3, dtype=aesara.config.floatX)

    with pytest.warns(FutureWarning), model:
        b = pm.Uniform("b", testval=b_initval)

    b_value_var = model.rvs_to_values[b]
    b_initval_trans = b_value_var.tag.transform.forward(b_initval, *b.owner.inputs).eval()

    y_initval = np.array(-2.4, dtype=aesara.config.floatX)

    with model:
        y = pm.Normal("y", initval=y_initval)

    assert a in model.initial_values
    assert x in model.initial_values
    assert model.initial_values[b] == b_initval
    assert model.compute_initial_point(0)["b_interval__"] == b_initval_trans
    assert model.initial_values[y] == y_initval


def test_point_logps():

    with pm.Model() as model:
        a = pm.Uniform("a")
        pm.Normal("x", a)

    with pytest.warns(FutureWarning):
        logp_vals = model.check_test_point()

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
            pm.LogNormal("lognorm", mu=at.log(obs))
            pm.Normal("from_dims", dims=("city", "year"))
        shapes = pmodel.eval_rv_shapes()
        assert shapes["untransformed"] == (1, 2)
        assert shapes["transformed"] == (7,)
        assert shapes["transformed_interval__"] == (7,)
        assert shapes["lognorm"] == (3,)
        assert shapes["lognorm_log__"] == (3,)
        assert shapes["from_dims"] == (3, 4)


class TestCheckStartVals(SeededTest):
    def setup_method(self):
        super().setup_method()

    def test_valid_start_point(self):
        with pm.Model() as model:
            a = pm.Uniform("a", lower=0.0, upper=1.0)
            b = pm.Uniform("b", lower=2.0, upper=3.0)

        start = {
            "a_interval__": model.rvs_to_values[a]
            .tag.transform.forward(0.3, *a.owner.inputs)
            .eval(),
            "b_interval__": model.rvs_to_values[b]
            .tag.transform.forward(2.1, *b.owner.inputs)
            .eval(),
        }
        model.check_start_vals(start)

    def test_invalid_start_point(self):
        with pm.Model() as model:
            a = pm.Uniform("a", lower=0.0, upper=1.0)
            b = pm.Uniform("b", lower=2.0, upper=3.0)

        start = {
            "a_interval__": np.nan,
            "b_interval__": model.rvs_to_values[b]
            .tag.transform.forward(2.1, *b.owner.inputs)
            .eval(),
        }
        with pytest.raises(pm.exceptions.SamplingError):
            model.check_start_vals(start)

    def test_invalid_variable_name(self):
        with pm.Model() as model:
            a = pm.Uniform("a", lower=0.0, upper=1.0)
            b = pm.Uniform("b", lower=2.0, upper=3.0)

        start = {
            "a_interval__": model.rvs_to_values[a]
            .tag.transform.forward(0.3, *a.owner.inputs)
            .eval(),
            "b_interval__": model.rvs_to_values[b]
            .tag.transform.forward(2.1, *b.owner.inputs)
            .eval(),
            "c": 1.0,
        }
        with pytest.raises(KeyError):
            model.check_start_vals(start)


def test_set_initval():
    # Make sure the dependencies between variables are maintained when
    # generating initial values
    rng = np.random.RandomState(392)

    with pm.Model(rng_seeder=rng) as model:
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


def test_datalogpt_multiple_shapes():
    with pm.Model() as m:
        x = pm.Normal("x", 0, 1)
        z1 = pm.Potential("z1", x)
        z2 = pm.Potential("z2", at.full((1, 3), x))
        y1 = pm.Normal("y1", x, 1, observed=np.array([1]))
        y2 = pm.Normal("y2", x, 1, observed=np.array([1, 2]))
        y3 = pm.Normal("y3", x, 1, observed=np.array([1, 2, 3]))

    # This would raise a TypeError, see #4803 and #4804
    x_val = m.rvs_to_values[x]
    m.datalogpt.eval({x_val: 0})


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
    assert set(m2.RV_dims) < set(m1.RV_dims)


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


def test_empty_model_representation():
    assert pm.Model().str_repr() == ""
