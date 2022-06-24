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
from unittest import mock

import aesara
import aesara.tensor as at
import numpy as np
import numpy.ma as ma
import numpy.testing as npt
import pytest
import scipy.sparse as sps

from aeppl.abstract import MeasurableVariable
from aeppl.logprob import ParameterValueError
from aesara.compile.builders import OpFromGraph
from aesara.graph.basic import Constant, Variable, ancestors, equal_computations
from aesara.tensor.random.basic import normal, uniform
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.random.var import RandomStateSharedVariable
from aesara.tensor.subtensor import AdvancedIncSubtensor, AdvancedIncSubtensor1
from aesara.tensor.var import TensorVariable

import pymc as pm

from pymc.aesaraf import (
    change_rv_size,
    compile_pymc,
    convert_observed_data,
    extract_obs_data,
    reseed_rngs,
    rvs_to_value_vars,
    walk_model,
)
from pymc.distributions.dist_math import check_parameters
from pymc.exceptions import ShapeError
from pymc.vartypes import int_types


@pytest.mark.parametrize(
    argnames="np_array",
    argvalues=[np.array([[1.0], [2.0], [-1.0]]), np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])],
)
def test_pd_as_tensor_variable(np_array: np.ndarray) -> None:
    pd = pytest.importorskip("pandas")
    df: pd.DataFrame = pd.DataFrame(np_array)
    np.testing.assert_array_equal(x=at.as_tensor_variable(x=df).eval(), y=np_array)


def test_pd_as_tensor_variable_multiindex() -> None:
    pd = pytest.importorskip("pandas")

    tuples = [("L", "Q"), ("L", "I"), ("O", "L"), ("O", "I")]

    index = pd.MultiIndex.from_tuples(tuples, names=["Id1", "Id2"])

    df = pd.DataFrame({"A": [12.0, 80.0, 30.0, 20.0], "B": [120.0, 700.0, 30.0, 20.0]}, index=index)
    np_array = np.array([[12.0, 80.0, 30.0, 20.0], [120.0, 700.0, 30.0, 20.0]]).T
    assert isinstance(df.index, pd.MultiIndex)
    np.testing.assert_array_equal(x=at.as_tensor_variable(x=df).eval(), y=np_array)


def test_change_rv_size():
    loc = at.as_tensor_variable([1, 2])
    rv = normal(loc=loc)
    assert rv.ndim == 1
    assert tuple(rv.shape.eval()) == (2,)

    with pytest.raises(ShapeError, match="must be ≤1-dimensional"):
        change_rv_size(rv, new_size=[[2, 3]])
    with pytest.raises(ShapeError, match="must be ≤1-dimensional"):
        change_rv_size(rv, new_size=at.as_tensor_variable([[2, 3], [4, 5]]))

    rv_new = change_rv_size(rv, new_size=(3,), expand=True)
    assert rv_new.ndim == 2
    assert tuple(rv_new.shape.eval()) == (3, 2)

    # Make sure that the shape used to determine the expanded size doesn't
    # depend on the old `RandomVariable`.
    rv_new_ancestors = set(ancestors((rv_new,)))
    assert loc in rv_new_ancestors
    assert rv not in rv_new_ancestors

    rv_newer = change_rv_size(rv_new, new_size=(4,), expand=True)
    assert rv_newer.ndim == 3
    assert tuple(rv_newer.shape.eval()) == (4, 3, 2)

    # Make sure we avoid introducing a `Cast` by converting the new size before
    # constructing the new `RandomVariable`
    rv = normal(0, 1)
    new_size = np.array([4, 3], dtype="int32")
    rv_newer = change_rv_size(rv, new_size=new_size, expand=False)
    assert rv_newer.ndim == 2
    assert isinstance(rv_newer.owner.inputs[1], Constant)
    assert tuple(rv_newer.shape.eval()) == (4, 3)

    rv = normal(0, 1)
    new_size = at.as_tensor(np.array([4, 3], dtype="int32"))
    rv_newer = change_rv_size(rv, new_size=new_size, expand=True)
    assert rv_newer.ndim == 2
    assert tuple(rv_newer.shape.eval()) == (4, 3)

    rv = normal(0, 1)
    new_size = at.as_tensor(2, dtype="int32")
    rv_newer = change_rv_size(rv, new_size=new_size, expand=True)
    assert rv_newer.ndim == 1
    assert tuple(rv_newer.shape.eval()) == (2,)


def test_change_rv_size_default_update():
    rng = aesara.shared(np.random.default_rng(0))
    x = normal(rng=rng)

    # Test that "traditional" default_update is updated
    rng.default_update = x.owner.outputs[0]
    new_x = change_rv_size(x, new_size=(2,))
    assert rng.default_update is not x.owner.outputs[0]
    assert rng.default_update is new_x.owner.outputs[0]

    # Test that "non-traditional" default_update is left unchanged
    next_rng = aesara.shared(np.random.default_rng(1))
    rng.default_update = next_rng
    new_x = change_rv_size(x, new_size=(2,))
    assert rng.default_update is next_rng

    # Test that default_update is not set if there was none before
    del rng.default_update
    new_x = change_rv_size(x, new_size=(2,))
    assert not hasattr(rng, "default_update")


class TestBroadcasting:
    def test_make_shared_replacements(self):
        """Check if pm.make_shared_replacements preserves broadcasting."""

        with pm.Model() as test_model:
            test1 = pm.Normal("test1", mu=0.0, sigma=1.0, size=(1, 10))
            test2 = pm.Normal("test2", mu=0.0, sigma=1.0, size=(10, 1))

        # Replace test1 with a shared variable, keep test 2 the same
        replacement = pm.make_shared_replacements(
            test_model.initial_point(), [test_model.test2], test_model
        )
        assert (
            test_model.test1.broadcastable
            == replacement[test_model.test1.tag.value_var].broadcastable
        )

    def test_metropolis_sampling(self):
        """Check if the Metropolis sampler can handle broadcasting."""
        with pm.Model() as test_model:
            test1 = pm.Normal("test1", mu=0.0, sigma=1.0, size=(1, 10))
            test2 = pm.Normal("test2", mu=test1, sigma=1.0, size=(10, 10))

            step = pm.Metropolis()
            # TODO FIXME: Assert whatever it is we're testing
            pm.sample(tune=5, draws=7, cores=1, step=step, compute_convergence_checks=False)


def _make_along_axis_idx(arr_shape, indices, axis):
    # compute dimensions to iterate over
    if str(indices.dtype) not in int_types:
        raise IndexError("`indices` must be an integer array")
    shape_ones = (1,) * indices.ndim
    dest_dims = list(range(axis)) + [None] + list(range(axis + 1, indices.ndim))

    # build a fancy index, consisting of orthogonal aranges, with the
    # requested index inserted at the right location
    fancy_index = []
    for dim, n in zip(dest_dims, arr_shape):
        if dim is None:
            fancy_index.append(indices)
        else:
            ind_shape = shape_ones[:dim] + (-1,) + shape_ones[dim + 1 :]
            fancy_index.append(np.arange(n).reshape(ind_shape))

    return tuple(fancy_index)


def test_extract_obs_data():

    with pytest.raises(TypeError):
        extract_obs_data(at.matrix())

    data = np.random.normal(size=(2, 3))
    data_at = at.as_tensor(data)
    mask = np.random.binomial(1, 0.5, size=(2, 3)).astype(bool)

    for val_at in (data_at, aesara.shared(data)):
        res = extract_obs_data(val_at)

        assert isinstance(res, np.ndarray)
        assert np.array_equal(res, data)

    # AdvancedIncSubtensor check
    data_m = np.ma.MaskedArray(data, mask)
    missing_values = data_at.type()[mask]
    constant = at.as_tensor(data_m.filled())
    z_at = at.set_subtensor(constant[mask.nonzero()], missing_values)

    assert isinstance(z_at.owner.op, (AdvancedIncSubtensor, AdvancedIncSubtensor1))

    res = extract_obs_data(z_at)

    assert isinstance(res, np.ndarray)
    assert np.ma.allequal(res, data_m)

    # AdvancedIncSubtensor1 check
    data = np.random.normal(size=(3,))
    data_at = at.as_tensor(data)
    mask = np.random.binomial(1, 0.5, size=(3,)).astype(bool)

    data_m = np.ma.MaskedArray(data, mask)
    missing_values = data_at.type()[mask]
    constant = at.as_tensor(data_m.filled())
    z_at = at.set_subtensor(constant[mask.nonzero()], missing_values)

    assert isinstance(z_at.owner.op, (AdvancedIncSubtensor, AdvancedIncSubtensor1))

    res = extract_obs_data(z_at)

    assert isinstance(res, np.ndarray)
    assert np.ma.allequal(res, data_m)

    # Cast check
    data = np.array(5)
    t = at.cast(at.as_tensor(5.0), np.int64)
    res = extract_obs_data(t)

    assert isinstance(res, np.ndarray)
    assert np.array_equal(res, data)


@pytest.mark.parametrize("input_dtype", ["int32", "int64", "float32", "float64"])
def test_convert_observed_data(input_dtype):
    """
    Ensure that convert_observed_data returns the dense array, masked array,
    graph variable, TensorVariable, or sparse matrix as appropriate.
    """
    pd = pytest.importorskip("pandas")
    # Create the various inputs to the function
    sparse_input = sps.csr_matrix(np.eye(3)).astype(input_dtype)
    dense_input = np.arange(9).reshape((3, 3)).astype(input_dtype)

    input_name = "input_variable"
    aesara_graph_input = at.as_tensor(dense_input, name=input_name)
    pandas_input = pd.DataFrame(dense_input)

    # All the even numbers are replaced with NaN
    missing_numpy_input = np.array([[np.nan, 1, np.nan], [3, np.nan, 5], [np.nan, 7, np.nan]])
    missing_pandas_input = pd.DataFrame(missing_numpy_input)
    masked_array_input = ma.array(dense_input, mask=(np.mod(dense_input, 2) == 0))

    # Create a generator object. Apparently the generator object needs to
    # yield numpy arrays.
    square_generator = (np.array([i**2], dtype=int) for i in range(100))

    # Alias the function to be tested
    func = convert_observed_data

    #####
    # Perform the various tests
    #####
    # Check function behavior with dense arrays and pandas dataframes
    # without missing values
    for input_value in [dense_input, pandas_input]:
        func_output = func(input_value)
        assert isinstance(func_output, np.ndarray)
        assert func_output.shape == input_value.shape
        npt.assert_allclose(func_output, dense_input)

    # Check function behavior with sparse matrix inputs
    sparse_output = func(sparse_input)
    assert sps.issparse(sparse_output)
    assert sparse_output.shape == sparse_input.shape
    npt.assert_allclose(sparse_output.toarray(), sparse_input.toarray())

    # Check function behavior when using masked array inputs and pandas
    # objects with missing data
    for input_value in [missing_numpy_input, masked_array_input, missing_pandas_input]:
        func_output = func(input_value)
        assert isinstance(func_output, ma.core.MaskedArray)
        assert func_output.shape == input_value.shape
        npt.assert_allclose(func_output, masked_array_input)

    # Check function behavior with Aesara graph variable
    aesara_output = func(aesara_graph_input)
    assert isinstance(aesara_output, Variable)
    npt.assert_allclose(aesara_output.eval(), aesara_graph_input.eval())
    intX = pm.aesaraf._conversion_map[aesara.config.floatX]
    if dense_input.dtype == intX or dense_input.dtype == aesara.config.floatX:
        assert aesara_output.owner is None  # func should not have added new nodes
        assert aesara_output.name == input_name
    else:
        assert aesara_output.owner is not None  # func should have casted
        assert aesara_output.owner.inputs[0].name == input_name

    if "float" in input_dtype:
        assert aesara_output.dtype == aesara.config.floatX
    else:
        assert aesara_output.dtype == intX

    # Check function behavior with generator data
    generator_output = func(square_generator)

    # Output is wrapped with `pm.floatX`, and this unwraps
    wrapped = generator_output.owner.inputs[0]
    # Make sure the returned object has .set_gen and .set_default methods
    assert hasattr(wrapped, "set_gen")
    assert hasattr(wrapped, "set_default")
    # Make sure the returned object is an Aesara TensorVariable
    assert isinstance(wrapped, TensorVariable)


def test_pandas_to_array_pandas_index():
    pd = pytest.importorskip("pandas")
    data = pd.Index([1, 2, 3])
    result = convert_observed_data(data)
    expected = np.array([1, 2, 3])
    np.testing.assert_array_equal(result, expected)


def test_walk_model():
    d = at.vector("d")
    b = at.vector("b")
    c = uniform(0.0, d)
    c.name = "c"
    e = at.log(c)
    a = normal(e, b)
    a.name = "a"

    test_graph = at.exp(a + 1)
    res = list(walk_model((test_graph,)))
    assert a in res
    assert c not in res

    res = list(walk_model((test_graph,), walk_past_rvs=True))
    assert a in res
    assert c in res

    res = list(walk_model((test_graph,), walk_past_rvs=True, stop_at_vars={e}))
    assert a in res
    assert c not in res


def test_rvs_to_value_vars():

    with pm.Model() as m:
        a = pm.Uniform("a", 0.0, 1.0)
        b = pm.Uniform("b", 0, a + 1.0)
        c = pm.Normal("c")
        d = at.log(c + b) + 2.0

    a_value_var = m.rvs_to_values[a]
    assert a_value_var.tag.transform

    b_value_var = m.rvs_to_values[b]
    c_value_var = m.rvs_to_values[c]

    (res,), replaced = rvs_to_value_vars((d,))

    assert res.owner.op == at.add
    log_output = res.owner.inputs[0]
    assert log_output.owner.op == at.log
    log_add_output = res.owner.inputs[0].owner.inputs[0]
    assert log_add_output.owner.op == at.add
    c_output = log_add_output.owner.inputs[0]

    # We make sure that the random variables were replaced
    # with their value variables
    assert c_output == c_value_var
    b_output = log_add_output.owner.inputs[1]
    assert b_output == b_value_var

    res_ancestors = list(walk_model((res,), walk_past_rvs=True))
    res_rv_ancestors = [
        v for v in res_ancestors if v.owner and isinstance(v.owner.op, RandomVariable)
    ]

    # There shouldn't be any `RandomVariable`s in the resulting graph
    assert len(res_rv_ancestors) == 0
    assert b_value_var in res_ancestors
    assert c_value_var in res_ancestors
    assert a_value_var not in res_ancestors

    (res,), replaced = rvs_to_value_vars((d,), apply_transforms=True)

    res_ancestors = list(walk_model((res,), walk_past_rvs=True))
    res_rv_ancestors = [
        v for v in res_ancestors if v.owner and isinstance(v.owner.op, RandomVariable)
    ]

    assert len(res_rv_ancestors) == 0
    assert a_value_var in res_ancestors
    assert b_value_var in res_ancestors
    assert c_value_var in res_ancestors


def test_rvs_to_value_vars_nested():
    # Test that calling rvs_to_value_vars in models with nested transformations
    # does not change the original rvs in place. See issue #5172
    with pm.Model() as m:
        one = pm.LogNormal("one", mu=0)
        two = pm.LogNormal("two", mu=at.log(one))

        # We add potentials or deterministics that are not in topological order
        pm.Potential("two_pot", two)
        pm.Potential("one_pot", one)

        before = aesara.clone_replace(m.free_RVs)

        # This call would change the model free_RVs in place in #5172
        res, _ = rvs_to_value_vars(m.potentials, apply_transforms=True)

        after = aesara.clone_replace(m.free_RVs)

        assert equal_computations(before, after)


class TestCompilePyMC:
    def test_check_bounds_flag(self):
        """Test that CheckParameterValue Ops are replaced or removed when using compile_pymc"""
        logp = at.ones(3)
        cond = np.array([1, 0, 1])
        bound = check_parameters(logp, cond)

        with pm.Model() as m:
            pass

        with pytest.raises(ParameterValueError):
            aesara.function([], bound)()

        m.check_bounds = False
        with m:
            assert np.all(compile_pymc([], bound)() == 1)

        m.check_bounds = True
        with m:
            assert np.all(compile_pymc([], bound)() == -np.inf)

    def test_compile_pymc_sets_rng_updates(self):
        rng = aesara.shared(np.random.default_rng(0))
        x = pm.Normal.dist(rng=rng)
        assert x.owner.inputs[0] is rng
        f = compile_pymc([], x)
        assert not np.isclose(f(), f())

        # Check that update was not done inplace
        assert not hasattr(rng, "default_update")
        f = aesara.function([], x)
        assert f() == f()

    def test_compile_pymc_with_updates(self):
        x = aesara.shared(0)
        f = compile_pymc([], x, updates={x: x + 1})
        assert f() == 0
        assert f() == 1

    def test_compile_pymc_missing_default_explicit_updates(self):
        rng = aesara.shared(np.random.default_rng(0))
        x = pm.Normal.dist(rng=rng)

        # By default, compile_pymc should update the rng of x
        f = compile_pymc([], x)
        assert f() != f()

        # An explicit update should override the default_update, like aesara.function does
        # For testing purposes, we use an update that leaves the rng unchanged
        f = compile_pymc([], x, updates={rng: rng})
        assert f() == f()

        # If we specify a custom default_update directly it should use that instead.
        rng.default_update = rng
        f = compile_pymc([], x)
        assert f() == f()

        # And again, it should be overridden by an explicit update
        f = compile_pymc([], x, updates={rng: x.owner.outputs[0]})
        assert f() != f()

    def test_compile_pymc_updates_inputs(self):
        """Test that compile_pymc does not include rngs updates of variables that are inputs
        or ancestors to inputs
        """
        x = at.random.normal()
        y = at.random.normal(x)
        z = at.random.normal(y)

        for inputs, rvs_in_graph in (
            ([], 3),
            ([x], 2),
            ([y], 1),
            ([z], 0),
            ([x, y], 1),
            ([x, y, z], 0),
        ):
            fn = compile_pymc(inputs, z, on_unused_input="ignore")
            fn_fgraph = fn.maker.fgraph
            # Each RV adds a shared input for its rng
            assert len(fn_fgraph.inputs) == len(inputs) + rvs_in_graph
            # If the output is an input, the graph has a DeepCopyOp
            assert len(fn_fgraph.apply_nodes) == max(rvs_in_graph, 1)
            # Each RV adds a shared output for its rng
            assert len(fn_fgraph.outputs) == 1 + rvs_in_graph

    # Disable `reseed_rngs` so that we can test with simpler update rule
    @mock.patch("pymc.aesaraf.reseed_rngs")
    def test_compile_pymc_custom_update_op(self, _):
        """Test that custom MeasurableVariable Op updates are used by compile_pymc"""

        class UnmeasurableOp(OpFromGraph):
            def update(self, node):
                return {node.inputs[0]: node.inputs[0] + 1}

        dummy_inputs = [at.scalar(), at.scalar()]
        dummy_outputs = [at.add(*dummy_inputs)]
        dummy_x = UnmeasurableOp(dummy_inputs, dummy_outputs)(aesara.shared(1.0), 1.0)

        # Check that there are no updates at first
        fn = compile_pymc(inputs=[], outputs=dummy_x)
        assert fn() == fn() == 2.0

        # And they are enabled once the Op is registered as Measurable
        MeasurableVariable.register(UnmeasurableOp)
        fn = compile_pymc(inputs=[], outputs=dummy_x)
        assert fn() == 2.0
        assert fn() == 3.0

    def test_random_seed(self):
        seedx = aesara.shared(np.random.default_rng(1))
        seedy = aesara.shared(np.random.default_rng(1))
        x = at.random.normal(rng=seedx)
        y = at.random.normal(rng=seedy)

        # Shared variables are the same, so outputs will be identical
        f0 = aesara.function([], [x, y])
        x0_eval, y0_eval = f0()
        assert x0_eval == y0_eval

        # The variables will be reseeded with new seeds by default
        f1 = compile_pymc([], [x, y])
        x1_eval, y1_eval = f1()
        assert x1_eval != y1_eval

        # Check that seeding works
        f2 = compile_pymc([], [x, y], random_seed=1)
        x2_eval, y2_eval = f2()
        assert x2_eval != x1_eval
        assert y2_eval != y1_eval

        f3 = compile_pymc([], [x, y], random_seed=1)
        x3_eval, y3_eval = f3()
        assert x3_eval == x2_eval
        assert y3_eval == y2_eval


def test_reseed_rngs():
    # Reseed_rngs uses the `PCG64` bit_generator, which is currently the default
    # bit_generator used by NumPy. If this default changes in the future, this test will
    # catch that. We will then have to decide whether to switch to the new default in
    # PyMC or whether to stick with the older one (PCG64). This will pose a trade-off
    # between backwards reproducibility and better/faster seeding. If we decide to change,
    # the next line should be updated:
    default_rng = np.random.PCG64
    assert isinstance(np.random.default_rng().bit_generator, default_rng)

    seed = 543

    bit_generators = [default_rng(sub_seed) for sub_seed in np.random.SeedSequence(seed).spawn(2)]

    rngs = [
        aesara.shared(rng_type(default_rng()))
        for rng_type in (np.random.Generator, np.random.RandomState)
    ]
    for rng, bit_generator in zip(rngs, bit_generators):
        if isinstance(rng, RandomStateSharedVariable):
            assert rng.get_value()._bit_generator.state != bit_generator.state
        else:
            assert rng.get_value().bit_generator.state != bit_generator.state

    reseed_rngs(rngs, seed)
    for rng, bit_generator in zip(rngs, bit_generators):
        if isinstance(rng, RandomStateSharedVariable):
            assert rng.get_value()._bit_generator.state == bit_generator.state
        else:
            assert rng.get_value().bit_generator.state == bit_generator.state
