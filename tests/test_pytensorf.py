#   Copyright 2024 The PyMC Developers
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
import numpy.ma as ma
import numpy.testing as npt
import pandas as pd
import pytensor
import pytensor.tensor as pt
import pytest
import scipy.sparse as sps

from pytensor import scan, shared
from pytensor.compile import UnusedInputError
from pytensor.compile.builders import OpFromGraph
from pytensor.graph.basic import Variable, equal_computations
from pytensor.tensor.random.basic import normal, uniform
from pytensor.tensor.subtensor import AdvancedIncSubtensor
from pytensor.tensor.variable import TensorVariable

import pymc as pm

from pymc.data import Minibatch, MinibatchOp
from pymc.distributions.dist_math import check_parameters
from pymc.distributions.distribution import SymbolicRandomVariable
from pymc.exceptions import NotConstantValueError
from pymc.logprob.utils import ParameterValueError
from pymc.pytensorf import (
    GeneratorOp,
    collect_default_updates,
    compile_pymc,
    constant_fold,
    convert_data,
    convert_generator_data,
    extract_obs_data,
    hessian,
    hessian_diag,
    replace_rng_nodes,
    replace_vars_in_graphs,
    reseed_rngs,
    smarttypeX,
    walk_model,
)
from pymc.vartypes import int_types


@pytest.mark.parametrize(
    argnames="np_array",
    argvalues=[
        np.array([[1.0], [2.0], [-1.0]]),
        np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]),
        np.ones(shape=(10, 1)),
    ],
)
def test_pd_dataframe_as_tensor_variable(np_array: np.ndarray) -> None:
    df = pd.DataFrame(np_array)
    np.testing.assert_array_equal(x=pt.as_tensor_variable(x=df).eval(), y=np_array)


@pytest.mark.parametrize(
    argnames="np_array",
    argvalues=[np.array([1.0, 2.0, -1.0]), np.ones(shape=4), np.zeros(shape=10), [1, 2, 3, 4]],
)
def test_pd_series_as_tensor_variable(np_array: np.ndarray) -> None:
    df = pd.Series(np_array)
    np.testing.assert_array_equal(x=pt.as_tensor_variable(x=df).eval(), y=np_array)


def test_pd_as_tensor_variable_multiindex() -> None:
    tuples = [("L", "Q"), ("L", "I"), ("O", "L"), ("O", "I")]

    index = pd.MultiIndex.from_tuples(tuples, names=["Id1", "Id2"])

    df = pd.DataFrame({"A": [12.0, 80.0, 30.0, 20.0], "B": [120.0, 700.0, 30.0, 20.0]}, index=index)
    np_array = np.array([[12.0, 80.0, 30.0, 20.0], [120.0, 700.0, 30.0, 20.0]]).T
    assert isinstance(df.index, pd.MultiIndex)
    np.testing.assert_array_equal(x=pt.as_tensor_variable(x=df).eval(), y=np_array)


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
            == replacement[test_model.rvs_to_values[test_model.test1]].broadcastable
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
    dest_dims = [*list(range(axis)), None, *list(range(axis + 1, indices.ndim))]

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


class TestExtractObsData:
    def test_root_variable(self):
        with pytest.raises(TypeError):
            extract_obs_data(pt.matrix())

    def test_constant_variable(self):
        data = np.random.normal(size=(2, 3))
        data_pt = pt.as_tensor(data)
        res = extract_obs_data(data_pt)

        assert isinstance(res, np.ndarray)
        np.testing.assert_array_equal(res, data)

    def test_shared_variable(self):
        data = np.random.normal(size=(2, 3))
        data_pt = shared(data)

        res = extract_obs_data(data_pt)
        assert isinstance(res, np.ndarray)
        np.testing.assert_array_equal(res, data)

    def test_masked_variable(self):
        # Extract data from auto-imputation graph
        data = np.random.normal(size=(2, 3))
        data_pt = pt.as_tensor(data)
        mask = np.random.binomial(1, 0.5, size=(2, 3)).astype(bool)

        # AdvancedIncSubtensor check
        data_m = np.ma.MaskedArray(data, mask)
        missing_values = data_pt.type()[mask]
        constant = pt.as_tensor(data_m.filled())
        z_at = pt.set_subtensor(constant[mask.nonzero()], missing_values)
        assert isinstance(z_at.owner.op, AdvancedIncSubtensor)

        res = extract_obs_data(z_at)
        assert isinstance(res, np.ndarray)
        assert np.ma.allequal(res, data_m)

    def test_cast_variable(self):
        # Cast check
        data = np.array(5)
        data_pt = pt.cast(pt.as_tensor(5.0), np.int64)

        res = extract_obs_data(data_pt)
        assert isinstance(res, np.ndarray)
        np.testing.assert_array_equal(res, data)

    def test_minibatch_variable(self):
        x = np.arange(5)
        y = x * 2

        x_mb, y_mb = Minibatch(x, y, batch_size=2)
        assert isinstance(x_mb.owner.op, MinibatchOp)
        assert isinstance(y_mb.owner.op, MinibatchOp)

        res = extract_obs_data(x_mb)
        assert isinstance(res, np.ndarray)
        np.testing.assert_array_equal(res, x)

        res = extract_obs_data(y_mb)
        assert isinstance(res, np.ndarray)
        np.testing.assert_array_equal(res, y)


@pytest.mark.parametrize("input_dtype", ["int32", "int64", "float32", "float64"])
def test_convert_data(input_dtype):
    """
    Ensure that convert_data returns the dense array, masked array,
    graph variable, TensorVariable, or sparse matrix as appropriate.
    """
    # Create the various inputs to the function
    sparse_input = sps.csr_matrix(np.eye(3)).astype(input_dtype)
    dense_input = np.arange(9).reshape((3, 3)).astype(input_dtype)

    input_name = "input_variable"
    pytensor_graph_input = pt.as_tensor(dense_input, name=input_name)
    pandas_input = pd.DataFrame(dense_input)

    # All the even numbers are replaced with NaN
    missing_numpy_input = np.array([[np.nan, 1, np.nan], [3, np.nan, 5], [np.nan, 7, np.nan]])
    missing_pandas_input = pd.DataFrame(missing_numpy_input)
    masked_array_input = ma.array(dense_input, mask=(np.mod(dense_input, 2) == 0))

    # Alias the function to be tested
    func = convert_data

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

    # Check function behavior with PyTensor graph variable
    pytensor_output = func(pytensor_graph_input)
    assert isinstance(pytensor_output, Variable)
    npt.assert_allclose(pytensor_output.eval(), pytensor_graph_input.eval())
    intX = pm.pytensorf._conversion_map[pytensor.config.floatX]
    if dense_input.dtype == intX or dense_input.dtype == pytensor.config.floatX:
        assert pytensor_output.owner is None  # func should not have added new nodes
        assert pytensor_output.name == input_name
    else:
        assert pytensor_output.owner is not None  # func should have casted
        assert pytensor_output.owner.inputs[0].name == input_name

    if "float" in input_dtype:
        assert pytensor_output.dtype == pytensor.config.floatX
    else:
        assert pytensor_output.dtype == intX


@pytest.mark.parametrize("input_dtype", ["int32", "int64", "float32", "float64"])
def test_convert_generator_data(input_dtype):
    # Create a generator object producing NumPy arrays with the intended dtype.
    # This is required to infer the correct dtype.
    square_generator = (np.array([i**2], dtype=input_dtype) for i in range(100))

    # Output is NOT wrapped with `pm.floatX`/`intX`,
    # but produced from calling a special Op.
    with pytest.warns(DeprecationWarning, match="get in touch"):
        result = convert_generator_data(square_generator)
    apply = result.owner
    op = apply.op
    # Make sure the returned object is a PyTensor TensorVariable
    assert isinstance(result, TensorVariable)
    assert isinstance(op, GeneratorOp), f"It's a {type(apply)}"
    # There are no inputs - because it generates...
    assert apply.inputs == []

    # Evaluation results should have the correct* dtype!
    # (*intX/floatX will be enforced!)
    evaled = result.eval()
    expected_dtype = smarttypeX(np.array(1, dtype=input_dtype)).dtype
    assert result.type.dtype == expected_dtype
    assert evaled.dtype == np.dtype(expected_dtype)


def test_pandas_to_array_pandas_index():
    data = pd.Index([1, 2, 3])
    result = convert_data(data)
    expected = np.array([1, 2, 3])
    np.testing.assert_array_equal(result, expected)


def test_walk_model():
    a = pt.vector("a")
    b = uniform(0.0, a, name="b")
    c = pt.log(b)
    c.name = "c"
    d = pt.vector("d")
    e = normal(c, d, name="e")

    test_graph = pt.exp(e + 1)

    with pytest.warns(FutureWarning):
        res = list(walk_model((test_graph,)))
    assert a in res
    assert b in res
    assert c in res
    assert d in res
    assert e in res

    with pytest.warns(FutureWarning):
        res = list(walk_model((test_graph,), stop_at_vars={c}))
    assert a not in res
    assert b not in res
    assert c in res
    assert d in res
    assert e in res

    with pytest.warns(FutureWarning):
        res = list(walk_model((test_graph,), stop_at_vars={b}))
    assert a not in res
    assert b in res
    assert c in res
    assert d in res
    assert e in res


class TestCompilePyMC:
    def test_check_bounds_flag(self):
        """Test that CheckParameterValue Ops are replaced or removed when using compile_pymc"""
        logp = pt.ones(3)
        cond = np.array([1, 0, 1])
        bound = check_parameters(logp, cond)

        with pm.Model() as m:
            pass

        with pytest.raises(ParameterValueError):
            pytensor.function([], bound)()

        m.check_bounds = False
        with m:
            assert np.all(compile_pymc([], bound)() == 1)

        m.check_bounds = True
        with m:
            assert np.all(compile_pymc([], bound)() == -np.inf)

    def test_check_parameters_can_be_replaced_by_ninf(self):
        expr = pt.vector("expr", shape=(3,))
        cond = pt.ge(expr, 0)

        final_expr = check_parameters(expr, cond, can_be_replaced_by_ninf=True)
        fn = compile_pymc([expr], final_expr)
        np.testing.assert_array_equal(fn(expr=[1, 2, 3]), [1, 2, 3])
        np.testing.assert_array_equal(fn(expr=[-1, 2, 3]), [-np.inf, -np.inf, -np.inf])

        final_expr = check_parameters(expr, cond, msg="test", can_be_replaced_by_ninf=False)
        fn = compile_pymc([expr], final_expr)
        np.testing.assert_array_equal(fn(expr=[1, 2, 3]), [1, 2, 3])
        with pytest.raises(ParameterValueError, match="test"):
            fn([-1, 2, 3])

    def test_compile_pymc_sets_rng_updates(self):
        rng = pytensor.shared(np.random.default_rng(0))
        x = pm.Normal.dist(rng=rng)
        assert x.owner.inputs[0] is rng
        f = compile_pymc([], x)
        assert not np.isclose(f(), f())

        # Check that update was not done inplace
        assert rng.default_update is None
        f = pytensor.function([], x)
        assert f() == f()

    def test_compile_pymc_with_updates(self):
        x = pytensor.shared(0)
        f = compile_pymc([], x, updates={x: x + 1})
        assert f() == 0
        assert f() == 1

    def test_compile_pymc_missing_default_explicit_updates(self):
        rng = pytensor.shared(np.random.default_rng(0))
        x = pm.Normal.dist(rng=rng)

        # By default, compile_pymc should update the rng of x
        f = compile_pymc([], x)
        assert f() != f()

        # An explicit update should override the default_update, like pytensor.function does
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
        x = pt.random.normal()
        y = pt.random.normal(x)
        z = pt.random.normal(y)

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

    def test_compile_pymc_symbolic_rv_missing_update(self):
        """Test that error is raised if SymbolicRandomVariable Op does not
        provide rule for updating RNG"""

        class SymbolicRV(OpFromGraph):
            def update(self, node):
                # Update is provided for rng1 but not rng2
                return {node.inputs[0]: node.outputs[0]}

        SymbolicRandomVariable.register(SymbolicRV)

        # No problems at first, as the one RNG is given the update rule
        rng1 = pytensor.shared(np.random.default_rng())
        dummy_rng1 = rng1.type()
        dummy_next_rng1, dummy_x1 = SymbolicRV(
            [dummy_rng1],
            pt.random.normal(rng=dummy_rng1).owner.outputs,
        )(rng1)
        fn = compile_pymc(inputs=[], outputs=dummy_x1, random_seed=433)
        assert fn() != fn()

        # Now there's a problem as there is no update rule for rng2
        rng2 = pytensor.shared(np.random.default_rng())
        dummy_rng2 = rng2.type()
        dummy_next_rng1, dummy_x1, dummy_next_rng2, dummy_x2 = SymbolicRV(
            [dummy_rng1, dummy_rng2],
            [
                *pt.random.normal(rng=dummy_rng1).owner.outputs,
                *pt.random.normal(rng=dummy_rng2).owner.outputs,
            ],
        )(rng1, rng2)
        with pytest.raises(
            ValueError, match="No update found for at least one RNG used in SymbolicRandomVariable"
        ):
            compile_pymc(inputs=[], outputs=[dummy_x1, dummy_x2])

    def test_random_seed(self):
        seedx = pytensor.shared(np.random.default_rng(1))
        seedy = pytensor.shared(np.random.default_rng(1))
        x = pt.random.normal(rng=seedx)
        y = pt.random.normal(rng=seedy)

        # Shared variables are the same, so outputs will be identical
        f0 = pytensor.function([], [x, y])
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

    @pytest.mark.filterwarnings("error")  # This is part of the test
    def test_multiple_updates_same_variable(self):
        rng = pytensor.shared(np.random.default_rng(), name="rng")
        x = pt.random.normal(0, rng=rng)
        y = pt.random.normal(1, rng=rng)

        # No warnings if only one variable is used
        assert compile_pymc([], [x])
        assert compile_pymc([], [y])

        user_warn_msg = "RNG Variable rng has multiple distinct clients"
        with pytest.warns(UserWarning, match=user_warn_msg):
            f = compile_pymc([], [x, y], random_seed=456)
        assert f() == f()

        # The user can provide an explicit update, but we will still issue a warning
        with pytest.warns(UserWarning, match=user_warn_msg):
            f = compile_pymc([], [x, y], updates={rng: y.owner.outputs[0]}, random_seed=456)
        assert f() != f()

        # Same with default update
        rng.default_update = x.owner.outputs[0]
        with pytest.warns(UserWarning, match=user_warn_msg):
            f = compile_pymc([], [x, y], updates={rng: y.owner.outputs[0]}, random_seed=456)
        assert f() != f()

    @pytest.mark.filterwarnings("error")  # This is part of the test
    def test_duplicated_client_nodes(self):
        """Test compile_pymc can handle duplicated (mergeable) RV updates."""
        rng = pytensor.shared(np.random.default_rng(1))
        x = pt.random.normal(rng=rng)
        y = x.owner.clone().default_output()

        fn = compile_pymc([], [x, y], random_seed=1)
        res_x1, res_y1 = fn()
        assert res_x1 == res_y1
        res_x2, res_y2 = fn()
        assert res_x2 == res_y2
        assert res_x1 != res_x2

    def test_nested_updates(self):
        rng = pytensor.shared(np.random.default_rng())
        next_rng1, x = pt.random.normal(rng=rng).owner.outputs
        next_rng2, y = pt.random.normal(rng=next_rng1).owner.outputs
        next_rng3, z = pt.random.normal(rng=next_rng2).owner.outputs

        collect_default_updates(inputs=[], outputs=[x, y, z]) == {rng: next_rng3}

        fn = compile_pymc([], [x, y, z], random_seed=514)
        assert not set(np.array(fn())) & set(np.array(fn()))

        # A local myopic rule (as PyMC used before, would not work properly)
        fn = pytensor.function([], [x, y, z], updates={rng: next_rng1})
        assert set(np.array(fn())) & set(np.array(fn()))

    def test_collect_default_updates_must_be_shared(self):
        shared_rng = pytensor.shared(np.random.default_rng())
        nonshared_rng = shared_rng.type()

        next_rng_of_shared, x = pt.random.normal(rng=shared_rng).owner.outputs
        next_rng_of_nonshared, y = pt.random.normal(rng=nonshared_rng).owner.outputs

        res = collect_default_updates(inputs=[nonshared_rng], outputs=[x, y])
        assert res == {shared_rng: next_rng_of_shared}

        res = collect_default_updates(inputs=[nonshared_rng], outputs=[x, y], must_be_shared=False)
        assert res == {shared_rng: next_rng_of_shared, nonshared_rng: next_rng_of_nonshared}

    def test_scan_updates(self):
        def step_with_update(x, rng):
            next_rng, x = pm.Normal.dist(x, rng=rng).owner.outputs
            return x, {rng: next_rng}

        def step_wo_update(x, rng):
            return step_with_update(x, rng)[0]

        rng = pytensor.shared(np.random.default_rng())

        xs, next_rng = scan(
            fn=step_wo_update,
            outputs_info=[pt.zeros(())],
            non_sequences=[rng],
            n_steps=10,
            name="test_scan",
        )

        assert not next_rng

        with pytest.raises(
            ValueError,
            match="No update found for at least one RNG used in Scan Op",
        ):
            collect_default_updates([xs])

        ys, next_rng = scan(
            fn=step_with_update,
            outputs_info=[pt.zeros(())],
            non_sequences=[rng],
            n_steps=10,
        )

        assert collect_default_updates([ys]) == {rng: next(iter(next_rng.values()))}

        fn = compile_pymc([], ys, random_seed=1)
        assert not (set(fn()) & set(fn()))

    def test_op_from_graph_updates(self):
        rng = pytensor.shared(np.random.default_rng())
        next_rng_, x_ = pt.random.normal(size=(10,), rng=rng).owner.outputs

        x = OpFromGraph([], [x_])()
        with pytest.raises(
            ValueError,
            match="No update found for at least one RNG used in OpFromGraph Op",
        ):
            collect_default_updates([x])

        next_rng, x = OpFromGraph([], [next_rng_, x_])()
        assert collect_default_updates([x]) == {rng: next_rng}
        fn = compile_pymc([], x, random_seed=1)
        assert not (set(fn()) & set(fn()))


def test_replace_rng_nodes():
    rng = pytensor.shared(np.random.default_rng())
    x = pt.random.normal(rng=rng)
    x_rng, *x_non_rng_inputs = x.owner.inputs

    cloned_x = x.owner.clone().default_output()
    cloned_x_rng, *cloned_x_non_rng_inputs = cloned_x.owner.inputs

    # RNG inputs are the same across the two variables
    assert x_rng is cloned_x_rng

    (new_x,) = replace_rng_nodes([cloned_x])
    new_x_rng, *new_x_non_rng_inputs = new_x.owner.inputs

    # Variables are still the same
    assert new_x is cloned_x

    # RNG inputs are not the same as before
    assert new_x_rng is not x_rng

    # All other inputs are the same as before
    for non_rng_inputs, new_non_rng_inputs in zip(x_non_rng_inputs, new_x_non_rng_inputs):
        assert non_rng_inputs is new_non_rng_inputs


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

    rngs = [pytensor.shared(np.random.Generator(default_rng())) for _ in range(2)]
    for rng, bit_generator in zip(rngs, bit_generators):
        assert rng.get_value().bit_generator.state != bit_generator.state

    reseed_rngs(rngs, seed)
    for rng, bit_generator in zip(rngs, bit_generators):
        assert rng.get_value().bit_generator.state == bit_generator.state


class TestConstantFold:
    def test_constant_fold(self):
        x = pt.random.normal(size=(5,))
        y = pt.arange(x.size)

        res = constant_fold((y, y.shape))
        assert np.array_equal(res[0], np.arange(5))
        assert tuple(res[1]) == (5,)

    def test_constant_fold_raises(self):
        size = pytensor.shared(5)
        x = pt.random.normal(size=(size,))
        y = pt.arange(x.size)

        with pytest.raises(NotConstantValueError):
            constant_fold((y, y.shape))

        res = constant_fold((y, y.shape), raise_not_constant=False)
        assert tuple(res[1].eval()) == (5,)

    def test_inputs_preserved(self):
        # Make sure constant_folded graph depends on original graph inputs (not copies)
        # Regression test for #7387
        a = pt.scalar("a", dtype="int")
        out = pt.empty((a,))
        (out_shape,) = constant_fold((out.shape[0],), raise_not_constant=False)
        assert out_shape is a

    def test_constant_fold_alloc(self):
        # By default, Alloc outputs cannot be constant folded
        x = pt.alloc(pt.arange(5), 2, 5)
        np.testing.assert_allclose(constant_fold([x])[0], np.broadcast_to(np.arange(5), (2, 5)))


def test_replace_vars_in_graphs():
    inp = shared(0.0, name="inp")
    x = pm.Normal.dist(inp)

    replacements = {inp: inp + 100}
    [new_x] = replace_vars_in_graphs([x], replacements=replacements)

    assert x.eval() < 50
    assert new_x.eval() > 50


def test_replace_vars_in_graphs_nested_reference():
    # Replace both `x` and `y`, where the replacement of y references `x`
    x = pm.HalfNormal.dist(1e-3, name="x")
    neg_x = -x
    y = pm.Uniform.dist(neg_x, x, name="y")
    x_value = x.clone()
    y_value = y.clone()
    replacements = {x: x_value, y: neg_x + y_value}
    [new_x, new_y] = replace_vars_in_graphs([x, y], replacements=replacements)
    assert new_x.eval({x_value: 100}) == 100
    assert new_y.eval({x_value: 100, y_value: 1}) == -99
    assert new_y.eval({neg_x: 100, y_value: 1}) == 101
    assert np.abs(x.eval()) < 1
    # Confirm the original `y` variable is changed in place
    # This is unavoidable if we want to respect the identity of the replacement variables
    # As when imputing `neg_x` and `x` while evaluating `new_y` above and below.
    # This assertion could fail with probability 1/10000
    assert np.abs(y.eval({x_value: 10000})) > 1

    # Only replace `y`, same replacement as before
    x = pm.HalfNormal.dist(1e-3, name="x")
    neg_x = -x
    y = pm.Uniform.dist(neg_x, x, name="y")
    y_value = y.clone()
    replacements = {y: neg_x + y_value}
    [new_y] = replace_vars_in_graphs([y], replacements=replacements)
    assert np.abs(new_y.eval({y_value: 0})) < 1
    # Confirm that `x` and `neg_x` are still in the graph of `new_y` and that we can impute either
    assert new_y.eval({x: 100, y_value: 1}) == -99
    assert new_y.eval({neg_x: 100, y_value: 1}) == 101
    assert np.abs(x.eval()) < 1
    # In this case the original `y` is not altered, because we did not replace `x`
    assert np.abs(y.eval()) < 1

    # Replacement introduces equivalent but not identical operations
    x = pm.HalfNormal.dist(1e-3, name="x")
    neg_x = -x
    neg_x.name = "neg_x"
    y = pm.Uniform.dist(neg_x, x, name="y")
    x_value = x.clone()
    y_value = y.clone()
    # We clone neg_x!
    replacements = {x: x_value, y: neg_x.owner.clone().outputs[0] + y_value}
    [new_x, new_y] = replace_vars_in_graphs([x, y], replacements=replacements)
    assert new_x.eval({x_value: 100}) == 100
    assert new_y.eval({x_value: 100, y_value: 1}) == -99
    # This now fails because the original `neg_x` is not in the replaced graph!
    with pytest.raises(UnusedInputError, match="neg_x"):
        new_y.eval({neg_x: 100, y_value: 1})
    # We can retrieve the cloned variable by name
    assert new_y.eval({"neg_x": 100, y_value: 1}) == 101
    assert np.abs(x.eval()) < 1
    # Confirm the original `y` variable is not changed in place
    assert np.abs(y.eval()) < 1


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("func", (hessian, hessian_diag))
def test_hessian_sign_change_warning(func):
    x = pt.vector("x")
    f = (x**2).sum()
    with pytest.warns(
        FutureWarning,
        match="will stop negating the output",
    ):
        res_neg = func(f, vars=[x])
    res = func(f, vars=[x], negate_output=False)
    assert equal_computations([res_neg], [-res])
