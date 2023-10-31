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
import warnings

import numpy as np
import numpy.ma as ma
import numpy.testing as npt
import pandas as pd
import pytensor
import pytensor.tensor as pt
import pytest
import scipy.sparse as sps

from pytensor import scan, shared
from pytensor.compile.builders import OpFromGraph
from pytensor.graph.basic import Variable, equal_computations
from pytensor.tensor.random.basic import normal, uniform
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.random.var import RandomStateSharedVariable
from pytensor.tensor.subtensor import AdvancedIncSubtensor, AdvancedIncSubtensor1
from pytensor.tensor.variable import TensorVariable

import pymc as pm

from pymc.distributions.dist_math import check_parameters
from pymc.distributions.distribution import SymbolicRandomVariable
from pymc.distributions.transforms import Interval
from pymc.exceptions import NotConstantValueError
from pymc.logprob.utils import ParameterValueError
from pymc.pytensorf import (
    _replace_vars_in_graphs,
    collect_default_updates,
    compile_pymc,
    constant_fold,
    convert_observed_data,
    extract_obs_data,
    replace_rng_nodes,
    replace_rvs_by_values,
    reseed_rngs,
    rvs_to_value_vars,
    walk_model,
)
from pymc.testing import assert_no_rvs
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
        extract_obs_data(pt.matrix())

    data = np.random.normal(size=(2, 3))
    data_at = pt.as_tensor(data)
    mask = np.random.binomial(1, 0.5, size=(2, 3)).astype(bool)

    for val_at in (data_at, pytensor.shared(data)):
        res = extract_obs_data(val_at)

        assert isinstance(res, np.ndarray)
        assert np.array_equal(res, data)

    # AdvancedIncSubtensor check
    data_m = np.ma.MaskedArray(data, mask)
    missing_values = data_at.type()[mask]
    constant = pt.as_tensor(data_m.filled())
    z_at = pt.set_subtensor(constant[mask.nonzero()], missing_values)

    assert isinstance(z_at.owner.op, (AdvancedIncSubtensor, AdvancedIncSubtensor1))

    res = extract_obs_data(z_at)

    assert isinstance(res, np.ndarray)
    assert np.ma.allequal(res, data_m)

    # AdvancedIncSubtensor1 check
    data = np.random.normal(size=(3,))
    data_at = pt.as_tensor(data)
    mask = np.random.binomial(1, 0.5, size=(3,)).astype(bool)

    data_m = np.ma.MaskedArray(data, mask)
    missing_values = data_at.type()[mask]
    constant = pt.as_tensor(data_m.filled())
    z_at = pt.set_subtensor(constant[mask.nonzero()], missing_values)

    assert isinstance(z_at.owner.op, (AdvancedIncSubtensor, AdvancedIncSubtensor1))

    res = extract_obs_data(z_at)

    assert isinstance(res, np.ndarray)
    assert np.ma.allequal(res, data_m)

    # Cast check
    data = np.array(5)
    t = pt.cast(pt.as_tensor(5.0), np.int64)
    res = extract_obs_data(t)

    assert isinstance(res, np.ndarray)
    assert np.array_equal(res, data)


@pytest.mark.parametrize("input_dtype", ["int32", "int64", "float32", "float64"])
def test_convert_observed_data(input_dtype):
    """
    Ensure that convert_observed_data returns the dense array, masked array,
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

    # Check function behavior with generator data
    generator_output = func(square_generator)

    # Output is wrapped with `pm.floatX`, and this unwraps
    wrapped = generator_output.owner.inputs[0]
    # Make sure the returned object has .set_gen and .set_default methods
    assert hasattr(wrapped, "set_gen")
    assert hasattr(wrapped, "set_default")
    # Make sure the returned object is an PyTensor TensorVariable
    assert isinstance(wrapped, TensorVariable)


def test_pandas_to_array_pandas_index():
    data = pd.Index([1, 2, 3])
    result = convert_observed_data(data)
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

    res = list(walk_model((test_graph,)))
    assert a in res
    assert b in res
    assert c in res
    assert d in res
    assert e in res

    res = list(walk_model((test_graph,), stop_at_vars={c}))
    assert a not in res
    assert b not in res
    assert c in res
    assert d in res
    assert e in res

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

    def test_compile_pymc_symbolic_rv_update(self):
        """Test that SymbolicRandomVariable Op update methods are used by compile_pymc"""

        class NonSymbolicRV(OpFromGraph):
            def update(self, node):
                return {node.inputs[0]: node.outputs[0]}

        rng = pytensor.shared(np.random.default_rng())
        dummy_rng = rng.type()
        dummy_next_rng, dummy_x = NonSymbolicRV(
            [dummy_rng], pt.random.normal(rng=dummy_rng).owner.outputs
        )(rng)

        # Check that there are no updates at first
        fn = compile_pymc(inputs=[], outputs=dummy_x)
        assert fn() == fn()

        # And they are enabled once the Op is registered as a SymbolicRV
        SymbolicRandomVariable.register(NonSymbolicRV)
        fn = compile_pymc(inputs=[], outputs=dummy_x, random_seed=431)
        assert fn() != fn()

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

    def test_multiple_updates_same_variable(self):
        # Raise if unexpected warning is issued
        with warnings.catch_warnings():
            warnings.simplefilter("error")

            rng = pytensor.shared(np.random.default_rng(), name="rng")
            x = pt.random.normal(rng=rng)
            y = pt.random.normal(rng=rng)

            # No warnings if only one variable is used
            assert compile_pymc([], [x])
            assert compile_pymc([], [y])

            user_warn_msg = "RNG Variable rng has multiple clients"
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

    def test_nested_updates(self):
        rng = pytensor.shared(np.random.default_rng())
        next_rng1, x = pt.random.normal(rng=rng).owner.outputs
        next_rng2, y = pt.random.normal(rng=next_rng1).owner.outputs
        next_rng3, z = pt.random.normal(rng=next_rng2).owner.outputs

        collect_default_updates(inputs=[], outputs=[x, y, z]) == {rng: next_rng3}

        fn = compile_pymc([], [x, y, z], random_seed=514)
        assert not set(list(np.array(fn()))) & set(list(np.array(fn())))

        # A local myopic rule (as PyMC used before, would not work properly)
        fn = pytensor.function([], [x, y, z], updates={rng: next_rng1})
        assert set(list(np.array(fn()))) & set(list(np.array(fn())))

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

        assert collect_default_updates([ys]) == {rng: tuple(next_rng.values())[0]}

        fn = compile_pymc([], ys, random_seed=1)
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

    rngs = [
        pytensor.shared(rng_type(default_rng()))
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


def test_constant_fold():
    x = pt.random.normal(size=(5,))
    y = pt.arange(x.size)

    res = constant_fold((y, y.shape))
    assert np.array_equal(res[0], np.arange(5))
    assert tuple(res[1]) == (5,)


def test_constant_fold_raises():
    size = pytensor.shared(5)
    x = pt.random.normal(size=(size,))
    y = pt.arange(x.size)

    with pytest.raises(NotConstantValueError):
        constant_fold((y, y.shape))

    res = constant_fold((y, y.shape), raise_not_constant=False)
    assert tuple(res[1].eval()) == (5,)


class TestReplaceRVsByValues:
    @pytest.mark.parametrize("symbolic_rv", (False, True))
    @pytest.mark.parametrize("apply_transforms", (True, False))
    @pytest.mark.parametrize("test_deprecated_fn", (True, False))
    def test_basic(self, symbolic_rv, apply_transforms, test_deprecated_fn):
        # Interval transform between last two arguments
        interval = (
            Interval(bounds_fn=lambda *args: (args[-2], args[-1])) if apply_transforms else None
        )

        with pm.Model() as m:
            a = pm.Uniform("a", 0.0, 1.0)
            if symbolic_rv:
                raw_b = pm.Uniform.dist(0, a + 1.0)
                b = pm.Censored("b", raw_b, lower=0, upper=a + 1.0, transform=interval)
                # If not True, another distribution has to be used
                assert isinstance(b.owner.op, SymbolicRandomVariable)
            else:
                b = pm.Uniform("b", 0, a + 1.0, transform=interval)
            c = pm.Normal("c")
            d = pt.log(c + b) + 2.0

        a_value_var = m.rvs_to_values[a]
        assert m.rvs_to_transforms[a] is not None

        b_value_var = m.rvs_to_values[b]
        c_value_var = m.rvs_to_values[c]

        if test_deprecated_fn:
            with pytest.warns(FutureWarning, match="Use model.replace_rvs_by_values instead"):
                (res,) = rvs_to_value_vars((d,), apply_transforms=apply_transforms)
        else:
            (res,) = replace_rvs_by_values(
                (d,),
                rvs_to_values=m.rvs_to_values,
                rvs_to_transforms=m.rvs_to_transforms,
            )

        assert res.owner.op == pt.add
        log_output = res.owner.inputs[0]
        assert log_output.owner.op == pt.log
        log_add_output = res.owner.inputs[0].owner.inputs[0]
        assert log_add_output.owner.op == pt.add
        c_output = log_add_output.owner.inputs[0]

        # We make sure that the random variables were replaced
        # with their value variables
        assert c_output == c_value_var
        b_output = log_add_output.owner.inputs[1]
        # When transforms are applied, the input is the back-transformation of the value_var,
        # otherwise it is the value_var itself
        if apply_transforms:
            assert b_output != b_value_var
        else:
            assert b_output == b_value_var

        res_ancestors = list(walk_model((res,)))
        res_rv_ancestors = [
            v for v in res_ancestors if v.owner and isinstance(v.owner.op, RandomVariable)
        ]

        # There shouldn't be any `RandomVariable`s in the resulting graph
        assert len(res_rv_ancestors) == 0
        assert b_value_var in res_ancestors
        assert c_value_var in res_ancestors
        # When transforms are used, `d` depends on `a` through the back-transformation of
        # `b`, otherwise there is no direct connection between `d` and `a`
        if apply_transforms:
            assert a_value_var in res_ancestors
        else:
            assert a_value_var not in res_ancestors

    @pytest.mark.parametrize("test_deprecated_fn", (True, False))
    def test_unvalued_rv(self, test_deprecated_fn):
        with pm.Model() as m:
            x = pm.Normal("x")
            y = pm.Normal.dist(x)
            z = pm.Normal("z", y)
            out = z + y

        x_value = m.rvs_to_values[x]
        z_value = m.rvs_to_values[z]

        if test_deprecated_fn:
            with pytest.warns(FutureWarning, match="Use model.replace_rvs_by_values instead"):
                (res,) = rvs_to_value_vars((out,))
        else:
            (res,) = replace_rvs_by_values(
                (out,),
                rvs_to_values=m.rvs_to_values,
                rvs_to_transforms=m.rvs_to_transforms,
            )

        assert res.owner.op == pt.add
        assert res.owner.inputs[0] is z_value
        res_y = res.owner.inputs[1]
        # Graph should have be cloned, and therefore y and res_y should have different ids
        assert res_y is not y
        assert res_y.owner.op == pt.random.normal
        assert res_y.owner.inputs[3] is x_value

    @pytest.mark.parametrize("test_deprecated_fn", (True, False))
    def test_no_change_inplace(self, test_deprecated_fn):
        # Test that calling rvs_to_value_vars in models with nested transformations
        # does not change the original rvs in place. See issue #5172
        with pm.Model() as m:
            one = pm.LogNormal("one", mu=0)
            two = pm.LogNormal("two", mu=pt.log(one))

            # We add potentials or deterministics that are not in topological order
            pm.Potential("two_pot", two)
            pm.Potential("one_pot", one)

        before = pytensor.clone_replace(m.free_RVs)

        # This call would change the model free_RVs in place in #5172
        if test_deprecated_fn:
            with pytest.warns(FutureWarning, match="Use model.replace_rvs_by_values instead"):
                rvs_to_value_vars(m.potentials)
        else:
            replace_rvs_by_values(
                m.potentials,
                rvs_to_values=m.rvs_to_values,
                rvs_to_transforms=m.rvs_to_transforms,
            )

        after = pytensor.clone_replace(m.free_RVs)
        assert equal_computations(before, after)

    @pytest.mark.parametrize("test_deprecated_fn", (True, False))
    @pytest.mark.parametrize("reversed", (False, True))
    def test_interdependent_transformed_rvs(self, reversed, test_deprecated_fn):
        # Test that nested transformed variables, whose transformed values depend on other
        # RVs are properly replaced
        with pm.Model() as m:
            transform = pm.distributions.transforms.Interval(
                bounds_fn=lambda *inputs: (inputs[-2], inputs[-1])
            )
            x = pm.Uniform("x", lower=0, upper=1, transform=transform)
            y = pm.Uniform("y", lower=0, upper=x, transform=transform)
            z = pm.Uniform("z", lower=0, upper=y, transform=transform)
            w = pm.Uniform("w", lower=0, upper=z, transform=transform)

        rvs = [x, y, z, w]
        if reversed:
            rvs = rvs[::-1]

        if test_deprecated_fn:
            with pytest.warns(FutureWarning, match="Use model.replace_rvs_by_values instead"):
                transform_values = rvs_to_value_vars(rvs)
        else:
            transform_values = replace_rvs_by_values(
                rvs,
                rvs_to_values=m.rvs_to_values,
                rvs_to_transforms=m.rvs_to_transforms,
            )

        for transform_value in transform_values:
            assert_no_rvs(transform_value)

        if reversed:
            transform_values = transform_values[::-1]
        transform_values_fn = m.compile_fn(transform_values, point_fn=False)

        x_interval_test_value = np.random.rand()
        y_interval_test_value = np.random.rand()
        z_interval_test_value = np.random.rand()
        w_interval_test_value = np.random.rand()

        # The 3 Nones correspond to unused rng, dtype and size arguments
        expected_x = transform.backward(x_interval_test_value, None, None, None, 0, 1).eval()
        expected_y = transform.backward(
            y_interval_test_value, None, None, None, 0, expected_x
        ).eval()
        expected_z = transform.backward(
            z_interval_test_value, None, None, None, 0, expected_y
        ).eval()
        expected_w = transform.backward(
            w_interval_test_value, None, None, None, 0, expected_z
        ).eval()

        np.testing.assert_allclose(
            transform_values_fn(
                x_interval__=x_interval_test_value,
                y_interval__=y_interval_test_value,
                z_interval__=z_interval_test_value,
                w_interval__=w_interval_test_value,
            ),
            [expected_x, expected_y, expected_z, expected_w],
        )

    def test_replace_input(self):
        inp = shared(0.0, name="inp")
        x = pm.Normal.dist(inp)

        assert x.eval() < 50

        new_inp = inp + 100

        def replacement_fn(var, replacements):
            if var is x:
                replacements[x.owner.inputs[3]] = new_inp

            return []

        [new_x], _ = _replace_vars_in_graphs([x], replacement_fn=replacement_fn)

        assert new_x.eval() > 50
