import numpy as np
import numpy.ma as ma
import numpy.testing as npt
import pandas as pd
from networkx import DiGraph, is_directed_acyclic_graph, topological_sort
import pymc3 as pm
from pymc3.model import (build_dependence_dag_from_model,
                         matching_dependence_dags,
                         add_to_dependence_dag,
                         get_sub_dag)
from pymc3.util import WrapAsHashable
import scipy.sparse as sps

import theano
import theano.tensor as tt
import theano.sparse as sparse


class TestHelperFunc(object):
    def test_pandas_to_array(self):
        """
        Ensure that pandas_to_array returns the dense array, masked array,
        graph variable, TensorVariable, or sparse matrix as appropriate.
        """
        # Create the various inputs to the function
        sparse_input = sps.csr_matrix(np.eye(3))
        dense_input = np.arange(9).reshape((3, 3))

        input_name = 'input_variable'
        theano_graph_input = tt.as_tensor(dense_input, name=input_name)

        pandas_input = pd.DataFrame(dense_input)

        # All the even numbers are replaced with NaN
        missing_pandas_input = pd.DataFrame(np.array([[np.nan, 1, np.nan],
                                                      [3, np.nan, 5],
                                                      [np.nan, 7, np.nan]]))
        masked_array_input = ma.array(dense_input,
                                      mask=(np.mod(dense_input, 2) == 0))

        # Create a generator object. Apparently the generator object needs to
        # yield numpy arrays.
        square_generator = (np.array([i**2], dtype=int) for i in range(100))

        # Alias the function to be tested
        func = pm.model.pandas_to_array

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
        npt.assert_allclose(sparse_output.toarray(),
                            sparse_input.toarray())

        # Check function behavior when using masked array inputs and pandas
        # objects with missing data
        for input_value in [masked_array_input, missing_pandas_input]:
            func_output = func(input_value)
            assert isinstance(func_output, ma.core.MaskedArray)
            assert func_output.shape == input_value.shape
            npt.assert_allclose(func_output, masked_array_input)

        # Check function behavior with Theano graph variable
        theano_output = func(theano_graph_input)
        assert isinstance(theano_output, theano.gof.graph.Variable)
        assert theano_output.owner.inputs[0].name == input_name

        # Check function behavior with generator data
        generator_output = func(square_generator)

        # Output is wrapped with `pm.floatX`, and this unwraps
        wrapped = generator_output.owner.inputs[0]
        # Make sure the returned object has .set_gen and .set_default methods
        assert hasattr(wrapped, "set_gen")
        assert hasattr(wrapped, "set_default")
        # Make sure the returned object is a Theano TensorVariable
        assert isinstance(wrapped, tt.TensorVariable)

    def test_as_tensor(self):
        """
        Check returned values for `data` given known inputs to `as_tensor()`.

        Note that ndarrays should return a TensorConstant and sparse inputs
        should return a Sparse Theano object.
        """
        # Create the various inputs to the function
        input_name = 'testing_inputs'
        sparse_input = sps.csr_matrix(np.eye(3))
        dense_input = np.arange(9).reshape((3, 3))
        masked_array_input = ma.array(dense_input,
                                      mask=(np.mod(dense_input, 2) == 0))

        # Create a fake model and fake distribution to be used for the test
        fake_model = pm.Model()
        with fake_model:
            fake_distribution = pm.Normal.dist(mu=0, sd=1)
            # Create the testval attribute simply for the sake of model testing
            fake_distribution.testval = None

        # Alias the function to be tested
        func = pm.model.as_tensor

        # Check function behavior using the various inputs
        dense_output = func(dense_input,
                            input_name,
                            fake_model,
                            fake_distribution)
        sparse_output = func(sparse_input,
                             input_name,
                             fake_model,
                             fake_distribution)
        masked_output = func(masked_array_input,
                             input_name,
                             fake_model,
                             fake_distribution)

        # Ensure that the missing values are appropriately set to None
        for func_output in [dense_output, sparse_output]:
            assert func_output.missing_values is None

        # Ensure that the Theano variable names are correctly set.
        # Note that the output for masked inputs do not have their names set
        # to the passed value.
        for func_output in [dense_output, sparse_output]:
            assert func_output.name == input_name

        # Ensure the that returned functions are all of the correct type
        assert isinstance(dense_output, tt.TensorConstant)
        assert sparse.basic._is_sparse_variable(sparse_output)

        # Masked output is something weird. Just ensure it has missing values
        # self.assertIsInstance(masked_output, tt.TensorConstant)
        assert masked_output.missing_values is not None

        return None


class TestDependenceDAG(object):
    def setup_method(self):
        self.obs = np.random.randn(1000,) + 2
        with pm.Model() as model:
            self.a = pm.Normal('a', mu=0, sd=100)
            self.b = pm.Normal('b', mu=self.a, sd=1e-8)
            self.c = pm.Normal('c', mu=self.a, sd=1e-8)
            self.d = pm.Deterministic('d', self.b + self.c)
            self.e = pm.Normal('e', mu=self.d, sd=1, observed=self.obs)
        self.model = model
        self.expected_full_dag = DiGraph()
        add_to_dependence_dag(self.expected_full_dag, self.e)

    def test_built_DependenceDAG(self):
        assert matching_dependence_dags(self.expected_full_dag,
                                        self.model.dependence_dag)
        assert matching_dependence_dags(
            build_dependence_dag_from_model(self.model),
            self.model.dependence_dag)

    def test_get_sub_dag(self):
        dag = self.model.dependence_dag
        sub1 = get_sub_dag(dag, self.a)[0]
        assert len(sub1.nodes) == 1
        assert is_directed_acyclic_graph(sub1)

        sub2 = get_sub_dag(dag, [self.a])[0]
        assert len(sub2.nodes) == 1
        assert is_directed_acyclic_graph(sub2)
        assert matching_dependence_dags(sub1, sub2)

        sub3 = get_sub_dag(dag, [self.e])[0]
        assert len(sub3.nodes) == 5
        assert is_directed_acyclic_graph(sub3)
        assert matching_dependence_dags(sub3, dag)

        hard_expr = (theano.tensor.exp(self.b + self.e * self.e) *
                     self.e * self.b + self.a)
        hard = get_sub_dag(dag, [self.e, hard_expr])[0]
        assert len(hard.nodes) == 6
        assert is_directed_acyclic_graph(hard)
        sorted_nodes = list(topological_sort(hard))
        expected = [(self.a,),
                    (self.b, self.c),
                    (self.b, self.c),
                    (self.d,),
                    (self.e,),
                    (hard_expr,)]
        assert all((n in e for n, e in zip(sorted_nodes, expected)))
        assert matching_dependence_dags(get_sub_dag(hard, self.e)[0], dag)

        params = [self.d,
                  0.,
                  np.zeros((10, 2), dtype=np.float32),
                  theano.tensor.constant(5.354),
                  theano.shared(np.array([2, 6, 8])),
                  ]
        with_non_theano, index = get_sub_dag(dag,
                                             params)
        assert is_directed_acyclic_graph(with_non_theano)
        assert matching_dependence_dags(get_sub_dag(with_non_theano,
                                                    self.d)[0],
                                        get_sub_dag(dag, [self.d])[0])
        assert index[0] == params[0]
        assert all([isinstance(index[i], WrapAsHashable)
                    for i in range(1, 3)])
        for i in range(len(params)):
            supplied = params[i]
            obj = index[i]
            if not isinstance(obj, WrapAsHashable):
                continue
            assert obj in with_non_theano
            if obj.node_is_hashable:
                assert obj.node == supplied
            else:
                assert id(obj.node) == id(params[i])
            obj_value = obj.get_value()
            if isinstance(supplied, theano.tensor.sharedvar.SharedVariable):
                expected_value = supplied.get_value()
            elif isinstance(supplied, theano.tensor.TensorConstant):
                expected_value = supplied.value
            else:
                expected_value = supplied
            if isinstance(obj_value, np.ndarray):
                assert np.all(obj_value == expected_value)
            else:
                assert obj_value == expected_value

        wnt2 = with_non_theano.copy()
        assert matching_dependence_dags(with_non_theano, wnt2)
        node2 = add_to_dependence_dag(wnt2,
                                      params[-1],
                                      force=True)
        assert matching_dependence_dags(wnt2, with_non_theano)
        assert node2 == index[len(params) - 1]
