import numpy as np
import numpy.ma as ma
import numpy.testing as npt
import pandas as pd
import pymc3 as pm
from pymc3.model import DependenceDAG
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
        self.expected_full_dag = DependenceDAG()
        self.expected_full_dag.add(self.e)

    def test_built_DependenceDAG(self):
        assert self.expected_full_dag.match(self.model.variable_dependence_dag)

    def test_get_sub_dag(self):
        dag = self.model.variable_dependence_dag
        sub1 = dag.get_sub_dag(self.a)
        assert len(sub1.nodes) == 1
        assert sub1.check_integrity()

        sub2 = dag.get_sub_dag([self.a])
        assert len(sub2.nodes) == 1
        assert sub2.check_integrity()
        assert sub1.match(sub2)

        sub3 = dag.get_sub_dag([self.e])
        assert len(sub3.nodes) == 5
        assert sub3.check_integrity()
        assert sub3.match(dag)

        hard = dag.get_sub_dag([self.e,
                                theano.tensor.exp(self.b +
                                                  self.e * self.e) * self.e *
                                self.b + self.a])
        assert len(hard.nodes) == 6
        assert hard.check_integrity()
        depth = hard.get_node_depths()
        new_node_depth = [depth[n] for n in hard
                          if n not in self.model.basic_RVs +
                          self.model.deterministics][0]
        assert new_node_depth == 4
        assert hard.get_sub_dag(self.e).match(dag)

        params = [self.d,
                  0.,
                  np.zeros((10, 2), dtype=np.float32),
                  theano.tensor.constant(5.354),
                  theano.shared(np.array([2, 6, 8])),
                  ]
        with_non_theano, index = dag.get_sub_dag(params,
                                                 return_index=True)
        assert with_non_theano.check_integrity()
        assert with_non_theano.get_sub_dag(self.d).match(dag.get_sub_dag([self.d]))
        assert index[0] == params[0]
        assert all([isinstance(index[i], WrapAsHashable)
                    for i in range(1, 3)])
        for i in range(1, 3):
            supplied = params[i]
            obj = index[i]
            assert isinstance(obj, WrapAsHashable)
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
        assert with_non_theano.match(wnt2)
        wnt2, node2 = wnt2.add(params[-1], force=True, return_added_node=True)
        assert wnt2.match(with_non_theano)
        assert node2 == index[len(params) - 1]
