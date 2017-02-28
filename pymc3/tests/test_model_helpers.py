import unittest

import numpy as np
import numpy.ma as ma
import numpy.testing as npt
import pandas as pd
import pymc3 as pm
import scipy.sparse as sps

import theano
import theano.tensor as tt
import theano.sparse as sparse


class HelperFuncTests(unittest.TestCase):
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
            self.assertIsInstance(func_output, np.ndarray)
            self.assertEqual(func_output.shape, input_value.shape)
            npt.assert_allclose(func_output, dense_input)

        # Check function behavior with sparse matrix inputs
        sparse_output = func(sparse_input)
        self.assertTrue(sps.issparse(sparse_output))
        self.assertEqual(sparse_output.shape, sparse_input.shape)
        npt.assert_allclose(sparse_output.toarray(),
                            sparse_input.toarray())

        # Check function behavior when using masked array inputs and pandas
        # objects with missing data
        for input_value in [masked_array_input, missing_pandas_input]:
            func_output = func(input_value)
            self.assertIsInstance(func_output, ma.core.MaskedArray)
            self.assertEqual(func_output.shape, input_value.shape)
            npt.assert_allclose(func_output, masked_array_input)

        # Check function behavior with Theano graph variable
        theano_output = func(theano_graph_input)
        self.assertIsInstance(theano_output, theano.gof.graph.Variable)
        self.assertEqual(theano_output.name, input_name)

        # Check function behavior with generator data
        generator_output = func(square_generator)
        # Make sure the returned object has .set_gen and .set_default methods
        self.assertTrue(hasattr(generator_output, "set_gen"))
        self.assertTrue(hasattr(generator_output, "set_default"))
        # Make sure the returned object is a Theano TensorVariable
        self.assertIsInstance(generator_output, tt.TensorVariable)

        return None




