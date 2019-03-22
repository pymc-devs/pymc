import pymc3 as pm
from .helpers import SeededTest
import numpy as np
import pytest


class TestData(SeededTest):
    def test_deterministic(self):
        data_values = np.array([.5, .4, 5, 2])
        with pm.Model() as model:
            X = pm.Data('X', data_values)
            pm.Normal('y', 0, 1, observed=X)
            model.logp(model.test_point)

    def test_sample(self):
        x = np.random.normal(size=100)
        y = x + np.random.normal(scale=1e-2, size=100)

        x_pred = np.linspace(-3, 3, 200, dtype='float32')

        with pm.Model():
            x_shared = pm.Data('x_shared', x)
            b = pm.Normal('b', 0., 10.)
            pm.Normal('obs', b * x_shared, np.sqrt(1e-2), observed=y)
            prior_trace0 = pm.sample_prior_predictive(1000)

            trace = pm.sample(1000, init=None, progressbar=False)
            pp_trace0 = pm.sample_posterior_predictive(trace, 1000)

            x_shared.set_value(x_pred)
            pp_trace1 = pm.sample_posterior_predictive(trace, samples=1000)
            prior_trace1 = pm.sample_prior_predictive(1000)

        assert prior_trace0['b'].shape == (1000,)
        assert prior_trace0['obs'].shape == (1000, 100)
        np.testing.assert_allclose(x, pp_trace0['obs'].mean(axis=0), atol=1e-1)

        assert prior_trace1['b'].shape == (1000,)
        assert prior_trace1['obs'].shape == (1000, 200)
        np.testing.assert_allclose(x_pred, pp_trace1['obs'].mean(axis=0),
                                   atol=1e-1)

    def test_sample_posterior_predictive_after_set_data(self):
        with pm.Model() as model:
            x = pm.Data('x', [1., 2., 3.])
            y = pm.Data('y', [1., 2., 3.])
            beta = pm.Normal('beta', 0, 10.)
            pm.Normal('obs', beta * x, np.sqrt(1e-2), observed=y)
            trace = pm.sample(1000, tune=1000, chains=1)
        # Predict on new data.
        with model:
            x_test = [5, 6, 9]
            pm.set_data(new_data={'x': x_test})
            y_test = pm.sample_posterior_predictive(trace)

        assert y_test['obs'].shape == (1000, 3)
        np.testing.assert_allclose(x_test, y_test['obs'].mean(axis=0),
                                   atol=1e-1)

    def test_sample_after_set_data(self):
        with pm.Model() as model:
            x = pm.Data('x', [1., 2., 3.])
            y = pm.Data('y', [1., 2., 3.])
            beta = pm.Normal('beta', 0, 10.)
            pm.Normal('obs', beta * x, np.sqrt(1e-2), observed=y)
            pm.sample(1000, init=None, tune=1000, chains=1)
        # Predict on new data.
        new_x = [5, 6, 9]
        new_y = [5, 6, 9]
        with model:
            pm.set_data(new_data={'x': new_x, 'y': new_y})
            new_trace = pm.sample()
            pp_trace = pm.sample_posterior_predictive(new_trace, 1000)

        assert pp_trace['obs'].shape == (1000, 3)
        np.testing.assert_allclose(new_y, pp_trace['obs'].mean(axis=0),
                                   atol=1e-1)

    def test_creation_of_data_outside_model_context(self):
        with pytest.raises(TypeError) as error:
            pm.Data('data', [1.1, 2.2, 3.3])
        error.match('No model on context stack')

    def test_set_data_to_non_data_container_variables(self):
        with pm.Model() as model:
            x = np.array([1., 2., 3.])
            y = np.array([1., 2., 3.])
            beta = pm.Normal('beta', 0, 10.)
            pm.Normal('obs', beta * x, np.sqrt(1e-2), observed=y)
            pm.sample(1000, init=None, tune=1000, chains=1)
        with pytest.raises(TypeError) as error:
            pm.set_data({'beta': [1.1, 2.2, 3.3]}, model=model)
        error.match('defined as `pymc3.Data` inside the model')

    def test_model_to_graphviz_for_model_with_data_container(self):
        with pm.Model() as model:
            x = pm.Data('x', [1., 2., 3.])
            y = pm.Data('y', [1., 2., 3.])
            beta = pm.Normal('beta', 0, 10.)
            pm.Normal('obs', beta * x, np.sqrt(1e-2), observed=y)
            pm.sample(1000, init=None, tune=1000, chains=1)

        g = pm.model_to_graphviz(model)
        text = 'x [label="x ~ Deterministic" shape=box style=filled]'
        assert text in g.source
