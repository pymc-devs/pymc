import pymc3 as pm
from .helpers import SeededTest
import numpy as np
import theano


class TestShared(SeededTest):
    def test_deterministic(self):
        with pm.Model() as model:
            data_values = np.array([.5, .4, 5, 2])
            X = theano.shared(np.asarray(data_values, dtype=theano.config.floatX), borrow=True)
            pm.Normal('y', 0, 1, observed=X)
            model.logp(model.test_point)

    def test_sample_ppc(self):
        x = np.random.normal(size=100)
        y = x + np.random.normal(scale=1e-2, size=100)

        x_pred = np.linspace(-3, 3, 200)

        x_shared = theano.shared(x)

        with pm.Model() as model:
            b = pm.Normal('b', 0., 10.)
            pm.Normal('obs', b * x_shared, np.sqrt(1e-2), observed=y)

            trace = pm.sample(1000, init=None, progressbar=False)

            x_shared.set_value(x_pred)
            pp_trace = pm.sample_ppc(trace, 1000)

        np.testing.assert_allclose(x_pred, pp_trace['obs'].mean(axis=0), atol=1e-1)
