import pymc3 as pm
import numpy as np
import theano


def test_deterministic():
    with pm.Model() as model:
        data_values = np.array([.5, .4, 5, 2])
        X = theano.shared(np.asarray(data_values, dtype=theano.config.floatX), borrow=True)
        pm.Normal('y', 0, 1, observed=X)
        model.logp(model.test_point)
