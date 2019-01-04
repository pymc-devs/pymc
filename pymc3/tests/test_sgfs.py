import numpy as np
import pymc3 as pm
from pymc3 import Model, Normal
import theano.tensor as tt

def test_minibatch():
    draws = 3000
    mu0 = 1
    sd0 = 1
    
    def f(x, a, b, c):
        return a*x**2 + b*x + c
    
    a, b, c = 1, 2, 3

    batch_size = 50
    total_size = batch_size*500
    x_train = np.random.uniform(-10, 10, size=(total_size,)).astype('float32')
    x_obs = pm.data.Minibatch(x_train, batch_size=batch_size)

    y_train = f(x_train, a, b, c) + np.random.normal(size=x_train.shape).astype('float32')
    y_obs = pm.data.Minibatch(y_train, batch_size=batch_size)

    with Model():
        abc = Normal('abc', mu=mu0, sigma=sd0, shape=(3,))
        x = x_obs
        x2 = x**2
        o = tt.ones_like(x)
        X = tt.stack([x2, x, o]).T
        y = X.dot(abc)
        pm.Normal('y', mu=y, observed=y_obs)

        step_method = pm.SGFS(batch_size=batch_size, step_size=1., total_size=total_size)
        trace = pm.sample(draws=draws, step=step_method, init=None, cores=2)

    np.testing.assert_allclose(np.mean(trace['abc'], axis=0), np.asarray([a, b, c]), rtol=0.1)
