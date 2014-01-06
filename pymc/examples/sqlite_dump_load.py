import os
import numpy as np
import numpy.testing as npt

import pymc as pm

# import pydevd
# pydevd.set_pm_excepthook()
np.seterr(invalid='raise')

data = np.random.normal(size=(2, 20))
model = pm.Model()

with model:
    x = pm.Normal('x', mu=.5, tau=2. ** -2, shape=(2, 1))
    z = pm.Beta('z', alpha=10, beta=5.5)
    d = pm.Normal('data', mu=x, tau=.75 ** -2, observed=data)


def run(n=50):
    if n == 'short':
        n = 5
    with model:
        try:
            trace = pm.sample(n, step=pm.Metropolis(),
                              db=pm.backends.SQLite('test.db'),
                              threads=2)
            dumped = pm.backends.sqlite.load('test.db')

            assert trace[x][0].shape[0] == n
            assert trace[x][1].shape[0] == n
            assert trace.get_values('z', burn=3,
                                   combine=True).shape[0] == n * 2 - 3 * 2

            assert trace.nchains == dumped.nchains
            assert list(sorted(trace.var_names)) == list(sorted(dumped.var_names))

            for chain in trace.chains:
                for var_name in trace.var_names:
                    data = trace.samples[chain][var_name]
                    dumped_data = dumped.samples[chain][var_name]
                    npt.assert_equal(data, dumped_data)
        finally:
            try:
                os.remove('test.db')
            except FileNotFoundError:
                pass

if __name__ == '__main__':
    run('short')
