from pylab import *
from pymc import *
import numpy as np
import theano

# import pydevd
# pydevd.set_pm_excepthook()
np.seterr(invalid='raise')

data = np.random.normal(size=(2, 20))


model = Model()

with model:
    x = Normal('x', mu=.5, tau=2. ** -2, shape=(2, 1))

    z = Beta('z', alpha=10, beta=5.5)

    d = Normal('data', mu=x, tau=.75 ** -2, observed=data)

    step = NUTS()

    trace = sample(1e3, step)

subplot(2, 2, 1)
plot(trace[x][:, 0, 0])
subplot(2, 2, 2)
hist(trace[x][:, 0, 0])

subplot(2, 2, 3)
plot(trace[x][:, 1, 0])
subplot(2, 2, 4)
hist(trace[x][:, 1, 0])
show()
