import matplotlib.pyplot as plt
import pymc3 as pm
import numpy as np

# import pydevd
# pydevd.set_pm_excepthook()
np.seterr(invalid='raise')

data = np.random.normal(size=(2, 20))


model = pm.Model()

with model:
    x = pm.Normal('x', mu=.5, tau=2. ** -2, shape=(2, 1))
    z = pm.Beta('z', alpha=10, beta=5.5)
    d = pm.Normal('data', mu=x, tau=.75 ** -2, observed=data)
    step = pm.NUTS()


def run(n=1000):
    if n == "short":
        n = 50
    with model:
        trace = pm.sample(n, step)

    plt.subplot(2, 2, 1)
    plt.plot(trace[x][:, 0, 0])
    plt.subplot(2, 2, 2)
    plt.hist(trace[x][:, 0, 0])

    plt.subplot(2, 2, 3)
    plt.plot(trace[x][:, 1, 0])
    plt.subplot(2, 2, 4)
    plt.hist(trace[x][:, 1, 0])
    plt.show()

if __name__ == '__main__':
    run()
