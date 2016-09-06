import numpy as np
import scipy.optimize as opt

from pymc3 import Model, glm, find_MAP, sample

# Generate data
size = 50
true_intercept = 1
true_slope = 2

x = np.linspace(0, 1, size)
y = true_intercept + x * true_slope + np.random.normal(scale=.5, size=size)

data = dict(x=x, y=y)

with Model() as model:
    glm.glm('y ~ x', data)


def run(n=2000):
    if n == "short":
        n = 50
    import matplotlib.pyplot as plt

    with model:
        start = find_MAP(fmin=opt.fmin_powell)
        trace = sample(n, Slice(), start=start)

    plt.plot(x, y, 'x')
    glm.plot_posterior_predictive(trace)
    # plt.show()

if __name__ == '__main__':
    run()
