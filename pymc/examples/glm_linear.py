import numpy as np

try:
    import statsmodels.api as sm
except ImportError:
    print("Example requires statsmodels")
    sys.exit(0)

from pymc import *

# Generate data
size = 50
true_intercept = 1
true_slope = 2

x = np.linspace(0, 1, size)
y = true_intercept + x*true_slope + np.random.normal(scale=.5, size=size)

data = dict(x=x, y=y)

with Model() as model:
    glm.glm('y ~ x', data)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    with model:
        trace = sample(2000, Slice(model.vars))

    plt.plot(x, y, 'x')
    glm.plot_posterior_predictive(trace)
    plt.show()
