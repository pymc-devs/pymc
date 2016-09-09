import numpy as np

from pymc3 import *

# Generate data
size = 50
true_intercept = 1
true_slope = 2

x = np.linspace(0, 1, size)
y = true_intercept + x * true_slope + np.random.normal(scale=.5, size=size)

# Add outliers
x = np.append(x, [.1, .15, .2])
y = np.append(y, [8, 6, 9])

data_outlier = dict(x=x, y=y)

with Model() as model:
    family = glm.families.StudentT(  # link=glm.families.identity,
        priors={'nu': 1.5,
                'lam': Uniform.dist(0, 20)})
    glm.glm('y ~ x', data_outlier, family=family)


def run(n=2000):
    if n == "short":
        n = 50
    import matplotlib.pyplot as plt

    with model:
        trace = sample(n, Slice())

    plt.plot(x, y, 'x')
    glm.plot_posterior_predictive(trace)
    plt.show()

if __name__ == '__main__':
    run()
