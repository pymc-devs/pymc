import numpy as np
from pymc3 import *

model = Model()
with model:

    k = 5
    a = constant(np.array([2, 3., 4, 2, 2]))

    p = Dirichlet('p', a, shape=k)

    c = Categorical('c', p, observed=np.random.randint(0, k, 5))


def run(n=3000):
    if n == "short":
        n = 50
    with model:
        step = Slice()
        trace = sample(n, step)

if __name__ == '__main__':
    run()
