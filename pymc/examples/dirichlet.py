import numpy as np
from pymc import *

model = Model()
with model:

    k = 5
    a = constant(np.array([2, 3., 4, 2, 2]))

    p, p_m1 = model.TransformedVar(
        'p', Dirichlet.dist(a, shape=k),
        simplextransform)

    c = Categorical('c', p, observed=np.random.randint(0, k, 5))

def run(n=3000):
    if n == "short":
        n = 50
    with model:
        H = model.fastd2logp()

        s = find_MAP()

        step = HamiltonianMC(model.vars, H(s))
        trace = sample(n, step, s)

if __name__ == '__main__':
    run()


