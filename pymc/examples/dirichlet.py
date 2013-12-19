import numpy as np
from pymc import *

model = Model()
with model:

    k = 5
    a = constant(np.array([2, 3., 4, 2, 2]))

    p, p_m1 = model.TransformedVar(
        'p', Dirichlet.dist(k, a, shape=k),
        simplextransform)

def run(n=3000):

    with model:
        H = model.fastd2logp()

        s = find_MAP()

        step = HamiltonianMC(model.vars, H(s))
        trace = sample(n, step, s)

if __name__ == '__main__':
    run()


