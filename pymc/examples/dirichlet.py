import numpy as np
from pymc import *

model = Model()
with model:

    k = 5
    a = constant(np.array([2, 3., 4, 2, 2]))

    p, p_m1 = model.TransformedVar(
        'p', Dirichlet.dist(k, a, shape=k),
        simplextransform)

if __name__ == '__main__':

    with model:
        H = model.fastd2dlogp()

        s = find_MAP()

        step = HamiltonianMC(model.vars, H(s))
        trace = sample(1000, step, s)
