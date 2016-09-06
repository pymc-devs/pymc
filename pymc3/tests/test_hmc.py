import pymc3 as pm
import numpy as np
from . import models
from pymc3.step_methods.hmc import leapfrog, Hamiltonian
from .checks import close_to
from ..blocking import DictToArrayBijection


def test_leapfrog_reversible():
    n = 3
    start, model, _ = models.non_normal(n)

    with model:
        h = pm.find_hessian(start, model=model)
        step = pm.HamiltonianMC(model.vars, h, model=model)

    bij = DictToArrayBijection(step.ordering, start)

    logp, dlogp = list(map(bij.mapf, step.fs))
    H = Hamiltonian(logp, dlogp, step.potential)

    q0 = bij.map(start)
    p0 = np.ones(n) * .05
    for e in [.01, .1, 1.2]:
        for L in [1, 2, 3, 4, 20]:

            q, p = q0, p0
            q, p = leapfrog(H, q, p, L, e)
            q, p = leapfrog(H, q, -p, L, e)

            close_to(q, q0, 1e-8, str((L, e)))
            close_to(-p, p0, 1e-8, str((L, e)))
