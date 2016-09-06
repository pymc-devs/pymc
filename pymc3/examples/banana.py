'''
Created on May 10, 2012

@author: jsalvatier
'''
from pymc3 import *
import numpy as np
import pylab as pl


"""
This model is U shaped because of the non identifiability. I think this is the same as the Rosenbrock function.
As n increases, the walls become steeper but the distribution does not shrink towards the mode.
As n increases this distribution gets harder and harder for HMC to sample.

Low Flip HMC seems to do a bit better.

This example comes from
Discussion of Riemann manifold Langevin and
Hamiltonian Monte Carlo methods by M.
Girolami and B. Calderhead

http://arxiv.org/abs/1011.0057
"""
N = 200
with Model() as model:

    x = Normal('x', 0, 1)
    y = Normal('y', 0, 1)
    N = 200
    d = Normal('d', x + y ** 2, 1., observed=np.zeros(N))

    start = model.test_point
    h = np.ones(2) * np.diag(find_hessian(start))[0]

    step = HamiltonianMC(model.vars, h, path_length=4.)


def run(n=3000):
    if n == "short":
        n = 50
    with model:
        trace = sample(n, step, start)

        pl.figure()
        pl.hexbin(trace['x'], trace['y'])

        # lets plot the samples vs. the actual distribution
        xn = 1500
        yn = 1000

        xs = np.linspace(-3, .25, xn)[np.newaxis, :]
        ys = np.linspace(-1.5, 1.5, yn)[:, np.newaxis]

        like = (xs + ys ** 2) ** 2 * N
        post = np.exp(-.5 * (xs ** 2 + ys ** 2 + like))
        post = post

        pl.figure()
        extent = np.min(xs), np.max(xs), np.min(ys), np.max(ys)
        pl.imshow(post, extent=extent)
