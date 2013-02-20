from pymc import Model, Normal, metropolis_step
import numpy as np 
import pymc as pm

def simple_init(): 
    start, model, moments = simple_model()

    step = metropolis_step(model, model.vars, np.diag([1.]))
    return start, step, moments


def simple_model():
    model = Model()
    Var = model.Var

    mu = -2.1
    tau = 1.3
    x = Var('x', Normal(mu,tau), testval = .1)

    return model.test_point, model, (mu, tau**-1)

def mv_simple():
    mu = np.array([-.1,.5, 1.1])
    p = np.array([
        [2. , 0 ,  0],
        [.05 , .1,  0],
        [1. ,-0.05,5.5]])

    tau = np.dot(p,p.T) 

    model = pm.Model()
    Var = model.Var

    x = Var('x', pm.MvNormal(pm.constant(mu),pm.constant(tau)), 3, testval = np.array([.1, 1., .8]))

    H = tau
    C = np.linalg.inv(H)

    return model.test_point, model, (mu, C)
