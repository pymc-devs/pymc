from pymc import Model, Normal, metropolis_step
import numpy as np 

def simple_init(): 
    start, model = simple_model()

    step = metropolis_step(model, model.vars, np.diag([1.]))
    return start, step


def simple_model():
    start = {'x' : .1}
    model = Model(start)
    Var = model.Var

    x = Var('x', Normal(0,1))
    return start, model
