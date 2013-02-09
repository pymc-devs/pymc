from pymc import Model, Normal, sample, psample, metropolis_step
import numpy as np 

def simple_model(): 
    start = {'x' : .1}
    model = Model(start)
    Var = model.Var

    x = Var('x', Normal(0,1))

    step = metropolis_step(model, model.vars, np.diag([1.]))
    return start, step



def test_sample():
    start, step = simple_model()

    for samplr  in [sample, psample]: 
    for n       in [0, 10, 1000]:
        yield samplr, n, step, start



