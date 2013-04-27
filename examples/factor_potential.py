from pymc import * 

with Model() as model: 
    x = Normal('x', 1,1)
    model.AddPotential(-x**2)

    start = model.test_point
    h = find_hessian(start)
    step = Metropolis(model.vars, h)
    trace = sample(3000, step, start)
