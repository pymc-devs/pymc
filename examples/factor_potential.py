from pymc import * 

with Model() as model: 
    x = Normal('x', 1,1)
    model.AddPotential(-x**2)

    
    h = find_hessian()
    step = Metropolis(h)
    trace = sample(3000, step)
