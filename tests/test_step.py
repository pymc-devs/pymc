from checks import *
from models import simple_model
from theano.tensor import constant

def check_moment(h, var, moment, value, bound):
    assert assert_almost_equal(
            np.moment(h[var], moment, axis =0),
            value,
            bound)


def test_step_continuous():

    mu = as_tensor_variable(np.array([-.1,.5, 1.1]))
    p = as_tensor_variable(np.array([
        [2. , 0 ,  0],
        [.05 , .1,  0],
        [1. ,-0.05,5.5]]))

    tau = np.dot(p,p.T) 

                    

    start = {'x' : [.1, 1., .8]}
    model = pm.Model(start)
    Var = model.Var

    x = Var('x', pm.MvNormal(mu,tau))

    H = tau

    hmc = pm.hmc_step(model, H)
    mh = pm.metropolis_step(model, model.vars , H)
    compound = pm.compound_step(hmc, mh)
    steps = [hmc, mh, compound]


    moments = [('x', 0, mu, .01)]

    for st in steps:
        for (var, moment, val, bound) in moments:
            h, _, _ = psample(2000, st, start)

            yield check_moment, h, var, moment, val, bound  



        

