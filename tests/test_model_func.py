import pymc as pm 
from models import *
from checks import *

def test_lop():
    start, model,_ = simple_model()

    lp = model.logp()

    lp(start)


def test_dlogp():
    start, model, (mu, sig) = simple_model()
    
    close_to(model.dlogp(n = 1)(start), -(start['x'] - mu)/sig, 1./sig/100.)


def test_dlogp2():
    start, model, (mu, sig) = mv_simple()
    H = np.linalg.inv(sig)

    close_to(model.dlogp(n = 2)(start), -H, np.abs(H/100.))
