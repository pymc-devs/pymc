from pymc import sample, psample
from models import simple_init

def test_sample():
    start, step,_ = simple_init()

    for     samplr  in [sample, psample]: 
        for n       in [0, 10, 1000]:
            yield samplr, n, step, start



