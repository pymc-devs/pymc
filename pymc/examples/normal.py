from pymc import *
import numpy as np
import theano

# import pydevd
# pydevd.set_pm_excepthook()
np.seterr(invalid='raise')

data = np.random.normal(size=(3, 20))
n = 1

model = Model()
with model:
    x = Normal('x', 0, 1., shape=n)

    # start sampling at the MAP
    start = find_MAP()

    step = NUTS(scaling=start)

def run(n=3000):
    with model:
        trace = sample(n, step, start=start)
        
if __name__ == '__main__':
    run()
