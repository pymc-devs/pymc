from pymc import *
import numpy as np
import theano

# import pydevd
# pydevd.set_pm_excepthook()
np.seterr(invalid='raise')

n = 3

model = Model()
with model:
    x = Normal('x', 0, 1., shape=n)

    # start sampling at the MAP
    start = find_MAP()

    step = Slicer()

def run(n=3000):
    if n == "short":
        n = 50
    with model:
        trace = sample(n, step, start=start)
        
if __name__ == '__main__':
    run()
