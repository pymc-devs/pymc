from pymc3 import *

with Model() as model:
    x = Normal('x', 1, 1)
    x2 = Potential('x2', -x ** 2)

    start = model.test_point
    h = find_hessian(start)
    step = Metropolis(model.vars, h)


def run(n=3000):
    if n == "short":
        n = 50
    with model:
        trace = sample(n, step, start)

if __name__ == '__main__':
    run()
