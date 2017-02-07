import pymc3 as pm

with pm.Model() as model:
    x = pm.Normal('x', 1, 1)
    x2 = pm.Potential('x2', -x ** 2)

    start = model.test_point
    h = pm.find_hessian(start)
    step = pm.Metropolis(model.vars, h)


def run(n=3000):
    if n == "short":
        n = 50
    with model:
        pm.sample(n, step=step, start=start)

if __name__ == '__main__':
    run()
