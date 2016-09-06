from pymc3 import *
import theano.tensor as tt
from numpy import random, sum as nsum, concatenate, newaxis, dot
import numpy as np

random.seed(1)

n_groups = 10
no_pergroup = 30
n_observed = no_pergroup * n_groups
n_group_predictors = 1
n_predictors = 3

group = concatenate([[i] * no_pergroup for i in range(n_groups)])
# random.normal(size = (n_groups, n_group_predictors))
group_predictors = random.normal(size=(n_groups, n_group_predictors))
predictors = random.normal(size=(n_observed, n_predictors))

group_effects_a = random.normal(size=(n_group_predictors, n_predictors))
effects_a = random.normal(
    size=(n_groups, n_predictors)) + dot(group_predictors, group_effects_a)

y = nsum(
    effects_a[group, :] * predictors, 1) + random.normal(size=(n_observed))


model = Model()
with model:

    # m_g ~ N(0, .1)
    group_effects = Normal(
        "group_effects", 0, .1, shape=(1, n_group_predictors, n_predictors))

    # sg ~ Uniform(.05, 10)
    sg = Uniform("sg", .05, 10, testval=2.)

    # m ~ N(mg * pg, sg)
    effects = Normal("effects",
                     sum(group_predictors[:, :, newaxis] *
                         group_effects, 1), sg ** -2,
                     shape=(n_groups, n_predictors))

    s = Uniform("s", .01, 10, shape=n_groups)

    g = tt.constant(group)

    # y ~ Normal(m[g] * p, s)
    yd = Normal('y', sum(effects[g] * predictors, 1), s[g] ** -2, observed=y)

    start = find_MAP()
    #h = find_hessian(start)

    step = NUTS(model.vars, scaling=start)


def run(n=3000):
    if n == "short":
        n = 50
    with model:
        trace = sample(n, step, start)

if __name__ == '__main__':
    run()
