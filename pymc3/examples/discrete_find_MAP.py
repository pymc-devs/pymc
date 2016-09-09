# Using `find_MAP` on models with discrete variables

# Maximum a posterior(MAP) estimation, can be difficult in models which have
# discrete stochastic variables. Here we demonstrate the problem with a simple
# model, and present a few possible work arounds.

import pymc3 as mc

# We define a simple model of a survey with one data point. We use a $Beta$
# distribution for the $p$ parameter in a binomial. We would like to know both
# the posterior distribution for p, as well as the predictive posterior
# distribution over the survey parameter.

alpha = 4
beta = 4
n = 20
yes = 15

with mc.Model() as model:
    p = mc.Beta('p', alpha, beta)
    surv_sim = mc.Binomial('surv_sim', n=n, p=p)
    surv = mc.Binomial('surv', n=n, p=p, observed=yes)

# First let's try and use `find_MAP`.

with model:
    print(mc.find_MAP())

# `find_map` defaults to find the MAP for only the continuous variables we have
# to specify if we would like to use the discrete variables.

with model:
    print(mc.find_MAP(vars=model.vars))

# We set the `disp` variable to display a warning that we are using a
# non-gradient minimization technique, as discrete variables do not give much
# gradient information. To demonstrate this, if we use a gradient based
# minimization, `fmin_bfgs`, with various starting points we see that the map
# does not converge.

with model:
    for i in range(n + 1):
        s = {'p_logodds_': 0.5, 'surv_sim': i}
        map_est = mc.find_MAP(start=s, vars=model.vars,
                              fmin=mc.starting.optimize.fmin_bfgs)
        print('surv_sim: %i->%i, p: %f->%f, LogP:%f' % (s['surv_sim'],
                                                        map_est['surv_sim'],
                                                        s['p_logodds_'],
                                                        map_est['p_logodds_'],
                                                        model.logp(map_est)))

# Once again because the gradient of `surv_sim` provides no information to the
# `fmin` routine and it is only changed in a few cases, most of which are not
# correct. Manually, looking at the log proability we can see that the maximum
# is somewhere around `surv_sim`$=14$ and `p`$=0.7$. If we employ a
# non-gradient minimization, such as `fmin_powell` (the default when discrete
# variables are detected), we might be able to get a better estimate.

with model:
    for i in range(n + 1):
        s = {'p_logodds_': 0.0, 'surv_sim': i}
        map_est = mc.find_MAP(start=s, vars=model.vars)
        print('surv_sim: %i->%i, p: %f->%f, LogP:%f' % (s['surv_sim'],
                                                        map_est['surv_sim'],
                                                        s['p_logodds_'],
                                                        map_est['p_logodds_'],
                                                        model.logp(map_est)))

# For most starting values this converges to the maximum log likelihood of
# $\approx -3.15$, but for particularly low starting values of `surv_sim`, or
# values near `surv_sim`$=14$ there is still some noise. The scipy optimize
# package contains some more general 'global' minimization functions that we
# can utilize. The `basinhopping` algorithm restarts the optimization at places
# near found minimums. Because it has a slightly different interface to other
# minimization schemes we have to define a wrapper function.


def bh(*args, **kwargs):
    result = mc.starting.optimize.basinhopping(*args, **kwargs)
    # A `Result` object is returned, the argmin value can be in `x`
    return result['x']

with model:
    for i in range(n + 1):
        s = {'p_logodds_': 0.0, 'surv_sim': i}
        map_est = mc.find_MAP(start=s, vars=model.vars, fmin=bh)
        print('surv_sim: %i->%i, p: %f->%f, LogP:%f' % (s['surv_sim'],
                                                        map_est['surv_sim'],
                                                        s['p_logodds_'],
                                                        map_est['p_logodds_'],
                                                        model.logp(map_est)))

# By default `basinhopping` uses a gradient minimization technique,
# `fmin_bfgs`, resulting in inaccurate predictions many times. If we force
# `basinhoping` to use a non-gradient technique we get much better results

with model:
    for i in range(n + 1):
        s = {'p_logodds_': 0.0, 'surv_sim': i}
        map_est = mc.find_MAP(start=s, vars=model.vars,
                              fmin=bh, minimizer_kwargs={"method": "Powell"})
        print('surv_sim: %i->%i, p: %f->%f, LogP:%f' % (s['surv_sim'],
                                                        map_est['surv_sim'],
                                                        s['p_logodds_'],
                                                        map_est['p_logodds_'],
                                                        model.logp(map_est)))

# Confident in our MAP estimate we can sample from the posterior, making sure
# we use the `Metropolis` method for our discrete variables.

with model:
    step1 = mc.step_methods.HamiltonianMC(vars=[p])
    step2 = mc.step_methods.Metropolis(vars=[surv_sim])

with model:
    trace = mc.sample(25000, [step1, step2], start=map_est)

mc.traceplot(trace)
