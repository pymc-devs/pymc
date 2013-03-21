# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from pymc import *
import numpy as np 
from numpy.random import normal, beta
import theano 

# <markdowncell>

# Model
# -----
# We consider the following generative model

# <codecell>

xtrue = normal(scale = 2., size = 1)
ytrue = normal(loc = np.exp(xtrue), scale = 1, size = (2,1))
zdata = normal(loc = xtrue + ytrue, scale = .75, size = (2, 20))

# <markdowncell>

# we observe `zdata` but not `xtrue` or `ytrue`, so we want to come up with posterior distributions for x and y.

# <markdowncell>

# Build Model
# -----------
# The `Model` encapsulates a statistical model. It has very simple internals: just a list of unobserved variables (`Model.vars`) and a list of factors which go into computing the posterior density (`Model.factors`) (see model.py for more).
# 
# The `Var` and `Data` method add unobserved and observed random variables to our model respectively.

# <codecell>

model = Model()
Var = model.Var
Data = model.Data 

# <markdowncell>

# The `Var` method adds an unobserved random variable to the model. `Var` needs the name and prior distribution for the random variable, and optionally the shape of the parameter. 
# It returns a Theano variable which represents (the value of) the random variable we have added to the model.
# 
# The `Var` method is also very simple (see model.py), it creates a Theano variable, adds it to the model
# list and adds the likelihood factor to the model's factor list.
# 
# The distribution classes (see distributions/continuous.py), such as `Normal(mu, tau)` below, take some parameters and have a `logp(value)` method for calculating the likelihood.

# <codecell>

x = Var('x', Normal(mu = 0., tau = 1))
y = Var('y', Normal(mu = exp(x), tau = 2.**-2), shape = (2,1))

# <markdowncell>

# The `Data` method adds an observed random variable to the model. It functions similar to `Var`. `Data` takes the observed data, and the distribution for that data. 

# <codecell>

Data(zdata, Normal(mu = x + y, tau = .75**-2))

# <markdowncell>

# Fit Model
# ---------
# We need a starting point for our sampling. The `find_MAP` function finds the maximum a posteriori point (MAP), which is often a good choice for starting point. `find_MAP` uses an optimization algorithm to find the local maximum of the log posterior.

# <codecell>

start = find_MAP(model)

# <markdowncell>

# Points in parameter space are represented by dictionaries with parameter names as they keys and the value of the parameters as the values.

# <codecell>

start

# <markdowncell>

# We will use Hamiltonian Monte Carlo (HMC) to sample from the posterior as implemented by the `HamiltonianMC` step method class. HMC requires an (inverse) covariance matrix to scale its proposal points. So first we pick one. It is helpful if it approximates the true (inverse) covariance matrix. For distributions which are somewhat normal-like, the hessian matrix (matrix of 2nd derivatives of the log posterior) close to the MAP will approximate the inverse covariance matrix of the posterior.
# 
# The `approx_hess(model, point)` function works similarly to the find_MAP function and returns the hessian at a given point.

# <codecell>

h = approx_hess(model, start)

# <markdowncell>

# Now we build our step method. `HamiltonianMC` takes a model object, a set of variables it should update and a scaling matrix.

# <codecell>

step = HamiltonianMC(model, model.vars, h)

# <markdowncell>

# The `sample` function takes a number of steps to sample, a step method, a starting point. It returns a trace, the final state of 
# the step method and the seconds that sampling took.

# <codecell>

trace, state, t = sample(3e3, step, start)

# <markdowncell>

# To use more than one sampler, use a CompoundStep which takes a list of step methods. 
# 
# The trace object can be indexed by the variables returning an array with the first index being the sample index
# and the other indexes the shape of the parameter. Thus the shape of trace[x].shape == (ndraw, 2,1).
# 
# Pymc3 provides `traceplot` a simple plot for a trace.

# <codecell>

traceplot(trace)

