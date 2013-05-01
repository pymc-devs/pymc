# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from matplotlib.pylab import *
from pymc import *
import numpy as np 
from numpy.random import normal, beta

# <markdowncell>

# Model
# -----
# We consider the following generative model

# <codecell>

xtrue = normal(scale = 2., size = 1)
ytrue = normal(loc = np.exp(xtrue), scale = 1, size = (2,1))
zdata = normal(loc = xtrue + ytrue, scale = .75, size = (2, 20))

# <markdowncell>

# `zdata` is observed but `xtrue` and `ytrue` are not. Thus x and y are unknown, and we want to come up with posterior distributions for them. 

# <markdowncell>

# Build Model
# ----------- 
# We create a new `Model` objects, and do operations within its context. The `with` lets PyMC know this model is the current model of interest. 
# 
# We construct new random variables with the constructor for its prior distribution such as `Normal` while within a model context (inside the `with`). When you make a random variable it is automatically added to the model. The constructor returns a Theano variable.
# 
# Using the constructor may specify the name of the random variable, the parameters of a random variable's prior distribution, as well as the shape of the random variable. We can specify that a random variable is observed by specifying the data that was observed.

# <codecell>

with Model() as model:
    x = Normal('x', mu = 0., tau = 1)
    y = Normal('y', mu = exp(x), tau = 2.**-2, shape = (2,1))
    
    z = Normal('z', mu = x + y, tau = .75**-2, observed = zdata)

# <markdowncell>

# Fit Model
# ---------
# We need a starting point for our sampling. The `find_MAP` function finds the maximum a posteriori point (MAP), which is often a good choice for starting point. `find_MAP` uses an optimization algorithm to find the local maximum of the log posterior. 

# <codecell>

with model:
    start = find_MAP()

# <markdowncell>

# Points in parameter space are represented by dictionaries with parameter names as they keys and the value of the parameters as the values.

# <codecell>

start

# <markdowncell>

# We will use Hamiltonian Monte Carlo (HMC) to sample from the posterior as implemented by the `HamiltonianMC` step method class. 
# 
# HMC requires an (inverse) covariance matrix to scale its proposal points. So first we pick one. It is helpful if it approximates the true (inverse) covariance matrix of the posterior. For distributions which are somewhat normal-like, the hessian matrix (matrix of 2nd derivatives of the log posterior) close to the MAP will approximate the inverse covariance matrix of the posterior.
# 
# The `approx_hess(start)` function works similarly to the find_MAP function and returns the hessian at a given point.

# <codecell>

with model:
    h = approx_hess(start)

# <markdowncell>

# Now we build our step method. The `HamiltonianMC` constructor takes a set of variables it should update and a scaling matrix.

# <codecell>

with model:
    step = HamiltonianMC(model.vars, h)

# <markdowncell>

# The `sample` function takes a number of steps to sample, a step method, a starting point. It returns a trace object which contains our samples.

# <codecell>

with model: 
    trace = sample(3000, step, start)

# <markdowncell>

# To use more than one sampler, pass a list of step methods to `sample`. 
# 
# The trace object can be indexed by the variables in the model, returning an array with the first index being the sample index
# and the other indexes the shape of the parameter. Thus for this example:

# <codecell>

trace[y].shape == (3000, 2,1)

# <markdowncell>

# `traceplot` is a summary plotting function for a trace.

# <codecell>

traceplot(trace)

# <markdowncell>

# ## PyMC Internals
# 
# ### Model 
# 
# The `Model` class has very simple internals: just a list of unobserved variables (`Model.vars`) and a list of factors which go into computing the posterior density (`Model.factors`) (see model.py for more).
# 
# A Python "`with model:`" block has `model` as the current model. Many functions, like `find_MAP` and `sample`, must be in such a block to work correctly by default. They look in the current context for a model to use. You may also explicitly specify the model for them to use. This allows us to treat the current model as an implicit parameter to these functions. 
# 
# ### Distribution Classes
# 
# `Normal` and other distributions are actually `Distribution` subclasses. The constructors have different behavior depending on whether they are called with a name argument or not (string argument in 1st slot). This allows PyMC to have intuitive model specification syntax and still distinguish between random variables and distributions.
# 
# When a `Distribution` constructor is called:
# 
# * Without a name argument, it simply constructs a distribution object and returns it. It won't construct a random variable. This object has properties like `logp` (density function) and `expectation`.
# * With a name argument, it constructs a random variable using the distrubtion object as the prior distribution and inserts this random variable into the current model. Then the constructor returns the random variable. 

