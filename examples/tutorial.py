from pymc import *
import numpy as np 
import theano 

data = np.random.normal(size = (2, 20))
"""
the first thing you interact with is the model class
The model class is a dead simple class (see core.py), it contains a list of random variables (model.vars)
for parameters and a list of factors that go into the posterior (model.factors), and no methods.
"""


model = Model()
Var = model.Var
Data = model.Data 

"""
So that Theano throws errors at appropriate times, we should pass a test point to the model.

To add a parameter to the model we use a function, in this case Var.
We must pass Var the model we are adding a parameter to, the name of the parameter, 
the prior distribution for the parameter and optionally the shape and dtype of the parameter.
The Var returns a Theano variable which represents the parameter we have added to the model. 

The Var function is also very simple (see core.py), it creates a Theano variable, adds it to the model
list and adds the likelihood factor to the model list.

The distribution functions (see distributions.py), such as the one we see here, Normal(mu, tau), take some parameters and return
a function that takes a value for the parameter (here, x) and returns the log likelihood. The parameters to
distributions may be either model parameters (see Data section) or constants.
"""
x = Var('x', Normal(mu = .5, tau = 2.**-2), (2,1))
z = Var('z', Beta(alpha = 10, beta =5.5))


"""
In order to add observed data to the model, we use the Data function. It works much the same
as the Var function. It takes the data, and the distribution for that data. 
"""
Data(data, Normal(mu = x, tau = .75**-2))


"""
We need to specify a point in parameter space as our starting point for fitting.
To specify a point in the parameter space, we use a 
dictionary of parameter names-> parameter values. You can also get an automatically generated one from model.test_point
"""
start = {'x' : np.array([[0.2],[.3]]),
         'z' : np.array([.5])}

"""
The maximum a posteriori point (MAP) may be a better starting point than the one we have chosen. 
The find_MAP function uses an optimization algorithm on the log posterior to find and return the local maximum.
"""


start = find_MAP(model, start)

"""
The step method we will use (Hamiltonian Monte Carlo) requires a covariance matrix. It is helpful if it 
approximates the true covariance matrix. For distributions which are somewhat normal-like the inverse hessian 
at the MAP (or points close to it) will approximate this covariance matrix. 

the approx_cov() function works similarly to the find_MAP function and returns the inverse hessian at a given point.

"""
hmc_hess = approx_hess(model, start)


"""
To sample from the posterior, we must choose a sampling method. 
"""
step_method = hmc_step(model, model.vars, hmc_hess, is_cov = False)

"""
To use the step functions, we use the sample function. It takes a number of steps to do, 
a step method, a starting point and a storage object for the resulting samples. It returns the final state of 
the step method and the total time in seconds that the sampling took. The samples are now in history.
"""
history, state, t = sample(3e3, step_method, start)
print "took :", t, " seconds"

"""
To use more than one sampler, look at the compound_step step method which takes a list of step methods. 

The history object can be indexed by names of variables returning an array with the first index being the sample
and the other indexes the shape of the parameter. Thus the shape of history['x'].shape == (ndraw, 2,1)
""" 
print np.mean(history['x'], axis = 0)

try :
    from pylab import * 
    figure()
    subplot(2,2,1)
    plot(history['x'][:,0,0])
    subplot(2,2,2)
    hist(history['x'][:,0,0], 50)
    
    subplot(2,2,3)
    plot(history['z'])
    subplot(2,2,4)
    hist(history['z'], 50)
    show()
except ImportError : 
    pass 
