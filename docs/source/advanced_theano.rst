=================================
Advanced usage of Theano in PyMC3
=================================

Using shared variables
======================

Shared variables allow us to use values in theano functions that are
not considered an input to the function, but can still be changed
later. They are very similar to global variables in may ways::

    a = tt.scalar('a')
    # Create a new shared variable with initial value of 0.1
    b = theano.shared(0.1)
    func = theano.function([a], a * b)
    assert func(2.) == 0.2

    b.set_value(10.)
    assert func(2.) == 20.

Shared variables can also contain arrays, and are allowed to change
their shape as long as the number of dimensions stays the same.

We can use shared variables in PyMC3 to fit the same model to several
datasets without the need to recreate the model each time (which can
be time consuming if the number of datasets is large)::

    # We generate 10 datasets
    true_mu = [np.random.randn() for _ in range(10)]
    observed_data = [mu + np.random.randn(20) for mu in true_mu]

    data = theano.shared(observed_data[0])
    pm.Model() as model:
        mu = pm.Normal('mu', 0, 10)
        pm.Normal('y', mu=mu, sd=1, observed=data)

    # Generate one trace for each dataset
    traces = []
    for data_vals in observed_data:
        # Switch out the observed dataset
        data.set_value(data_vals)
        with model:
            traces.append(pm.sample())

We can also sometimes use shared variables to work around limitations
in the current PyMC3 api. A common task in Machine Learning is to predict
values for unseen data, and one way to achieve this is to use a shared
variable for our observations::

    x = np.random.randn(100)
    y = x > 0

    x_shared = theano.shared(x)

    with pm.Model() as model:
      coeff = pm.Normal('x', mu=0, sd=1)
      logistic = pm.math.sigmoid(coeff * x_shared)
      pm.Bernoulli('obs', p=logistic, observed=y)

      # fit the model
      trace = pm.sample()

      # Switch out the observations and use `sample_posterior_predictive` to predict
      x_shared.set_value([-1, 0, 1.])
      post_pred = pm.sample_posterior_predictive(trace, samples=500)

However, due to the way we handle shapes at the moment, it is
not possible to change the shape of a shared variable if that would
also change the shape of one of the variables.


Writing custom Theano Ops
=========================

While Theano includes a wide range of operations, there are cases where
it makes sense to write your own. But before doing this it is a good
idea to think hard if it is actually necessary. Especially if you want
to use algorithms that need gradient information — this includes NUTS and
all variational methods, and you probably *should* want to use those —
this is often quite a bit of work and also requires some math and
debugging skills for the gradients.

Good reasons for defining a custom Op might be the following:

- You require an operation that is not available in Theano and can't
  be build up out of existing Theano operations. This could for example
  include models where you need to solve differential equations or
  integrals, or find a root or minimum of a function that depends
  on your parameters.
- You want to connect your PyMC3 model to some existing external code.
- After carefully considering different parametrizations and a lot
  of profiling your model is still too slow, but you know of a faster
  way to compute the gradient than what theano is doing. This faster
  way might be anything from clever maths to using more hardware.
  There is nothing stopping anyone from using a cluster via MPI in
  a custom node, if a part of the gradient computation is slow enough
  and sufficiently parallelizable to make the cost worth it.
  We would definitely like to hear about any such examples.

Theano has extensive `documentation, <http://deeplearning.net/software/theano/extending/index.html>`_
about how to write new Ops.


Finding the root of a function
------------------------------

We'll use finding the root of a function as a simple example.
Let's say we want to define a model where a parameter is defined
implicitly as the root of a function, that depends on another
parameter:

.. math::

   \theta \sim N^+(0, 1)\\
   \text{$\mu\in \mathbb{R}^+$ such that $R(\mu, \theta)
         = \mu + \mu e^{\theta \mu} - 1= 0$}\\
   y \sim N(\mu, 0.1^2)

First, we observe that this problem is well-defined, because
:math:`R(\cdot, \theta)` is monotone and has the image :math:`(-1, \infty)`
for :math:`\mu, \theta \in \mathbb{R}^+`. To avoid overflows in
:math:`\exp(\mu \theta)` for large
values of :math:`\mu\theta` we instead find the root of

.. math::

    R'(\mu, \theta)
        = \log(R(\mu, \theta) + 1)
        = \log(\mu) + \log(1 + e^{\theta\mu}).

Also, we have

.. math::

    \frac{\partial}{\partial\mu}R'(\mu, \theta)
        = \theta\, \text{logit}^{-1}(\theta\mu) + \mu^{-1}.

We can now use `scipy.optimize.newton` to find the root::

    from scipy import optimize, special
    import numpy as np

    def func(mu, theta):
        thetamu = theta * mu
        value = np.log(mu) + np.logaddexp(0, thetamu)
        return value

    def jac(mu, theta):
        thetamu = theta * mu
        jac = theta * special.expit(thetamu) + 1 / mu
        return jac

    def mu_from_theta(theta):
        return optimize.newton(func, 1, fprime=jac, args=(0.4,))

We could wrap `mu_from_theta` with `tt.as_op` and use gradient-free
methods like Metropolis, but to get NUTS and ADVI working, we also
need to define the derivative of `mu_from_theta`. We can find this
derivative using the implicit function theorem, or equivalently we
take the derivative with respect of :math:`\theta` for both sides of
:math:`R(\mu(\theta), \theta) = 0` and solve for :math:`\frac{d\mu}{d\theta}`.
This isn't hard to do by hand, but for the fun of it, let's do it using
sympy::

    import sympy

    mu = sympy.Function('mu')
    theta = sympy.Symbol('theta')
    R = mu(theta) + mu(theta) * sympy.exp(theta * mu(theta)) - 1
    solution = sympy.solve(R.diff(theta), mu(theta).diff(theta))[0]

We get

.. math::

    \frac{d}{d\theta}\mu(\theta)
        = - \frac{\mu(\theta)^2}{1 + \theta\mu(\theta) + e^{-\theta\mu(\theta)}}

Now, we use this to define a theano op, that also computes the gradient::

    import theano
    import theano.tensor as tt
    import theano.tests.unittest_tools

    class MuFromTheta(tt.Op):
        itypes = [tt.dscalar]
        otypes = [tt.dscalar]

        def perform(self, node, inputs, outputs):
            theta, = inputs
            mu = mu_from_theta(theta)
            outputs[0][0] = np.array(mu)

        def grad(self, inputs, g):
            theta, = inputs
            mu = self(theta)
            thetamu = theta * mu
            return [- g[0] * mu ** 2 / (1 + thetamu + tt.exp(-thetamu))]

If you value your sanity, always check that the gradient is ok::

    theano.tests.unittest_tools.verify_grad(MuFromTheta(), [np.array(0.2)])
    theano.tests.unittest_tools.verify_grad(MuFromTheta(), [np.array(1e-5)])
    theano.tests.unittest_tools.verify_grad(MuFromTheta(), [np.array(1e5)])

We can now define our model using this new op::

    import pymc3 as pm

    tt_mu_from_theta = MuFromTheta()

    with pm.Model() as model:
        theta = pm.HalfNormal('theta', sd=1)
        mu = pm.Deterministic('mu', tt_mu_from_theta(theta))
        pm.Normal('y', mu=mu, sd=0.1, observed=[0.2, 0.21, 0.3])

        trace = pm.sample()
