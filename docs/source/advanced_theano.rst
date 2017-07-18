=================================
Advanced usage of Theano in PyMC3
=================================

Using shared variables
======================

Shared variables allow us to use values in theano functions that are
not considered an input to the function, but can still be changed
later. They are very similar to global variables in may ways.::

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
be time consuming if the number of datasets is large).::

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

      # Switch out the observations and use `sample_ppc` to predict
      x_shared.set_value([-1, 0, 1.])
      post_pred = pm.sample_ppc(trace, samples=500)

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
