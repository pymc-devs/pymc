:orphan:

..
    _href from docs/source/index.rst

=================
PyMC and PyTensor
=================

What is PyTensor
================

PyTensor is a package that allows us to define functions involving array
operations and linear algebra. When we define a PyMC model, we implicitly
build up an PyTensor function from the space of our parameters to
their posterior probability density up to a constant factor. We then use
symbolic manipulations of this function to also get access to its gradient.

For a thorough introduction to PyTensor see the
:doc:`pytensor docs <pytensor:index>`,
but for the most part you don't need detailed knowledge about it as long
as you are not trying to define new distributions or other extensions
of PyMC. But let's look at a simple example to get a rough
idea about how it works. Say, we'd like to define the (completely
arbitrarily chosen) function

.. math::

  f\colon \mathbb{R} \times \mathbb{R}^n \times \mathbb{N}^n \to \mathbb{R}\\
  (a, x, y) \mapsto \sum_{i=0}^{n} \exp(ax_i^3 + y_i^2).


First, we need to define symbolic variables for our inputs (this
is similar to eg SymPy's `Symbol`)::

    import pytensor
    import pytensor.tensor as pt
    # We don't specify the dtype of our input variables, so it
    # defaults to using float64 without any special config.
    a = pt.scalar('a')
    x = pt.vector('x')
    # `pt.ivector` creates a symbolic vector of integers.
    y = pt.ivector('y')

Next, we use those variables to build up a symbolic representation
of the output of our function. Note that no computation is actually
being done at this point. We only record what operations we need to
do to compute the output::

    inner = a * x**3 + y**2
    out = pt.exp(inner).sum()

.. note::

   In this example we use `pt.exp` to create a symbolic representation
   of the exponential of `inner`. Somewhat surprisingly, it
   would also have worked if we used `np.exp`. This is because numpy
   gives objects it operates on a chance to define the results of
   operations themselves. PyTensor variables do this for a large number
   of operations. We usually still prefer the PyTensor
   functions instead of the numpy versions, as that makes it clear that
   we are working with symbolic input instead of plain arrays.

Now we can tell PyTensor to build a function that does this computation.
With a typical configuration, PyTensor generates C code, compiles it,
and creates a python function which wraps the C function::

    func = pytensor.function([a, x, y], [out])

We can call this function with actual arrays as many times as we want::

    a_val = 1.2
    x_vals = np.random.randn(10)
    y_vals = np.random.randn(10)

    out = func(a_val, x_vals, y_vals)

For the most part the symbolic PyTensor variables can be operated on
like NumPy arrays. Most NumPy functions are available in `pytensor.tensor`
(which is typically imported as `pt`). A lot of linear algebra operations
can be found in `pt.nlinalg` and `pt.slinalg` (the NumPy and SciPy
operations respectively). Some support for sparse matrices is available
in `pytensor.sparse`. For a detailed overview of available operations,
see :mod:`the pytensor api docs <pytensor.tensor>`.

A notable exception where PyTensor variables do *not* behave like
NumPy arrays are operations involving conditional execution.

Code like this won't work as expected::

    a = pt.vector('a')
    if (a > 0).all():
        b = pt.sqrt(a)
    else:
        b = -a

`(a > 0).all()` isn't actually a boolean as it would be in NumPy, but
still a symbolic variable. Python will convert this object to a boolean
and according to the rules for this conversion, things that aren't empty
containers or zero are converted to `True`. So the code is equivalent
to this::

    a = pt.vector('a')
    b = pt.sqrt(a)

To get the desired behaviour, we can use `pt.switch`::

    a = pt.vector('a')
    b = pt.switch((a > 0).all(), pt.sqrt(a), -a)

Indexing also works similarly to NumPy::

    a = pt.vector('a')
    # Access the 10th element. This will fail when a function build
    # from this expression is executed with an array that is too short.
    b = a[10]

    # Extract a subvector
    b = a[[1, 2, 10]]

Changing elements of an array is possible using `pt.set_subtensor`::

    a = pt.vector('a')
    b = pt.set_subtensor(a[:10], 1)

    # is roughly equivalent to this (although pytensor avoids
    # the copy if `a` isn't used anymore)
    a = np.random.randn(10)
    b = a.copy()
    b[:10] = 1

How PyMC uses PyTensor
====================

Now that we have a basic understanding of PyTensor we can look at what
happens if we define a PyMC model. Let's look at a simple example::

    true_mu = 0.1
    data = true_mu + np.random.randn(50)

    with pm.Model() as model:
        mu = pm.Normal('mu', mu=0, sigma=1)
        y = pm.Normal('y', mu=mu, sigma=1, observed=data)

In this model we define two variables: `mu` and `y`. The first is
a free variable that we want to infer, the second is an observed
variable. To sample from the posterior we need to build the function

.. math::

   \log P(μ|y) + C = \log P(y|μ) + \log P(μ) =: \text{logp}(μ)\\

where with the normal likelihood :math:`N(x|μ,σ^2)`

.. math::

    \text{logp}\colon \mathbb{R} \to \mathbb{R}\\
    μ \mapsto \log N(μ|0, 1) + \log N(y|μ, 1),

To build that function we need to keep track of two things: The parameter
space (the *free variables*) and the logp function. For each free variable
we generate an PyTensor variable. And for each variable (observed or otherwise)
we add a term to the global logp. In the background something similar to
this is happening::

    # For illustration only, those functions don't actually exist
    # in exactly this way!
    model = pm.Model()

    mu = pt.scalar('mu')
    model.add_free_variable(mu)
    model.add_logp_term(pm.Normal.dist(0, 1).logp(mu))

    model.add_logp_term(pm.Normal.dist(mu, 1).logp(data))

So calling `pm.Normal()` modifies the model: It changes the logp function
of the model. If the `observed` keyword isn't set it also creates a new
free variable. In contrast, `pm.Normal.dist()` doesn't care about the model,
it just creates an object that represents the normal distribution. Calling
`logp` on this object creates an PyTensor variable for the logp probability
or log probability density of the distribution, but again without changing
the model in any way.

Continuous variables with support only on a subset of the real numbers
are treated a bit differently. We create a transformed variable
that has support on the reals and then modify this variable. For
example::

    with pm.Model() as model:
        mu = pm.Normal('mu', 0, 1)
        sigma = pm.HalfNormal('sigma', 1)
        y = pm.Normal('y', mu=mu, sigma=sigma, observed=data)

is roughly equivalent to this::

    # For illustration only, not real code!
    model = pm.Model()
    mu = pt.scalar('mu')
    model.add_free_variable(mu)
    model.add_logp_term(pm.Normal.dist(0, 1).logp(mu))

    sd_log__ = pt.scalar('sd_log__')
    model.add_free_variable(sd_log__)
    model.add_logp_term(corrected_logp_half_normal(sd_log__))

    sigma = pt.exp(sd_log__)
    model.add_deterministic_variable(sigma)

    model.add_logp_term(pm.Normal.dist(mu, sigma).logp(data))

The return values of the variable constructors are subclasses
of PyTensor variables, so when we define a variable we can use any
PyTensor operation on them::

    design_matrix = np.array([[...]])
    with pm.Model() as model:
        # beta is a pt.dvector
        beta = pm.Normal('beta', 0, 1, shape=len(design_matrix))
        predict = pt.dot(design_matrix, beta)
        sigma = pm.HalfCauchy('sigma', beta=2.5)
        pm.Normal('y', mu=predict, sigma=sigma, observed=data)
