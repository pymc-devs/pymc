# pymc3 and theano

## What is theano

Theano is a package that allows us to define functions involving array
operations and linear algebra. When we define a pymc3 model, we implicitly
build up a theano function from the space of our parameters to
their posterior probability density up to a constant factor. We then use
symbolic manipulations of this function to also get access to its gradient.

For a thorough introduction into theano see the thenao docs (link),
but for the most part you don't need many details about it as long
as you are not trying to define new distributions or other extensions
of pymc3. But let's look at a simple example to give you a rough
idea about how it works. Say, we'd like to define the (completely
arbitrarily chosen) function
$$
  f\colon \mathbb{R} \times \mathbb{R}^n \mathbb{N}^n \to \mathbb{R}\\
  (a, x, y) \mapsto \sum_{i=0}{n} \exp(ax_i^3 + y_i^2).
$$

First, we need to define symbolic variables for our inputs (this
is similar to eg sympy's `Symbol`)::

    import theano
    import theano.tensor as tt
    # We don't specify the dtype of our input variables, so it
    # defaults to using float64 without any special config.
    a = tt.scalar('a')
    x = tt.vector('x')
    y = tt.vector('y')

Next, we use those variables to build up a symbolic representation
of the output of our function. Note, that no computation is actually
being done at this point. We only record what operations we need to
do to compute the output::

    inner = a * x**3 + y**2
    out = tt.exp(inner).sum()

.. note::

   In this example we use `tt.exp` to create a symbolic representation
   of the exponential of `inner`. Somewhat surprisingly, it
   would also have worked if we used `np.exp`. This is because numpy
   gives objects it operates on a chance to define the results of
   operations themselves. Theano variables do this for a large number
   of operations. We usually still prefer it if people use the theano
   function instead of the numpy versions, as that makes it clear that
   we are working with symbolic input instead of plain arrays.

Now we can tell theano to build a function that does this computation.
With a typical configuration theano generates C code, compiles it,
and creates a python function which wrapps the C function::

    func = theano.function([a, x, y], [out])

We can call this function with actual arrays as many times as we want::

    a_val = 1.2
    x_vals = np.random.randn(10)
    y_vals = np.random.randn(10)

    out = func(a_val, x_vals, y_vals)

For the most part the symbolic theano variables can be operated on
like numpy arrays. Most numpy functions are available in `theano.tensor`
(which is typically imported as `tt`). A lot of linear algebra operations
can be found in `tt.nlinalg` and `tt.slinalg` (the numpy and scipy
operations respectively). Some support for sparse matrices is available
in `theano.sparse`.

A notable exception where theano variables *don't* behave like
numpy arrays are operations involving conditional execution:

Code like this won't work as expected if used on theano variables::

    a = tt.vector('a')
    if (a > 0).all():
        b = tt.sqrt(a)
    else:
        b = -a

`(a > 0).all()` isn't actually a boolean as it would be in numpy, but
still a symbolic variable. Python will convert this object to a boolean
and according to the rules for this conversions, things that aren't empty
containers or zero are converted to True. So the code is equivalent
to this::

    a = tt.vector('a')
    b = tt.sqrt(a)

To get the desired behaviour, we can use `tt.switch`::

    a = tt.vector('a')
    b = tt.switch((a > 0).all(), tt.sqrt(a), -a)

Indexing also works similar to numpy::

    a = tt.vector('a')
    # Access the 10th element. This will fail when a function build
    # from this expression is executed with an array that is too short.
    b = a[10]

    # Extract a subvector
    b = a[[1, 2, 10]]

Changing element of an array is possible using `tt.set_subtensor`::

    a = tt.vector('a')
    b = tt.set_subtensor(a[:10], 1)

    # is roughly equivalent this (although theano is usually able
    # to avoid the copy)
    a = np.random.randn(10)
    b = a.copy()
    b[:10] = 1

## How pymc3 uses theano

Now that we have a basic understanding of theano we can look at what
happens if we define a pymc3 model. Let's look at a simple example::

    true_mu = 0.1
    data = true_mu + np.random.randn(50)

    with pm.Model() as model:
        mu = pm.Normal('mu', mu=0, sd=1)
        y = pm.Normal('y', mu=mu, sd=1, observed=data)

In this model we define two variables: `mu` and `y`. The first is
a free variable that we want to infer, the second is an observed
variable. To sample from the posterior we need to build the function
$$
    \log P(μ|y) + C = \log P(y|μ) + \log P(μ) =: f(μ)\\
$$
where with the normal likelihood $N(x|μ,σ^2)$
$$
    f\colon \mathbb{R} \to \mathbb{R}\\
    f(μ) = \log N(μ|0, 1) + \log N(y|0, 1),
$$

To build that function we need to keep track of two things: The parameter
space (the *free variables*) and the logp function. For each free variable
we generate a theano variable. And for each variable (observed or otherwise)
we add a term to the global logp. In the background something similar to
this is happening::

    # For illustration only, those functions don't actually exist
    # in exactly this way!
    model = pm.Model()

    mu = tt.scalar('mu')
    model.add_free_variable(mu)
    model.add_logp_term(normal_logp(μ| 0, 1))

    model.add_logp_term(normal_logp(data| mu, 1))
