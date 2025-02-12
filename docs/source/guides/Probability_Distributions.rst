:orphan:

..
    _href from docs/source/index.rst

.. _prob_dists:

*********************************
Probability Distributions in PyMC
*********************************

The most fundamental step in building Bayesian models is the specification of a full probability model for the problem at hand. This primarily involves assigning parametric statistical distributions to unknown quantities in the model, in addition to appropriate functional forms for likelihoods to represent the information from the data. To this end, PyMC includes a comprehensive set of pre-defined statistical distributions that can be used as model building blocks.

For example, if we wish to define a particular variable as having a normal prior, we can specify that using an instance of the ``Normal`` class.

::

    with pm.Model():

        x = pm.Normal('x', mu=0, sigma=1)

A variable requires at least a ``name`` argument, and zero or more model parameters, depending on the distribution. Parameter names vary by distribution, using conventional names wherever possible. The example above defines a scalar variable. To make a vector-valued variable, a ``shape`` argument should be provided; for example, a 3x3 matrix of beta random variables could be defined with:

::

    with pm.Model():

        p = pm.Beta('p', 1, 1, shape=(3, 3))

Probability distributions are all subclasses of ``Distribution``, which in turn has two major subclasses: ``Discrete`` and ``Continuous``. In terms of data types, a ``Continuous`` random variable is given whichever floating point type is defined by ``pytensor.config.floatX``, while ``Discrete`` variables are given ``int16`` types when ``pytensor.config.floatX`` is ``float32``, and ``int64`` otherwise.

All distributions in ``pm.distributions`` are associated with two key functions:

1. ``logp(dist, value)`` - Calculates log-probability at given value
2. ``draw(dist, size=...)`` - Generates random samples

For example, with a normal distribution:

::

    with pm.Model():
        x = pm.Normal('x', mu=0, sigma=1)

    # Calculate log-probability
    log_prob = pm.logp(x, 0.5)

    # Generate samples
    samples = pm.draw(x, size=100)

Custom distributions using ``CustomDist`` should provide logp via the ``dist`` parameter:

::

    def custom_logp(value, mu):
        return -0.5 * (value - mu)**2

    custom_dist = pm.CustomDist('custom', dist=custom_logp, mu=0)


Custom distributions
====================

Despite the fact that PyMC ships with a large set of the most common probability distributions, some problems may require the use of functional forms that are less common, and not available in ``pm.distributions``. One example of this is in survival analysis, where time-to-event data is modeled using probability densities that are designed to accommodate censored data.

An exponential survival function, where :math:`c=0` denotes failure (or non-survival), is defined by:

.. math::

    f(c, t) = \left\{ \begin{array}{l} \exp(-\lambda t), \text{if c=1} \\
               \lambda \exp(-\lambda t), \text{if c=0}  \end{array} \right.

Such a function can be implemented as a PyMC distribution by writing a function that specifies the log-probability, then passing that function as a keyword argument to the ``CustomDist`` function, which creates an instance of a PyMC distribution with the custom function as its log-probability.

For the exponential survival function, this is:

::

    def logp(value, t, lam):
        return (value * log(lam) - lam * t).sum()

    exp_surv = pm.CustomDist('exp_surv', dist=logp, t=t, lam=lam, observed=failure)

Similarly, if a random number generator is required, a function returning random numbers corresponding to the probability distribution can be passed as the ``random`` argument.


Using PyMC distributions without a Model
========================================

Distribution objects, as we have defined them so far, are only usable inside of a ``Model`` context. If they are created outside of the model context manager, it raises an error.

::

    y = Binomial('y', n=10, p=0.5)


::

    TypeError: No context on context stack

This is because the distribution classes are designed to integrate themselves automatically inside of a PyMC model. When a model cannot be found, it fails. However, each ``Distribution`` has a ``dist`` class method that returns a stripped-down distribution object that can be used outside of a PyMC model.

For example, a standalone binomial distribution can be created by:

::

    y = pm.Binomial.dist(n=10, p=0.5)

This allows for probabilities to be calculated and random numbers to be drawn.

::

    >>> pm.logp(y, 4).eval()
    array(-1.5843639373779297, dtype=float32)

    >>> pm.draw(y, size=3)
    array([5, 4, 3])


Auto-transformation
===================

To aid efficient MCMC sampling, any continuous variables that are constrained to a sub-interval of the real line are automatically transformed so that their support is unconstrained. This frees sampling algorithms from having to deal with boundary constraints.

For example, the gamma distribution is positive-valued. If we define one for a model:

::

    with pm.Model() as model:
        g = pm.Gamma('g', 1, 1)

We notice a modified variable inside the model ``value_vars`` attribute.  These variables represent the values of each random variable in the model's log-likelihood.

::

    >>> model.value_vars
    [g_log__]

As the name suggests, the variable ``g`` has been log-transformed, and this is the space over which posterior sampling takes place.

The value of the transformed variable is simply back-transformed when a sample is drawn in order to recover the original variable.

By default, auto-transformed variables are ignored when summarizing and plotting model output.
