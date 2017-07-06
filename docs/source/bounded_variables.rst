=================
Bounded Variables
=================

Some variables have constraints.  For instance, it makes no sense for a
standard deviation to have a negative value, so something like a Normal prior
on a parameter that represents a standard deviation would be inappropriate.
PyMC3 includes distributions that have positive valued support, such as
``Gamma`` or ``Exponential``.  PyMC3 also includes several bounded
distributions, such as ``Uniform``, ``HalfNormal``, and ``HalfCauchy``, that
are restricted to a specific domain.  Variables that have been assigned these
distributions are automatically transformed to remove the discontinuity of the
probability distribution introduced at zero. 

PyMC3 includes the construct ``Bound`` for placing constraints on existing
distributions.  For instance, one may have prior information that suggests that
parameter representing a standard deviation is near one.  One could use a
Normal distribution while constraining the support to positive values.  A
``BoundedNormal`` distribution can be defined before or within the model block.
If the bound is to be applied to a single variable in the model, it may be
cleaner notationally to define both the bound and variable together within the
model block. ::

    import pymc3 as pm

    BoundedNormal = pm.Bound(pm.Normal, lower=0.0)

    with pm.Model() as model:
        x = BoundedNormal('x', mu=1.0, sd=3.0)
        
        # implicit definition
        x = pm.Bound(pm.Normal, lower=0.0)('x', mu=1.0, sd=3.0)

The syntax is similar for vectors of random variables.  With the same
``BoundedNormal`` object we created previously we can write::

    with model:
        x_vector = BoundedNormal('x_vector', mu=1.0, sd=3.0, shape=3)

All univariate distributions in PyMC3 can be given bounds.  For example, here
is a Poisson distributed random variable with its support restricted to be
larger than two and smaller than seven::

    BoundedPoisson = pm.Bound(pm.Poisson, lower=2, upper=7)
    with model:
        p = BoundedPoisson('p', mu=5)


It's recommended to use ``Bound`` whenever there is a restriction on the
support of a random variable, even if the likely values of that variable are
far from the boundary.  Bounds cannot be given to variables that are
``observed``.  If you want to model truncated data, use a ``Potential`` in
combination with a cumulative probability function.  See `this example
<https://github.com/pymc-devs/pymc3/blob/master/pymc3/examples/censored_data.py>`_.
