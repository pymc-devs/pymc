=================
Bounded Variables
=================

PyMC3 includes the construct :class:`~pymc3.distributions.bound.Bound` for
placing constraints on existing probability distributions.  It modifies a given
distribution to take values only within a specified interval.

Some types of variables require constraints.  For instance, it doesn't make
sense for a standard deviation to have a negative value, so something like a
Normal prior on a parameter that represents a standard deviation would be
inappropriate.  PyMC3 includes distributions that have positive support, such
as :class:`~pymc3.distributions.continuous.Gamma` or
:class:`~pymc3.distributions.continuous.Exponential`.  PyMC3 also includes
several bounded distributions, such as
:class:`~pymc3.distributions.continuous.Uniform`,
:class:`~pymc3.distributions.continuous.HalfNormal`, and
:class:`~pymc3.distributions.continuous.HalfCauchy`, that are restricted to a
specific domain.

All univariate distributions in PyMC3 can be given bounds.  The distribution of
a continuous variable that has been bounded is automatically transformed into
an unnormalized distribution whose domain is unconstrained.  The transformation
improves the efficiency of sampling and variational inference algorithms.

Usage
#####

For example, one may have prior information that suggests that the value of a
parameter representing a standard deviation is near one.  One could use a
Normal distribution while constraining the support to be positive.  The
specification of a bounded distribution should go within the model block::

    import pymc3 as pm

    with pm.Model() as model:
        BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
        x = BoundedNormal('x', mu=1.0, sd=3.0)

If the bound will be applied to a single variable in the model, it may be
cleaner notationally to define both the bound and variable together. ::

    with model:
        x = pm.Bound(pm.Normal, lower=0.0)('x', mu=1.0, sd=3.0)

Bounds can also be applied to a vector of random variables.  With the same
``BoundedNormal`` object we created previously we can write::

    with model:
        x_vector = BoundedNormal('x_vector', mu=1.0, sd=3.0, shape=3)

Caveats
#######

* Bounds cannot be given to variables that are ``observed``.  To model
  truncated data, use a :func:`~pymc3.model.Potential` in combination with a cumulative
  probability function.  See `this example <https://github.com/pymc-devs/pymc3/blob/master/pymc3/examples/censored_data.py>`_.

* The automatic transformation applied to continuous distributions results in
  an unnormalized probability distribution.  This doesn't effect inference
  algorithms but may complicate some model comparison procedures.

API
###


.. currentmodule:: pymc3.distributions.bound
.. automodule:: pymc3.distributions.bound
   :members:
