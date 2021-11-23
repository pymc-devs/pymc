=================
Bounded Variables
=================

PyMC includes the construct :class:`~pymc.distributions.bound.Bound` for
placing constraints on existing probability distributions.  It modifies a given
distribution to take values only within a specified interval.

Some types of variables require constraints.  For instance, it doesn't make
sense for a standard deviation to have a negative value, so something like a
Normal prior on a parameter that represents a standard deviation would be
inappropriate.  PyMC includes distributions that have positive support, such
as :class:`~pymc.distributions.continuous.Gamma` or
:class:`~pymc.distributions.continuous.Exponential`.  PyMC also includes
several bounded distributions, such as
:class:`~pymc.distributions.continuous.Uniform`,
:class:`~pymc.distributions.continuous.HalfNormal`, and
:class:`~pymc.distributions.continuous.HalfCauchy`, that are restricted to a
specific domain.

All univariate distributions in PyMC can be given bounds.  The distribution of
a continuous variable that has been bounded is automatically transformed into
an unnormalized distribution whose domain is unconstrained.  The transformation
improves the efficiency of sampling and variational inference algorithms.

Usage
#####

For example, one may have prior information that suggests that the value of a
parameter representing a standard deviation is near one.  One could use a
Normal distribution while constraining the support to be positive.  The
specification of a bounded distribution should go within the model block::

    import pymc as pm

    with pm.Model() as model:
        norm = Normal.dist(mu=1.0, sigma=3.0)
        x = Bound('x', norm, lower=0.0)

Caveats
#######

* Bounds cannot be given to variables that are ``observed``.  To model
  truncated data, use a :func:`~pymc.model.Potential` in combination with a cumulative
  probability function.  See `this example notebook <https://docs.pymc.io/pymc-examples/examples/survival_analysis/weibull_aft.html>`_.

* The automatic transformation applied to continuous distributions results in
  an unnormalized probability distribution.  This doesn't effect inference
  algorithms but may complicate some model comparison procedures.

Bounded Variable API
####################


.. currentmodule:: pymc.distributions.bound
.. automodule:: pymc.distributions.bound
   :members:
