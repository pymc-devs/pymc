***************
Transformations
***************

.. currentmodule:: pymc.distributions.transforms

While many distributions are defined on constrained spaces (e.g. intervals), MCMC samplers typically perform best when sampling on the unconstrained real line; this is especially true of HMC samplers. PyMC balances this through the use of transforms. A transform instance can be passed to the constructor of a random variable to tell the sampler how to move between the underlying unconstrained space where the samples are actually drawn and the transformed space constituting the support of the random variable. Transforms are not currently implemented for discrete random variables.

All transforms have three core methods:

* ``forward``: The map from a constrained space to the unconstrained space.
* ``backward``: The inverse map from the unconstrained space to a constrained space.
* ``log_jac_det``: The log of the determinant of the Jacobian of the ``backward`` map. This is used to account for the transformed random variable correctly in the posterior log-probability.

.. note::
   Transforms are principally intended for internal use and in most cases users do not need to change them. In particular, all continuous distributions on a constrained domain that are implemented in PyMC have a ``default_transform`` that will automatically transform the random variables as required without needing any extra work from the user.

The main use-cases for setting custom transforms include the following:

#. The ``default_transform`` may need to be replaced with an alternative transform on the same constained space. For example, the ``default_transform`` for positive-valued random variables is the :class:`log` transform but in some cases it may be advantageous to use the :class:`log_exp_m1` transform instead.
#. The ``default_transform`` may be removed entirely in some cases when using non-HMC samplers.
#. Exceptionally, transforms can be used to *add* constraints to the model specification without modifying the ``default_transform``. This can be done by specifying the additional transform via the ``transform`` parameter. However this should not be viewed as a default use-case and, in practice, this is mostly limited to using :class:`ordered` in mixture models.

   * NB: :class:`ordered` is not guaranteed to work correctly when used in combination with other transforms, such as :class:`simplex` and :class:`ZeroSumTransform`.

.. warning::
   Transforms are **only** applied when sampling *unobserved* random variables with :func:`pymc.sample`. In particular:

   * Transforms are not applied during forward sampling, i.e. :func:`pymc.draw`, :func:`pymc.sample_prior_predictive` and :func:`pymc.sample_posterior_predictive`
   * Transforms are not applied when sampling *observed* random variables with :func:`pymc.sample`

Since transforms are not applied during :func:`pymc.sample_prior_predictive`, a workaround to carry out prior predictive checks is to remove observations from the likelihood and use :func:`pymc.sample` instead.

Transforms are not usually the correct tool to represent transformations that are part of the *generative* specification of the model. Such transformations should be included explicitly in the model, typically via :class:`pymc.Deterministic`. Doing so allows such transformed random variables to be sampled by forward samplers.


Transform Instances
~~~~~~~~~~~~~~~~~~~

Transform instances are the entities that should be used in the
``default_transform`` or ``transform`` parameters to a random variable
constructor.

.. autosummary::
   :toctree: generated

    circular
    log
    log_exp_m1
    logodds
    ordered
    simplex


Specific Transform Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~

An instance of these classes needs to be created before being used
in the ``default_transform`` or ``transform`` parameters to a random variable
constructor.

.. autosummary::
   :toctree: generated

    CholeskyCovPacked
    CircularTransform
    Interval
    LogExpM1
    LogOddsTransform
    LogTransform
    Ordered
    SimplexTransform
    ZeroSumTransform


Transform Composition Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An instance of this class needs to be created from a list of transforms before
being used in the ``transform`` parameter to a random variable constructor.

If a random variable has a ``default_transform`` and an additional transform
is provided through the ``transform`` parameter, PyMC will automatically
create an instance of the :class:`Chain` transform that applies the
user-provided transform on top of the default one.

.. autosummary::
   :toctree: generated

    Chain
