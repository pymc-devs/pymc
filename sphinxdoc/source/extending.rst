.. _chap:extending:

**************
Extending PyMC
**************

PyMC tries to make standard things easy, but keep unusual things possible. Its
openness, combined with Python's flexibility, invite extensions from using new
step methods to exotic stochastic processes (see the Gaussian process module).
This chapter briefly reviews the ways PyMC is designed to be extended.


.. _nonstandard:

Nonstandard Stochastics
=======================

The simplest way to create a ``Stochastic`` object with a nonstandard
distribution is to use the medium or long decorator syntax. See chapter
:ref:`chap:modelbuilding`. If you want to create many stochastics with the same
nonstandard distribution, the decorator syntax can become cumbersome. An actual
subclass of ``Stochastic`` can be created using the class factory
``stochastic_from_dist``. This function takes the following arguments:

* The name of the new class,

* A ``logp`` function,

* A ``random`` function,

* The NumPy datatype of the new class (for continuous distributions, this should
  be ``float``; for discrete distributions, ``int``; for variables valued as non-
  numerical objects, ``object``),

* A flag indicating whether the resulting class represents a vector-valued
  variable.

The necessary parent labels are read from the ``logp`` function, and a docstring
for the new class is automatically generated. Instances of the new class can be
created in one line.

Full subclasses of ``Stochastic`` may be necessary to provide nonstandard
behaviors (see ``gp.GP``).


.. _custom-stepper:

User-defined step methods
=========================

The ``StepMethod`` class is meant to be subclassed. There are an enormous number
of MCMC step methods in the literature, whereas PyMC provides only about half a
dozen. Most user-defined step methods will be either Metropolis-Hastings or
Gibbs step methods, and these should subclass ``Metropolis`` or ``Gibbs``
respectively. More unusual step methods should subclass ``StepMethod`` directly.



Example: an asymmetric Metropolis step
--------------------------------------

Consider the probability model in :file:`examples/custom_step.py`::

   mu = pm.Normal('mu',0,.01, value=0)
   tau = pm.Exponential('tau',.01, value=1)
   cutoff = pm.Exponential('cutoff',1, value=1.3)
   D = pm.Truncnorm('D',mu,tau,-np.inf,cutoff,value=data,observed=True)

The stochastic variable ``cutoff`` cannot be smaller than the largest element of
:math:`D`, otherwise :math:`D`'s density would be zero. The standard
``Metropolis`` step method can handle this case without problems; it will
propose illegal values occasionally, but these will be rejected.

Suppose we want to handle ``cutoff`` with a smarter step method that doesn't
propose illegal values. Specifically, we want to use the nonsymmetric proposal
distribution 

.. math::
  :nowrap:

  \begin{eqnarray*}
  	x_p | x \sim \textup{Truncnorm}(x, \sigma, \max(D), \infty).
  \end{eqnarray*}


We can implement this Metropolis-Hastings algorithm with the following step
method class::

   class TruncatedMetropolis(pm.Metropolis):
       def __init__(self, stochastic, low_bound, up_bound, *args, **kwargs):
           self.low_bound = low_bound
           self.up_bound = up_bound
           pm.Metropolis.__init__(self, stochastic, *args, **kwargs)

       # Propose method written by hacking Metropolis.propose()
       def propose(self):
           tau = 1./(self.adaptive_scale_factor * self.proposal_sd)**2
           self.stochastic.value = \
           pm.rtruncnorm(self.stochastic.value, tau, self.low_bound, self.up_bound)

       # Hastings factor method accounts for asymmetric proposal distribution
       def hastings_factor(self):
           tau = 1./(self.adaptive_scale_factor * self.proposal_sd)**2
           cur_val = self.stochastic.value
           last_val = self.stochastic.last_value

           lp_for = pm.truncnorm_like(cur_val, last_val, tau, self.low_bound, self.up_bound)
           lp_bak = pm.truncnorm_like(last_val, cur_val, tau, self.low_bound, self.up_bound)

           if self.verbose > 1:
               print self._id + ': Hastings factor %f'%(lp_bak - lp_for)
           return lp_bak - lp_for

The ``propose`` method sets the step method's stochastic's value to a new value,
drawn from a truncated normal distribution. The precision of this distribution
is computed from two factors: ``self.proposal_sd``, which can be set with an
input argument to Metropolis, and ``self.adaptive_scale_factor``. Metropolis
step methods' default tuning behavior is to reduce ``adaptive_scale_factor`` if
the acceptance rate is too low, and to increase ``adaptive_scale_factor`` if it
is too high. By incorporating ``adaptive_scale_factor`` into the proposal
standard deviation, we avoid having to write our own tuning infrastructure. If
we don't want the proposal to tune, we don't have to use
``adaptive_scale_factor``.

The ``hastings_factor`` method adjusts for the asymmetric proposal distribution
[Gelman:2004]_. It computes the log of the quotient of the 'backward' density and the
'forward' density. For symmetric proposal distributions, this quotient is 1, so
its log is zero. We have added some code to print the Hastings factor if the
step method's verbosity level is set high.

Having created our custom step method, we need to tell MCMC instances to use it
to handle the variable ``cutoff``. This is done in :file:`custom_step.py` with
the following line::

   M.use_step_method(TruncatedMetropolis, cutoff, D.value.max(), np.inf)

This call causes :math:`M` to pass the arguments ``cutoff, D.value.max(),
np.inf`` to a ``TruncatedMetropolis`` object's ``init`` method, and use the
object to handle ``cutoff``.

It's often convenient to get a handle to a custom step method instance directly
for debugging purposes. ``M.step_method_dict[cutoff]`` returns a list of all the
step methods :math:`M` will use to handle ``cutoff``::

   >>> M.step_method_dict[cutoff]
   [<custom_step.TruncatedMetropolis object at 0x3c91130>]

There may be more than one, and conversely step methods may handle more than one
stochastic variable. To see which variables step method :math:`S` is handling,
try  ::

   >>> S.stochastics
   set([<pymc.distributions.Exponential 'cutoff' at 0x3cd6b90>])



General step methods
--------------------

All step methods must implement the following methods:

``step()``:
   Updates the values of ``self.stochastics``.

``tune()``:
   Tunes the jumping strategy based on performance so far. A default method is
   available that increases ``self.adaptive_scale_factor`` (see below) when
   acceptance rate is high, and decreases it when acceptance rate is low. This
   method should return ``True`` if additional tuning will be required later, and
   ``False`` otherwise.

``competence(s):``
   A class method that examines stochastic variable :math:`s` and returns a value
   from 0 to 3 expressing the step method's ability to handle the variable. This
   method is used by ``MCMC`` instances when automatically assigning step methods.
   Conventions are:

   0
      I cannot safely handle this variable.

   1
      I can handle the variable about as well as the standard ``Metropolis`` step
      method.

   2
      I can do better than ``Metropolis``.

   3
      I am the best step method you are likely to find for this variable in most
      cases.

   For example, if you write a step method that can handle ``MyStochasticSubclass``
   well, the competence method might look like this::

      class MyStepMethod(pm.StepMethod):
         def __init__(self, stochastic, *args, **kwargs):
            ...

         @classmethod
         def competence(self, stochastic):
            if isinstance(stochastic, MyStochasticSubclass):
               return 3
            else:
               return 0

   Note that PyMC will not even attempt to assign a step method automatically if
   its ``init`` method cannot be called with a single stochastic instance, that is
   ``MyStepMethod(x)`` is a legal call. The list of step methods that PyMC will
   consider assigning automatically is called ``pymc.StepMethodRegistry``.

``current_state()``:
   This method is easiest to explain by showing the code::

      state = {}
      for s in self._state:
          state[s] = getattr(self, s)
      return state

   ``self._state`` should be a list containing the names of the attributes needed
   to reproduce the current jumping strategy. If an ``MCMC`` object writes its
   state out to a database, these attributes will be preserved. If an ``MCMC``
   object restores its state from the database later, the corresponding step method
   will have these attributes set to their saved values.

Step methods should also maintain the following attributes:

``_id``:
   A string that can identify each step method uniquely (usually something like
   ``<class_name>_<stochastic_name>``).

``adaptive_scale_factor``:
   An 'adaptive scale factor'. This attribute is only needed if the default
   ``tune()`` method is used.

``_tuning_info``:
   A list of strings giving the names of any tuning parameters. For ``Metropolis``
   instances, this would be ``adaptive_scale_factor``. This list is used to keep
   traces of tuning parameters in order to verify 'diminishing tuning' [Roberts:2007]_.

All step methods have a property called ``loglike``, which returns the sum of
the log-probabilities of the union of the extended children of
``self.stochastics``. This quantity is one term in the log of the Metropolis-
Hastings acceptance ratio.


.. _user-metro:

Metropolis-Hastings step methods
--------------------------------

A Metropolis-Hastings step method only needs to implement the following methods,
which are called by ``Metropolis.step()``:

``reject()``:
   Usually just  ::

      def reject(self):
          self.rejected += 1
          [s.value = s.last_value for s in self.stochastics]

``propose():``
   Sets the values of all ``self.stochastics`` to new, proposed values. This method
   may use the ``adaptive_scale_factor`` attribute to take advantage of the
   standard tuning scheme.

Metropolis-Hastings step methods may also override the ``tune`` and
``competence`` methods.

Metropolis-Hastings step methods with asymmetric jumping distributions may
implement a method called ``hastings_factor()``, which returns the log of the
ratio of the 'reverse' and 'forward' proposal probabilities. Note that no
``accept()`` method is needed or used.

By convention, Metropolis-Hastings step methods use attributes called
``accepted`` and ``rejected`` to log their performance.


.. _user-gibbs:

Gibbs step methods
------------------

Gibbs step methods handle conjugate submodels. These models usually have two
components: the 'parent' and the 'children'. For example, a gamma-distributed
variable serving as the precision of several normally-distributed variables is a
conjugate submodel; the gamma variable is the parent and the normal variables
are the children.

This section describes PyMC's current scheme for Gibbs step methods, several of
which are in a semi-working state in the sandbox. It is meant to be as generic
as possible to minimize code duplication, but it is admittedly complicated. Feel
free to subclass StepMethod directly when writing Gibbs step methods if you
prefer.

Gibbs step methods that subclass PyMC's ``Gibbs`` should define the following
class attributes:

``child_class``:
   The class of the children in the submodels the step method can handle.

``parent_class``:
   The class of the parent.

``parent_label``:
   The label the children would apply to the parent in a conjugate submodel. In the
   gamma-normal example, this would be ``tau``.

``linear_OK``:
   A flag indicating whether the children can use linear combinations involving the
   parent as their actual parent without destroying the conjugacy.

A subclass of ``Gibbs`` that defines these attributes only needs to implement a
``propose()`` method, which will be called by ``Gibbs.step()``. The resulting
step method will be able to handle both conjugate and 'non-conjugate' cases. The
conjugate case corresponds to an actual conjugate submodel. In the nonconjugate
case all the children are of the required class, but the parent is not. In this
case the parent's value is proposed from the likelihood and accepted based on
its prior. The acceptance rate in the nonconjugate case will be less than one.

The inherited class method ``Gibbs.competence`` will determine the new step
method's ability to handle a variable :math:`x` by checking whether:

* all :math:`x`'s children are of class ``child_class``, and either apply
  ``parent_label`` to :math:`x` directly or (if ``linear_OK=True``) to a
  ``LinearCombination`` object (chapter :ref:`chap:modelbuilding`), one of whose
  parents contains :math:`x`.

* :math:`x` is of class ``parent_class``

If both conditions are met, ``pymc.conjugate_Gibbs_competence`` will be
returned. If only the first is met, ``pymc.nonconjugate_Gibbs_competence`` will
be returned.


.. _custom-model:

New fitting algorithms
======================

PyMC provides a convenient platform for non-MCMC fitting algorithms in addition
to MCMC. All fitting algorithms should be implemented by subclasses of
``Model``. There are virtually no restrictions on fitting algorithms, but many
of ``Model``'s behaviors may be useful. See chapter :ref:`chap:modelfitting`.


.. _custom-mc:

Monte Carlo fitting algorithms
------------------------------

Unless there is a good reason to do otherwise, Monte Carlo fitting algorithms
should be implemented by subclasses of ``Sampler`` to take advantage of the
interactive sampling feature and database backends. Subclasses using the
standard ``sample()`` and ``isample()`` methods must define one of two methods:

``draw()``:
   If it is possible to generate an independent sample from the posterior at every
   iteration, the ``draw`` method should do so. The default ``_loop`` method can be
   used in this case.

``_loop()``:
   If it is not possible to implement a ``draw()`` method, but you want to take
   advantage of the interactive sampling option, you should override ``_loop()``.
   This method is responsible for generating the posterior samples and calling
   ``tally()`` when it is appropriate to save the model's state. In addition,
   ``_loop`` should monitor the sampler's ``status`` attribute at every iteration
   and respond appropriately. The possible values of ``status`` are:

   ``'ready'``:
      Ready to sample.

   ``'running'``:
      Sampling should continue as normal.

   ``'halt'``:
      Sampling should halt as soon as possible. ``_loop`` should call the ``halt()``
      method and return control. ``_loop`` can set the status to ``'halt'`` itself if
      appropriate (eg the database is full or a ``KeyboardInterrupt`` has been
      caught).

   ``'paused'``:
      Sampling should pause as soon as possible. ``_loop`` should return, but should
      be able to pick up where it left off next time it's called.

Samplers may alternatively want to override the default ``sample()`` method. In
that case, they should call the ``tally()`` method whenever it is appropriate to
save the current model state. Like custom ``_loop()`` methods, custom
``sample()`` methods should handle ``KeyboardInterrupts`` and call the
``halt()`` method when sampling terminates to finalize the traces.


.. _dont-update-indepth:

Don't update stochastic variables' values in-place
==================================================

If you're going to implement a new step method, fitting algorithm or unusual
(non-numeric-valued) ``Stochastic`` subclass, you should understand the issues
related to in-place updates of ``Stochastic`` objects' values. Fitting methods
should never update variables' values in-place for two reasons:

* In algorithms that involve accepting and rejecting proposals, the 'pre-
  proposal' value needs to be preserved uncorrupted. It would be possible to make
  a copy of the pre-proposal value and then allow in-place updates, but in PyMC we
  have chosen to store the pre-proposal value as ``Stochastic.last_value`` and
  require proposed values to be new objects. In-place updates would corrupt
  ``Stochastic.last_value``, and this would cause problems.

* ``LazyFunction``'s caching scheme checks variables' current values against its
  internal cache by reference. That means if you update a variable's value in-
  place, it or its child may miss the update and incorrectly skip recomputing its
  value or log-probability.

However, a ``Stochastic`` object's value can make in-place updates to itself if
the updates don't change its identity. For example, the ``Stochastic`` subclass
``gp.GP`` is valued as a ``gp.Realization`` object. GP realizations represent
random functions, which are infinite-dimensional stochastic processes, as
literally as possible. The strategy they employ is to 'self-discover' on demand:
when they are evaluated, they generate the required value conditional on
previous evaluations and then make an internal note of it. This is an in-place
update, but it is done to provide the same behavior as a single random function
whose value everywhere has been determined since it was created.

