.. _chap_modelfitting:

**************
Fitting Models
**************

PyMC provides three objects that fit models:

* ``MCMC``, which coordinates Markov chain Monte Carlo algorithms. The actual 
  work of updating stochastic variables conditional on the rest of the model is 
  done by ``StepMethod`` objects, which are described in this chapter.

* ``MAP``, which computes maximum *a posteriori* estimates.

* ``NormApprox``, which computes the 'normal approximation' [Gelman2004]_: the 
  joint distribution of all stochastic variables in a model is approximated as 
  normal using local information at the maximum *a posteriori* estimate.

All three objects are subclasses of ``Model``, which is PyMC's base class for 
fitting methods. ``MCMC`` and ``NormApprox``, both of which can produce samples 
from the posterior, are subclasses of ``Sampler``, which is PyMC's base class 
for Monte Carlo fitting methods. ``Sampler`` provides a generic sampling loop 
method and database support for storing large sets of joint samples. These base 
classes implement some basic methods that are inherited by the three 
implemented fitting methods, so they are documented at the end of this section.


.. _sec_modelinstantiation:

Creating models
===============

The first argument to any fitting method's ``__init__`` method, including that 
of ``MCMC``, is called ``input``. The ``input`` argument can be just about 
anything; once you have defined the nodes that make up your model, you 
shouldn't even have to think about how to wrap them in a ``Model`` instance. 
Some examples of model instantiation using nodes ``a``, ``b`` and ``c`` follow:

* ``M = Model(set([a,b,c]))``

* ``M = Model({`a': a, `d': [b,c]})`` In this case, :math:`M` will expose 
  :math:`a` and :math:`d` as attributes: ``M.a`` will be :math:`a`, and ``M.d`` 
  will be ``[b,c]``.

* ``M = Model([[a,b],c])``

* If file ``MyModule`` contains the definitions of ``a``, ``b`` and ``c``::

    import MyModule
    M = Model(MyModule)

In this case, :math:`M` will expose :math:`a`, :math:`b` and :math:`c` as 
attributes.

* Using a 'model factory' function::

    def make_model(x):
        a = pymc.Exponential('a', beta=x, value=0.5)

        @pymc.deterministic
        def b(a=a):
            return 100-a

        @pymc.stochastic
        def c(value=.5, a=a, b=b):
            return (value-a)**2/b

        return locals()

    M = pymc.Model(make_model(3))

In this case, :math:`M` will also expose :math:`a`, :math:`b` and :math:`c` as 
attributes.


.. _sec_model:

The Model class
===============

This class serves as a container for probability models and as a base class for 
the classes responsible for model fitting, such as ``MCMC``.

``Model``'s init method takes the following arguments:

``input``:
   Some collection of PyMC nodes defining a probability model. These may be 
   stored in a list, set, tuple, dictionary, array, module, or any object with 
   a ``__dict__`` attribute.

``verbose`` (optional):
   An integer controlling the verbosity of the model's output.

Models' useful methods are:

``draw_from_prior()``:
   Sets all stochastic variables' values to new random values, which would be a 
   sample from the joint distribution if all data and ``Potential`` instances' 
   log-probability functions returned zero. If any stochastic variables lack a 
   ``random()`` method, PyMC will raise an exception.

``seed()``:
   Same as ``draw_from_prior``, but only ``stochastics`` whose ``rseed`` 
   attribute is not ``None`` are changed.

As introduced in the previous chapter, the helper function ``graph.dag`` 
produces graphical representations of models (see [Jordan2004]_).

Models have the following important attributes:

* ``variables``

* ``nodes``

* ``stochastics``

* ``potentials``

* ``deterministics``

* ``observed_stochastics``

* ``containers``

* ``value``

* ``logp``

In addition, models expose each node they contain as an attribute. For 
instance, if model ``M`` were produced from model (:eq:`disastermodel`) ``M.s`` 
would return the switchpoint variable.


.. _sec_map:

Maximum a posteriori estimates
==============================

The ``MAP`` class sets all stochastic variables to their maximum *a posteriori* 
values using functions in SciPy's ``optimize`` package; hence, SciPy must be 
installed to use it. ``MAP`` can only handle variables whose dtype is 
``float``, so it will not work, for example, on model (:eq:`disastermodel`). To 
fit the model in :file:`examples/gelman_bioassay.py` using ``MAP``, do the 
following::

    >>> from pymc.examples import gelman_bioassay
    >>> M = pymc.MAP(gelman_bioassay)
    >>> M.fit()

This call will cause :math:`M` to fit the model using Nelder-Mead optimization, 
which does not require derivatives. The variables in ``DisasterModel`` have now 
been set to their maximum *a posteriori* values::

    >>> M.alpha.value
    array(0.8465892309923545)
    >>> M.beta.value
    array(7.7488499785334168)

In addition, the AIC and BIC of the model are now available::

    >>> M.AIC
    7.9648372671389458
    >>> M.BIC
    6.7374259893787265

``MAP`` has two useful methods:

``fit(method ='fmin', iterlim=1000, tol=.0001)``:
   The optimization method may be ``fmin``, ``fmin_l_bfgs_b``, ``fmin_ncg``, 
   ``fmin_cg``, or ``fmin_powell``. See the documentation of SciPy's 
   ``optimize`` package for the details of these methods. The ``tol`` and 
   ``iterlim`` parameters are passed to the optimization function under the 
   appropriate names.

``revert_to_max()``:
   If the values of the constituent stochastic variables change after fitting, 
   this function will reset them to their maximum *a posteriori* values.

If you're going to use an optimization method that requires derivatives, 
``MAP``'s ``__init__`` method can take additional parameters ``eps`` and 
``diff_order``. ``diff_order``, which must be an integer, specifies the order 
of the numerical approximation (see the SciPy function ``derivative``). The 
step size for numerical derivatives is controlled by ``eps``, which may be 
either a single value or a dictionary of values whose keys are variables 
(actual objects, not names).

The useful attributes of ``MAP`` are:

``logp``:
   The joint log-probability of the model.

``logp_at_max``:
   The maximum joint log-probability of the model.

``AIC``:
   Akaike's information criterion for this model 
   ([Akaike1973]_,[Burnham2002]_).

``BIC``:
   The Bayesian information criterion for this model [Schwarz1978]_.

One use of the ``MAP`` class is finding reasonable initial states for MCMC 
chains. Note that multiple ``Model`` subclasses can handle the same collection 
of nodes.


.. _sec_norm-approx:

Normal approximations
=====================

The ``NormApprox`` class extends the ``MAP`` class by approximating the 
posterior covariance of the model using the Fisher information matrix, or the 
Hessian of the joint log probability at the maximum. To fit the model in 
:file:`examples/gelman_bioassay.py` using ``NormApprox``, do::

    >>> N = pymc.NormApprox(gelman_bioassay)
    >>> N.fit()

The approximate joint posterior mean and covariance of the variables are 
available via the attributes ``mu`` and ``C``::

    >>> N.mu[N.alpha]
    array([ 0.84658923])
    >>> N.mu[N.alpha, N.beta]
    array([ 0.84658923,  7.74884998])
    >>> N.C[N.alpha]
    matrix([[ 1.03854093]])
    >>> N.C[N.alpha, N.beta]
    matrix([[  1.03854093,   3.54601911],
            [  3.54601911,  23.74406919]])

As with ``MAP``, the variables have been set to their maximum *a posteriori* 
values (which are also in the ``mu`` attribute) and the AIC and BIC of the 
model are available.

In addition, it's now possible to generate samples from the posterior as with 
``MCMC``::

    >>> N.sample(100)
    >>> N.trace('alpha')[::10]
    array([-0.85001278,  1.58982854,  1.0388088 ,  0.07626688,  1.15359581,
           -0.25211939,  1.39264616,  0.22551586,  2.69729987,  1.21722872])
    >>> N.trace('beta')[::10]
    array([  2.50203663,  14.73815047,  11.32166303,   0.43115426,
            10.1182532 ,   7.4063525 ,  11.58584317,   8.99331152,
            11.04720439,   9.5084239 ])

Any of the database backends can be used (chapter :ref:`chap_database`).

In addition to the methods and attributes of ``MAP``, ``NormApprox`` provides 
the following methods:

``sample(iter)``:
   Samples from the approximate posterior distribution are drawn and stored.

``isample(iter)``:
   An 'interactive' version of ``sample()``: sampling can be paused, returning 
   control to the user.

``draw``:
   Sets all variables to random values drawn from the approximate posterior.

It provides the following additional attributes:

``mu``:
   A special dictionary-like object that can be keyed with multiple variables. 
   ``N.mu[p1, p2, p3]`` would return the approximate posterior mean values of 
   stochastic variables ``p1``, ``p2`` and ``p3``, raveled and concatenated to 
   form a vector.

``C``:
   Another special dictionary-like object. ``N.C[p1, p2, p3]`` would return the 
   approximate posterior covariance matrix of stochastic variables ``p1``, 
   ``p2`` and ``p3``. As with ``mu``, these variables' values are raveled and 
   concatenated before their covariance matrix is constructed.


.. _sec_mcmc:

Markov chain Monte Carlo: the MCMC class
========================================

The ``MCMC`` class implements PyMC's core business: producing 'traces' for a 
model's variables which, with careful thinning, can be considered independent 
joint samples from the posterior. See :ref:`chap_tutorial` for an example of 
basic usage.

``MCMC``'s primary job is to create and coordinate a collection of 'step 
methods', each of which is responsible for updating one or more variables. The 
available step methods are described below. Instructions on how to create your 
own step method are available in :ref:`chap_extending`.

``MCMC`` provides the following useful methods:

``sample(iter, burn, thin, tune_interval, tune_throughout, save_interval, ...)``:
   Runs the MCMC algorithm and produces the traces. The ``iter`` argument 
   controls the total number of MCMC iterations. No tallying will be done 
   during the first ``burn`` iterations; these samples will be forgotten. After 
   this burn-in period, tallying will be done each ``thin`` iterations. Tuning 
   will be done each ``tune_interval`` iterations. If 
   ``tune_throughout=False``, no more tuning will be done after the burnin 
   period. The model state will be saved every ``save_interval`` iterations, if 
   given.


``isample(iter, burn, thin, tune_interval, tune_throughout, save_interval, ...)``:
   An interactive version of ``sample``. The sampling loop may be paused at any 
   time, returning control to the user.

``use_step_method(method, *args, **kwargs)``:
   Creates an instance of step method class ``method`` to handle some 
   stochastic variables. The extra arguments are passed to the ``init`` method 
   of ``method``. Assigning a step method to a variable manually will prevent 
   the ``MCMC`` instance from automatically assigning one. However, you may 
   handle a variable with multiple step methods.

``goodness()``:
   Calculates goodness-of-fit (GOF) statistics according to [Brooks2000]_.

``save_state()``:
   Saves the current state of the sampler, including all stochastics, to the 
   database. This allows the sampler to be reconstituted at a later time to 
   resume sampling. This is not supported yet for the ``sqlite`` backend.

``restore_state()``:
   Restores the sampler to the state stored in the database.

``stats()``:
   Generate summary statistics for all nodes in the model.

``remember(trace_index)``:
   Set all variables' values from frame ``trace_index`` in the database.

MCMC samplers' step methods can be accessed via the ``step_method_dict`` 
attribute. ``M.step_method_dict[x]`` returns a list of the step methods ``M`` 
will use to handle the stochastic variable ``x``.

After sampling, the information tallied by ``M`` can be queried via 
``M.db.trace_names``. In addition to the values of variables, tuning 
information for adaptive step methods is generally tallied. These ‘traces’ can 
be plotted to verify that tuning has in fact terminated.

You can produce 'traces' for arbitrary functions with zero arguments as well. 
If you issue the command ``M._funs_to_tally['trace_name'] = f`` before sampling 
begins, then each time the model variables’ values are tallied, ``f`` will be 
called with no arguments, and the return value will be tallied. After sampling 
ends you can retrieve the trace as ``M.trace[’trace_name’]``.


.. _sec_sampler:

The Sampler class
=================

``MCMC`` is a subclass of a more general class called ``Sampler``. Samplers fit 
models with Monte Carlo fitting methods, which characterize the posterior 
distribution by approximate samples from it. They are initialized as follows: 
``Sampler(input=None, db='ram', name='Sampler', reinit_model=True, 
calc_deviance=False, verbose=0)``. The ``input`` argument is a module, list, 
tuple, dictionary, set, or object that contains all elements of the model, the 
``db`` argument indicates which database backend should be used to store the 
samples (see chapter :ref:`chap_database`), ``reinit_model`` is a boolean flag 
that indicates whether the model should be re-initialised before running, and 
``calc_deviance`` is a boolean flag indicating whether deviance should be 
calculated for the model at each iteration. Samplers have the following 
important methods:

``sample(iter, length, verbose, ...)``:
   Samples from the joint distribution. The ``iter`` argument controls how many 
   times the sampling loop will be run, and the ``length`` argument controls 
   the initial size of the database that will be used to store the samples.

``isample(iter, length, verbose, ...)``:
   The same as ``sample``, but the sampling is done interactively: you can 
   pause sampling at any point and be returned to the Python prompt to inspect 
   progress and adjust fitting parameters. While sampling is paused, the 
   following methods are useful:

``icontinue()``: 
	Continue interactive sampling.

``halt()``:
    Truncate the database and clean up.

``tally()``:
   Write all variables' current values to the database. The actual write 
   operation depends on the specified database backend.

``save_state()``:
   Saves the current state of the sampler, including all stochastics, to the 
   database. This allows the sampler to be reconstituted at a later time to 
   resume sampling. This is not supported yet for the ``sqlite`` backend.

``restore_state()``:
   Restores the sampler to the state stored in the database.

``stats()``:
   Generate summary statistics for all nodes in the model.

``remember(trace_index)``:
   Set all variables' values from frame ``trace_index`` in the database. Note 
   that the ``trace_index`` is different from the current iteration, since not 
   all samples are necessarily saved due to burning and thinning.

In addition, the sampler attribute ``deviance`` is a deterministic variable 
valued as the model's deviance at its current state.


.. _sec_stepmethod:

Step methods
============

Step method objects handle individual stochastic variables, or sometimes groups 
of them. They are responsible for making the variables they handle take single 
MCMC steps conditional on the rest of the model. Each subclass of 
``StepMethod`` implements a method called ``step()``, which is called by 
``MCMC``. Step methods with adaptive tuning parameters can optionally implement 
a method called ``tune()``, which causes them to assess performance (based on 
the acceptance rates of proposed values for the variable) so far and adjust.

The major subclasses of ``StepMethod`` are ``Metropolis``, 
``AdaptiveMetropolis`` and ``Gibbs``. PyMC provides several flavors of the 
basic Metropolis steps. The ``Gibbs`` steps are not ready for use as of the 
current release, but since it is feasible to write Gibbs step methods for 
particular applications, the ``Gibbs`` base class will be documented here.

.. _metropolis:

Metropolis step methods
-----------------------

``Metropolis`` and subclasses implement Metropolis-Hastings steps. To tell an 
``MCMC`` object :math:`M` to handle a variable :math:`x` with a Metropolis step 
method, you might do the following::

    M.use_step_method(pymc.Metropolis, x, proposal_sd=1., proposal_distribution='Normal')

``Metropolis`` itself handles float-valued variables, and subclasses 
``DiscreteMetropolis`` and ``BinaryMetropolis`` handle integer- and 
boolean-valued variables, respectively. Subclasses of ``Metropolis`` must 
implement the following methods:

``propose()``:
   Sets the values of the variables handled by the Metropolis step method to proposed values.

``reject()``:
   If the Metropolis-Hastings acceptance test fails, this method is called to 
   reset the values of the variables to their values before ``propose()`` was 
   called.

Note that there is no ``accept()`` method; if a proposal is accepted, the 
variables' values are simply left alone. Subclasses that use proposal 
distributions other than symmetric random-walk may specify the 'Hastings 
factor' by changing the ``hastings_factor`` method. See :ref:`chap_extending` 
for an example.

``Metropolis``' ``__init__`` method takes the following arguments:

``stochastic``:
   The variable to handle.

``proposal_sd``:
   A float or array of floats. This sets the proposal standard deviation if the 
   proposal distribution is normal.

``scale``:
   A float, defaulting to 1. If the ``scale`` argument is provided but not 
   ``proposal_sd``, ``proposal_sd`` is computed as follows::

      if all(self.stochastic.value != 0.):
          self.proposal_sd = ones(shape(self.stochastic.value)) * \
                              abs(self.stochastic.value) * scale
      else:
          self.proposal_sd = ones(shape(self.stochastic.value)) * scale

``proposal_distribution``:
   A string indicating which distribution should be used for proposals. Current 
   options are ``'Normal'`` and ``'Prior'``. If ``proposal_distribution=None``, 
   the proposal distribution is chosen automatically. It is set to ``'Prior'`` 
   if the variable has no children and has a random method, and to ``'Normal'`` 
   otherwise.

``verbose``:
   An integer. By convention 0 indicates no output, 1 shows a progress bar 
   only, 2 provides basic feedback about the current MCMC run, while 3 and 4 
   provide low and high debugging verbosity, respectively.

Alhough the ``proposal_sd`` attribute is fixed at creation, Metropolis step 
methods adjust their initial proposal standard deviations using an attribute 
called ``adaptive_scale_factor``. When ``tune()`` is called, the acceptance 
ratio of the step method is examined, and this scale factor is updated 
accordingly. If the proposal distribution is normal, proposals will have 
standard deviation ``self.proposal_sd * self.adaptive_scale_factor``.

By default, tuning will continue throughout the sampling loop, even after the 
burnin period is over. This can be changed via the ``tune_throughout`` argument 
to ``MCMC.sample``. If an adaptive step method's ``tally`` flag is set (the 
default for ``Metropolis``), a trace of its tuning parameters will be kept. If 
you allow tuning to continue throughout the sampling loop, it is important to 
verify that the 'Diminishing Tuning' condition of [Roberts2007]_ is satisfied: 
the amount of tuning should decrease to zero, or tuning should become very 
infrequent.

If a Metropolis step method handles an array-valued variable, it proposes all 
elements independently but simultaneously. That is, it decides whether to 
accept or reject all elements together but it does not attempt to take the 
posterior correlation between elements into account. The ``AdaptiveMetropolis`` 
class (see below), on the other hand, does make correlated proposals.

.. _subsec:adaptive_metropolis:

The AdaptiveMetropolis class
----------------------------

The ``AdaptativeMetropolis`` (AM) step method works like a regular Metropolis 
step method, with the exception that its variables are block-updated using a 
multivariate jump distribution whose covariance is tuned during sampling. 
Although the chain is non-Markovian, it has correct ergodic properties (see 
[Haario2001]_).

To tell an ``MCMC`` object :math:`M` to handle variables :math:`x`, :math:`y` 
and :math:`z` with an ``AdaptiveMetropolis`` instance, you might do the 
following::

   M.use_step_method(pymc.AdaptiveMetropolis, [x,y,z], \
                      scales={'x':1, 'y':2, 'z':.5}, delay=10000)

``AdaptativeMetropolis``'s init method takes the following arguments:

``stochastics``:
   The stochastic variables to handle. These will be updated jointly.

``cov`` (optional):
   An initial covariance matrix. Defaults to the identity matrix, adjusted 
   according to the ``scales`` argument.

``delay`` (optional):
   The number of iterations to delay before computing the empirical covariance 
   matrix.

``scales`` (optional):
   The initial covariance matrix will be diagonal, and its diagonal elements 
   will be set to ``scales`` times the stochastics' values, squared.

``interval`` (optional):
   The number of iterations between updates of the covariance matrix. Defaults 
   to 1000.

``greedy`` (optional):
   If ``True``, only accepted jumps will be counted toward the delay before the 
   covariance is first computed. Defaults to ``True``.

``verbose``:
   An integer from 0 to 4 controlling the verbosity of the step method's 
   printed output.
	
``shrink_if_necessary`` (optional): 
    Whether the proposal covariance should be shrunk if the acceptance rate 
    becomes extremely small.

In this algorithm, jumps are proposed from a multivariate normal distribution 
with covariance matrix :math:`\Sigma`. The algorithm first iterates until 
``delay`` samples have been drawn (if ``greedy`` is true, until ``delay`` jumps 
have been accepted). At this point, :math:`\Sigma` is given the value of the 
empirical covariance of the trace so far and sampling resumes. The covariance 
is then updated each ``interval`` iterations throughout the entire sampling run 
[#]_. It is this constant adaptation of the proposal distribution that makes 
the chain non-Markovian.

The DiscreteMetropolis class
----------------------------

This class is just like ``Metropolis``, but specialized to handle 
``Stochastic`` instances with dtype ``int``. The jump proposal distribution can 
either be ``'Normal'``, ``'Prior'`` or ``'Poisson'`` (the default). In the 
normal case, the proposed value is drawn from a normal distribution centered at 
the current value and then rounded to the nearest integer.

The BinaryMetropolis class
--------------------------

This class is specialized to handle ``Stochastic`` instances with dtype 
``bool``.

For array-valued variables, ``BinaryMetropolis`` can be set to propose from the 
prior by passing in ``dist="Prior"``. Otherwise, the argument ``p_jump`` of the 
init method specifies how probable a change is. Like ``Metropolis``' attribute 
``proposal_sd``, ``p_jump`` is tuned throughout the sampling loop via 
``adaptive_scale_factor``.

For scalar-valued variables, ``BinaryMetropolis`` behaves like a Gibbs sampler, 
since this requires no additional expense. The ``p_jump`` and 
``adaptive_scale_factor`` parameters are not used in this case.


.. _gibbs:

Gibbs step methods
==================


Gibbs step methods handle conjugate submodels. These models usually have two 
components: the `parent' and the `children'. For example, a gamma-distributed 
variable serving as the precision of several normally-distributed variables is 
a conjugate submodel; the gamma variable is the parent and the normal variables 
are the children.

This section describes PyMC's current scheme for Gibbs step methods, several of 
which are in a semi-working state in the sandbox. It is meant to be as generic 
as possible to minimize code duplication, but it is admittedly complicated. 
Feel free to subclass ``StepMethod`` directly when writing Gibbs step methods 
if you prefer.

Gibbs step methods that subclass PyMC's ``Gibbs`` should define the following 
class attributes:

``child_class``:
	The class of the children in the submodels the step method can handle.
	
``parent_class``:
	The class of the parent.
	
``parent_label``: 
	The label the children would apply to the parent in a conjugate submodel. 	
	In the gamma-normal example, this would be ``tau``.

``linear_OK``:
	A flag indicating whether the children can use linear combinations 
	involving the parent as their actual parent without destroying the 
	conjugacy.

A subclass of ``Gibbs`` that defines these attributes only needs to implement a 
``propose()`` method, which will be called by ``Gibbs.step()``. The resulting 
step method will be able to handle both conjugate and 'non-conjugate' cases. 
The conjugate case corresponds to an actual conjugate submodel. In the 
nonconjugate case all the children are of the required class, but the parent is 
not. In this case the parent's value is proposed from the likelihood and 
accepted based on its prior. The acceptance rate in the nonconjugate case will 
be less than one.

The inherited class method ``Gibbs.competence`` will determine the new step 
method's ability to handle a variable :math:`x` by checking whether:

* all :math:`x`'s children are of class ``child_class``, and either apply 
  ``parent_label`` to `x` directly or (if ``linear_OK=True``) to a 
  ``LinearCombination`` object (:ref:`chap_modelbuilding`), one of whose 
  parents contains $x$.

* :math:`x` is of class ``parent_class``

If both conditions are met, ``pymc.conjugate_Gibbs_competence`` will be 
returned. If only the first is met, ``pymc.nonconjugate_Gibbs_competence`` will 
be returned.

.. _subsec:granularity:

Granularity of step methods: one-at-a-time vs. block updating
-------------------------------------------------------------

There is currently no way for a stochastic variable to compute individual terms 
of its log-probability; it is computed all together. This means that updating 
the elements of a array-valued variable individually would be inefficient, so 
all existing step methods update array-valued variables together, in a block 
update.

To update an array-valued variable's elements individually, simply break it up 
into an array of scalar-valued variables. Instead of this::

    A = pymc.Normal('A', value=zeros(100), mu=0., tau=1.)

do this::

    A = [pymc.Normal('A_%i'%i, value=0., mu=0., tau=1.) for i in range(100)]

An individual step method will be assigned to each element of ``A`` in the 
latter case, and the elements will be updated individually. Note that ``A`` can 
be broken up into larger blocks if desired.

Automatic assignment of step methods
------------------------------------

Every step method subclass (including user-defined ones) that does not require 
any ``__init__`` arguments other than the stochastic variable to be handled 
adds itself to a list called ``StepMethodRegistry`` in the PyMC namespace. If a 
stochastic variable in an ``MCMC`` object has not been explicitly assigned a 
step method, each class in ``StepMethodRegistry`` is allowed to examine the 
variable.

To do so, each step method implements a class method called 
``competence(stochastic)``, whose only argument is a single stochastic 
variable. These methods return values from 0 to 3; 0 meaning the step method 
cannot safely handle the variable and 3 meaning it will most likely perform 
well for variables like this. The ``MCMC`` object assigns the step method that 
returns the highest competence value to each of its stochastic variables.

.. rubric:: Footnotes

.. [#] The covariance is estimated recursively from the previous value and the last
   ``interval`` samples, instead of computing it each time from the entire trace.