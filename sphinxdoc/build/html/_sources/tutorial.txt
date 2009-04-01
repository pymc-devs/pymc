********
Tutorial
********

An example statistical model
============================

Consider the following dataset, which is a time series of recorded coal
mining disasters in the UK from 1851 to 1962 [Jarrett:1979]_. 

.. _disastersts:

.. figure:: disastersts_web.png
   :alt: Disasters time series.
   :scale: 100
   :align: center

   Number of mining disasters each year in the UK. 

Occurrences of disasters in the time series is thought to be derived from
a Poisson process with a large rate parameter in the early part of the
time series, and from one with a smaller rate in the later part. We are
interested in locating the change point in the series, which perhaps is
related to changes in mining safety regulations.

We represent our conceptual model formally as a statistical model:

.. math::   :label: disastermodel

       \begin{array}{ccc}
           (D_t | s, e, l) \sim \textup{Poisson}\left(r_t\right), & r_t=\left\{\begin{array}{lll}
               e &\textup{if}& t< s\\ l &\textup{if}& t\ge s
               \end{array}\right.,&t\in[t_l,t_h]\\
           s\sim \textup{Discrete Uniform}(t_l, t_h)\\
           e\sim \textup{Exponential}(r_e)\\
           l\sim \textup{Exponential}(r_l)
       \end{array}
       
The symbols are defined as:

* :math:`D_t`: The number of disasters in year :math:`t`.
* :math:`r_t`: The rate parameter of the Poisson distribution of disasters in year :math:`t`.
* :math:`s`:   The year in which the rate parameter changes (the switchpoint).
* :math:`e`:   The rate parameter before the switchpoint :math:`s`.
* :math:`l`:   The rate parameter after the switchpoint :math:`s`.
* :math:`t_l`, :math:`t_h`:    The lower and upper boundaries of year :math:`t`.
* :math:`r_e`, :math:`r_l`:    The rate parameters of the priors of the early and late rates, respectively.

Because we have defined :math:`D` by its dependence on :math:`s`,
:math:`e` and :math:`l`, the latter three are known as the 'parents' of
:math:`D` and :math:`D` is called their 'child'. Similarly, the parents of
:math:`s` are :math:`t_l` and :math:`t_h`, and :math:`s` is the child of
:math:`t_l` and :math:`t_h`.


Two types of variables
======================

At the model-specification stage (before the data are observed), :math:`D`,
:math:`s`, :math:`e`, :math:`r` and :math:`l` are all random variables. Bayesian
'random' variables have not necessarily arisen from a physical random process.
The Bayesian interpretation of probability is *epistemic*, meaning random
variable :math:`x`'s probability distribution :math:`p(x)` represents our
knowledge and uncertainty about :math:`x`'s value. Candidate values of :math:`x`
for which :math:`p(x)` is high are relatively more probable, given what we know.
Random variables are represented in PyMC by the classes ``Stochastic`` and
``Deterministic``.

The only ``Deterministic`` in the model is :math:`r`. If we knew the values of
:math:`r`'s parents (:math:`s`, :math:`l` and :math:`e`), we could compute the
value of :math:`r` exactly. A ``Deterministic`` like :math:`r` is defined by a
mathematical function that returns its value given values for its parents. The
nomenclature is a bit confusing, because these objects usually represent random
variables; since the parents of :math:`r` are random, :math:`r` is random also.
A more descriptive (though more awkward) name for this class would be
``DeterminedByValuesOfParents``.

On the other hand, even if the values of the parents of variables :math:`s`,
:math:`D` (before observing the data), :math:`e` or :math:`l` were known, we
would still be uncertain of their values. These variables are characterized by
probability distributions that express how plausible their candidate values are,
given values for their parents. The ``Stochastic`` class represents these
variables. A more descriptive name for these objects might be
``RandomEvenGivenValuesOfParents``.

We can represent model :eq:`disastermodel` in a file called
:mod:`DisasterModel.py` as follows. First, we import the PyMC and NumPy
namespaces::

   from pymc import DiscreteUniform, Exponential, deterministic, Poisson, Uniform
   import numpy as np

Notice that from ``pymc`` we have only imported a select few objects that are
needed for this particular model, whereas the entire ``numpy`` namespace has
been imported, and conveniently given a shorter name. Objects from NumPy are
subsequently accessed by prefixing ``np.`` to the name. Either approach is
acceptable.

Next, we enter the actual data values into an array::

   disasters_array =   np.array([ 4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                      3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                      2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
                      1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                      0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                      3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                      0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])

Next, we create the switchpoint variable :math:`s`::

   s = DiscreteUniform('s', lower=0, upper=110, doc='Switchpoint[year]')

``DiscreteUniform`` is a subclass of ``Stochastic`` that represents uniformly-
distributed discrete variables. Use of this distribution suggests that we have
no preference *a priori* regarding the location of the switchpoint; all values
are equally likely. Now we create the exponentially-distributed variables
:math:`e` and :math:`l` for the early and late Poisson rates, respectively::

   e = Exponential('e', beta=1)
   l = Exponential('l', beta=1)

Next, we define the variable :math:`r`, which selects the early rate :math:`e`
for times before :math:`s` and the late rate :math:`l` for times after
:math:`s`. We create :math:`r` using the ``deterministic`` decorator, which
converts the ordinary Python function :math:`r` into a ``Deterministic`` object.
::

   @deterministic(plot=False)
   def r(s=s, e=e, l=l):
   	""" Concatenate Poisson means """
       out = np.empty(len(disasters_array))
       out[:s] = e
       out[s:] = l
       return out

The last step is to define the number of disasters :math:`D`. This is a
stochastic variable, but unlike :math:`s`, :math:`e` and :math:`l` we have
observed its value. To express this, we set the argument ``observed`` to
``True`` (it is set to ``False`` by default). This tells PyMC that this object's
value should not be changed::

   D = Poisson('D', mu=r, value=disasters_array, observed=True)


Why are data and unknown variables represented by the same object?
------------------------------------------------------------------

Since it is represented by a ``Stochastic`` object, :math:`D` is defined by its
dependence on its parent :math:`r` even though its value is fixed. This isn't
just a quirk of PyMC's syntax; Bayesian hierarchical notation itself makes no
distinction between random variables and data. The reason is simple: to use
Bayes' theorem to compute the posterior :math:`p(e,s,l|D)` of model
:eq:`disastermodel`, we require the likelihood :math:`p(D|e,s,l)=p(D|r)`. Even
though :math:`D`'s value is known and fixed, we need to formally assign it a
probability distribution as if it were a random variable. Remember, the
likelihood and the probability function are essentially the same, except that
the former is regarded as a function of the parameters and the latter as a
function of the data.

This point can be counterintuitive at first, as many peoples' instinct is to
regard data as fixed a priori and unknown variables as dependent on the data.
One way to understand this is to think of statistical models like
(:eq:`disastermodel`) as predictive models for data, or as models of the
processes that gave rise to data. Before observing the value of :math:`D`, we
could have sampled from its prior predictive distribution :math:`p(D)` (*i.e.*
the marginal distribution of the data) as follows:

#. Sample :math:`e`, :math:`s` and :math:`l` from their priors.

#. Sample :math:`D` conditional on these values.

Even after we observe the value of :math:`D`, we need to use this process model
to make inferences about :math:`e`, :math:`s` and :math:`l` because its the only
information we have about how the variables are related.


Parents and children
====================

We have above created a PyMC probability model, which is simply a linked
collection of variables. To see the nature of the links, import or run
``DisasterModel.py`` and examine :math:`s`'s ``parents`` attribute from the
Python prompt::

   >>> s.parents
   >>> {'lower': 0, 'upper': 110}

The ``parents`` dictionary shows us the distributional parameters of :math:`s`,
which are constants. Now let's examinine :math:`D`'s parents::

   >>> D.parents
   >>> {'mu': <pymc.PyMCObjects.Deterministic 'r' at 0x3e51a70>}

We are using :math:`r` as a distributional parameter of :math:`D` (*i.e.*
:math:`r` is :math:`D`'s parent). :math:`D` internally labels :math:`r` as
``mu``, meaning :math:`r` plays the role of the rate parameter in :math:`D`'s
Poisson distribution. Now examine :math:`r`'s ``children`` attribute::

   >>> r.children
   >>> set([<pymc.distributions.Poisson 'D' at 0x3e51290>])

Because :math:`D` considers :math:`r` its parent, :math:`r` considers :math:`D`
its child. Unlike ``parents``, ``children`` is a set (an unordered collection of
objects); variables do not associate their children with any particular
distributional role. Try examining the ``parents`` and ``children`` attributes
of the other parameters in the model.

The following 'directed acyclic graph' is a visualization of the parent-child
relationships in the model. Unobserved stochastic variables :math:`s`, :math:`e`
and :math:`l` are open ellipses, observed stochastic variable :math:`D` is a
filled ellipse and deterministic variable :math:`r` is a triangle. Arrows point
from parent to child and display the label that the child assigns to the parent.
See section :ref:`graphical` for more details.

.. warning::

   Missing image.
   
.. figure:: DisasterModel2.pdf
   :alt: Disasters time series.
   :scale: 30
   :align: center

Variables' values and log-probabilities
=======================================

All PyMC variables have an attribute called ``value`` that stores the current
value of that variable. Try examining :math:`D`'s value, and you'll see the
initial value we provided for it::

   >>> D.value
   >>>
   array([4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6, 3, 3, 5, 4, 5, 3, 1,
          4, 4, 1, 5, 5, 3, 4, 2, 5, 2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3,
          0, 0, 1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1, 0, 1, 0, 1, 0,
          0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2, 3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2,
          0, 0, 1, 4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])

If you check :math:`e`'s, :math:`s`'s and :math:`l`'s values, you'll see random
initial values generated by PyMC::

   >>> s.value
   >>> 44

   >>> e.value
   >>> 0.33464706250079584

   >>> l.value
   >>> 2.6491936762267811

Of course, since these are ``Stochastic`` elements, your values will be
different than these. If you check :math:`r`'s value, you'll see an array whose
first :math:`s` elements are :math:`e` (here 0.33464706), and whose remaining
elements are :math:`l` (here 2.64919368)::

   >>> r.value
   >>>
   array([ 0.33464706,  0.33464706,  0.33464706,  0.33464706,  0.33464706,
           0.33464706,  0.33464706,  0.33464706,  0.33464706,  0.33464706,
           0.33464706,  0.33464706,  0.33464706,  0.33464706,  0.33464706,
           0.33464706,  0.33464706,  0.33464706,  0.33464706,  0.33464706,
           0.33464706,  0.33464706,  0.33464706,  0.33464706,  0.33464706,
           0.33464706,  0.33464706,  0.33464706,  0.33464706,  0.33464706,
           0.33464706,  0.33464706,  0.33464706,  0.33464706,  0.33464706,
           0.33464706,  0.33464706,  0.33464706,  0.33464706,  0.33464706,
           0.33464706,  0.33464706,  0.33464706,  0.33464706,  2.64919368,
           2.64919368,  2.64919368,  2.64919368,  2.64919368,  2.64919368,
           2.64919368,  2.64919368,  2.64919368,  2.64919368,  2.64919368,
           2.64919368,  2.64919368,  2.64919368,  2.64919368,  2.64919368,
           2.64919368,  2.64919368,  2.64919368,  2.64919368,  2.64919368,
           2.64919368,  2.64919368,  2.64919368,  2.64919368,  2.64919368,
           2.64919368,  2.64919368,  2.64919368,  2.64919368,  2.64919368,
           2.64919368,  2.64919368,  2.64919368,  2.64919368,  2.64919368,
           2.64919368,  2.64919368,  2.64919368,  2.64919368,  2.64919368,
           2.64919368,  2.64919368,  2.64919368,  2.64919368,  2.64919368,
           2.64919368,  2.64919368,  2.64919368,  2.64919368,  2.64919368,
           2.64919368,  2.64919368,  2.64919368,  2.64919368,  2.64919368,
           2.64919368,  2.64919368,  2.64919368,  2.64919368,  2.64919368,
           2.64919368,  2.64919368,  2.64919368,  2.64919368,  2.64919368])

To compute its value, :math:`r` calls the funtion we used to create it, passing
in the values of its parents.

``Stochastic`` objects can evaluate their probability mass or density functions
at their current values given the values of their parents. The logarithm of a
stochastic object's probability mass or density can be accessed via the ``logp``
attribute. For vector-valued variables like :math:`D`, the ``logp`` attribute
returns the sum of the logarithms of the joint probability or density of all
elements of the value. Try examining :math:`s`'s and :math:`D`'s log-
probabilities and :math:`e`'s and :math:`l`'s log-densities::

   >>> s.logp
   >>> -4.7095302013123339

   >>> D.logp
   >>> -1080.5149888046033

   >>> e.logp
   >>> -0.33464706250079584

   >>> l.logp
   >>> -2.6491936762267811

``Stochastic`` objects need to call an internal function to compute their
``logp`` attributes, as :math:`r` needed to call an internal function to compute
its value. Just as we created :math:`r` by decorating a function that computes
its value, it's possible to create custom ``Stochastic`` objects by decorating
functions that compute their log-probabilities or densities (see chapter
:ref:`chap:modelbuilding`). Users are thus not limited to the set of of
statistical distributions provided by PyMC.


Using Variables as parents of other Variables
---------------------------------------------

Let's take a closer look at our definition of :math:`r`::

   @deterministic(plot=False)
   def r(s=s, e=e, l=l):
       """ Concatenate Poisson means """
       out = np.empty(len(disasters_array))
       out[:s] = e
       out[s:] = l
       return out

The arguments :math:`s`, :math:`e` and :math:`l` are ``Stochastic`` objects, not
numbers. Why aren't errors raised when we attempt to slice array ``out`` up to a
``Stochastic`` object?

Whenever a variable is used as a parent for a child variable, PyMC replaces it
with its ``value`` attribute when the child's value or log-probability is
computed. When :math:`r`'s value is recomputed, ``s.value`` is passed to the
function as argument ``s``. To see the values of the parents of :math:`r` all
together, look at ``r.parents.value``.


Fitting the model with MCMC
===========================

PyMC provides several objects that fit probability models (linked collections of
variables) like ours. The primary such object, ``MCMC``, fits models with the
Markov chain Monte Carlo algorithm. See appendix :ref:`chap:mcmc` for an
introduction to the algorithm itself. To create an ``MCMC`` object to handle our
model, import :mod:`DisasterModel.py` and use it as an argument for ``MCMC``::

   import DisasterModel
   from pymc import MCMC
   M = MCMC(DisasterModel)

In this case ``M`` will expose variables ``s``, ``e``, ``l``, ``r`` and ``D`` as
attributes; that is, ``M.s`` will be the same object as ``DisasterModel.s``.

To run the sampler, call the MCMC object's ``isample()`` (or ``sample()``)
method with arguments for the number of iterations, burn-in length, and thinning
interval (if desired)::

   M.isample(iter=10000, burn=1000, thin=10)

After a few seconds, you should see that sampling has finished normally. The
model has been fitted.


What does it mean to fit a model?
---------------------------------

'Fitting' a model means characterizing its posterior distribution somehow. In
this case, we are trying to represent the posterior :math:`p(s,e,l|D)` by a set
of joint samples from it. To produce these samples, the MCMC sampler randomly
updates the values of :math:`s`, :math:`e` and :math:`l` according to the
Metropolis-Hastings algorithm ([Gelman et al., 2004]_) for ``iter``  iterations.

After a sufficiently large number of iterations, the current values of
:math:`s`, :math:`e` and :math:`l` can be considered a sample from the
posterior. PyMC assumes that the ``burn`` parameter specifies a 'sufficiently
large' number of iterations for convergence of the algorithm, so it is up to the
user to verify that this is the case (see chapter :ref:`chap:modelchecking`).
Consecutive values sampled from :math:`s`, :math:`e` and :math:`l` are
necessarily dependent on the previous sample, since it is a Markov chain.
However, MCMC often results in strong autocorrelation among samples that can
result in imprecise posterior inference. To circumvent this, it is often
effective to thin the sample by only retaining every :math:`k`th sample, where
:math:`k` is an integer value. This thinning interval is passed to the sampler
via the ``thin`` argument.

If you are not sure ahead of time what values to choose for the ``burn`` and
``thin`` parameters, you may want to retain all the MCMC samples, that is to set
``burn=0`` and ``thin=1``, and then discard the 'burnin period' and thin the
samples after examining the traces (the series of samples). See [Gelman et al., 2004]_ for
general guidance.


Accessing the samples
---------------------

The output of the MCMC algorithm is a 'trace', the sequence of retained samples
for each variable in the model. These traces can be accessed using the
``trace(name, chain=-1)`` method. For example::

   >>> M.trace('s')[:]
   array([41, 40, 40, ..., 43, 44, 44])

The trace slice ``[start:stop:step]`` works just like the NumPy array slice. By
default, the returned trace array contains the samples from the last call to
``sample``, that is, ``chain=-1``, but the trace from previous sampling runs can
be retrieved by specifying the correspondent chain index. To return the trace
from all chains, simply use ``chain=None``. [#]_


Sampling output
---------------

You can examine the marginal posterior of any variable by plotting a histogram
of its trace::

   >>> from pylab import hist, show
   >>> hist(M.trace('l')[:])
   >>>
   (array([   8,   52,  565, 1624, 2563, 2105, 1292,  488,  258,   45]),
    array([ 0.52721865,  0.60788251,  0.68854637,  0.76921023,  0.84987409,
           0.93053795,  1.01120181,  1.09186567,  1.17252953,  1.25319339]),
    <a list of 10 Patch objects>)
   >>> show()

You should see something like this:

PyMC has its own plotting functionality, via the optional :mod:`matplotlib`
module as noted in the installation notes. The :mod:`Matplot` module includes a
``plot`` function that takes the model (or a single parameter) as an argument::

   >>> from pymc.Matplot import plot
   >>> plot(M)

For each variable in the model, ``plot`` generates a composite figure, such as
this one for the switchpoint in the disasters model:

The left-hand pane of this figure shows the temporal series of the samples from
:math:`s`, while the right-hand pane shows a histogram of the trace. The trace
is useful for evaluating and diagnosing the algorithm's performance (see
[Gelman et al., 2004]_), while the histogram is useful for visualizing the posterior.

For a non-graphical summary of the posterior, simply call ``M.stats()``.


Imputation of Missing Data
--------------------------

As with most "textbook examples", the models we have examined so far assume that
the associated data are complete. That is, there are no missing values
corresponding to any observations in the dataset. However, many real-world
datasets contain one or more missing values, usually due to some logistical
problem during the data collection process. The easiest way of dealing with
observations that contain missing values is simply to exclude them from the
analysis. However, this results in loss of information if an excluded
observation contains valid values for other quantities. An alternative is to
impute the missing values, based on information in the rest of the model.

For example, consider a survey dataset for some wildlife species:

=====  ====  ========  ===========  
Count  Site  Observer  Temperature
=====  ====  ========  ===========
15       1       1          15
10       1       2          NA
 6       1       1          11
=====  ====  ========  ===========

Each row contains the number of individuals seen during the survey, along with
three covariates: the site on which the survey was conducted, the observer that
collected the data, and the temperature during the survey. If we are interested
in modelling, say, population size as a function of the count and the associated
covariates, it is difficult to accommodate the second observation because the
temperature is missing (perhaps the thermometer was broken that day). Ignoring
this observation will allow us to fit the model, but it wastes information that
is contained in the other covariates.

In a Bayesian modelling framework, missing data are accommodated simply by
treating them as unknown model parameters. Values for the missing data
:math:`\tilde{y}` are estimated naturally, using the posterior predictive
distribution:

   .. math::
   	p(\tilde{y}|y) = \int p(\tilde{y}|\theta) f(\theta|y) d\theta


This describes additional data :math:`\tilde{y}`, which may either be considered
unobserved data or potential future observations. We can use the posterior
predictive distribution to model the likely values of missing data.

Consider the coal mining disasters data introduced previously. Assume that two
years of data are missing from the time series; we indicate this in the data
array by the use of an arbitrary placeholder value, -999. ::

   x = numpy.array([ 4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
   3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
   2, 2, 3, 4, 2, 1, 3, -999, 2, 1, 1, 1, 1, 3, 0, 0,
   1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
   0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
   3, 3, 1, -999, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
   0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])

To estimate these values in PyMC, we generate a masked array. These are
specialised NumPy arrays that contain a matching True or False value for each
element to indicate if that value should be excluded from any computation.
Masked arrays can be generated using NumPy's ``ma.masked_equal`` function::

   >>> masked_data = numpy.ma.masked_equal(x, value=-999)
   >>> masked_data
   masked_array(data = [4 5 4 0 1 4 3 4 0 6 3 3 4 0 2 6 3 3 5 4 5 3 1 4 4 1 5 5 3
    4 2 5 2 2 3 4 2 1 3 -- 2 1 1 1 1 3 0 0 1 0 1 1 0 0 3 1 0 3 2 2 0 1 1 1 0 1 0
    1 0 0 0 2 1 0 0 0 1 1 0 2 3 3 1 -- 2 1 1 1 1 2 4 2 0 0 1 4 0 0 0 1 0 0 0 0 0 1
    0 0 1 0 1],
    mask = [False False False False False False False False False False False False
    False False False False False False False False False False False False
    False False False False False False False False False False False False
    False False False  True False False False False False False False False
    False False False False False False False False False False False False
    False False False False False False False False False False False False
    False False False False False False False False False False False  True
    False False False False False False False False False False False False
    False False False False False False False False False False False False
    False False False],
         fill_value=999999)


This masked array, in turn, can then be passed to PyMC's own ``ImputeMissing``
function, which replaces the missing values with Stochastic variables of the
desired type. For the coal mining disasters problem, recall that disaster events
were modelled as Poisson variates::

   >>> D = ImputeMissing('D', Poisson, masked_data, mu=r)
   >>> D
   [<pymc.distributions.Poisson 'D[0]' at 0x4ba42d0>,
    <pymc.distributions.Poisson 'D[1]' at 0x4ba4330>,
    <pymc.distributions.Poisson 'D[2]' at 0x4ba44d0>,
    <pymc.distributions.Poisson 'D[3]' at 0x4ba45f0>,
   ...
    <pymc.distributions.Poisson 'D[110]' at 0x4ba46d0>]

Here :math:`r` is an array of means for each year of data, allocated according
to the location of the switchpoint. Each element in :math:`D` is a Poisson
Stochastic, irrespective of whether the observation was missing or not. The
difference is that actual observations are data Stochastics (``observed=True``),
while the missing values are non-data Stochastics. The latter are considered
unknown, rather than fixed, and therefore estimated by the MCMC algorithm, just
as unknown model parameters.

The entire model looks very similar to the original model::

   # Switchpoint
   s = DiscreteUniform('s', lower=0, upper=110)
   # Early mean
   e = Exponential('e', beta=1)
   # Late mean
   l = Exponential('l', beta=1)

   @deterministic(plot=False)
   def r(s=s, e=e, l=l):
       """Allocate appropriate mean to time series"""
       out = np.empty(len(disasters_array))
       # Early mean prior to switchpoint
       out[:s] = e
       # Late mean following switchpoint
       out[s:] = l
       return out

   # Where the mask is true, the value is taken as missing.
   masked_data = np.ma.masked_array(disasters_array, disasters_mask)
   D = ImputeMissing('D', Poisson, masked_data, mu=r)

The main limitation of this approach for imputation is performance. Because each
element in the data array is modelled by an individual Stochastic, rather than a
single Stochastic for the entire array, the number of nodes in the overall model
increases from 4 to 113. This significantly slows the rate of sampling, since
the model iterates over each node at every iteration.

.. _fig:missing:

.. figure:: missing.png
   :alt: Trace and posterior distribution figure. 
   :scale: 70
   :align: center

   Trace and posterior distribution of the second missing data point in the example.


Fine-tuning the MCMC algorithm
==============================

MCMC objects handle individual variables via *step methods*, which determine how
parameters are updated at each step of the MCMC algorithm. By default, step
methods are automatically assigned to variables by PyMC. To see which step
methods :math:`M` is using, look at its ``step_method_dict`` attribute with
respect to each parameter::

   >>> M.step_method_dict[s]
   >>> [<pymc.StepMethods.DiscreteMetropolis object at 0x3e8cb50>]

   >>> M.step_method_dict[e]
   >>> [<pymc.StepMethods.Metropolis object at 0x3e8cbb0>]

   >>> M.step_method_dict[l]
   >>> [<pymc.StepMethods.Metropolis object at 0x3e8ccb0>]

The value of ``step_method_dict`` corresponding to a particular variable is a
list of the step methods :math:`M` is using to handle that variable.

You can force :math:`M` to use a particular step method by calling
``M.use_step_method`` before telling it to sample. The following call will cause
:math:`M` to handle :math:`l` with a standard ``Metropolis`` step method, but
with proposal standard deviation equal to :math:`2`::

   M.use_step_method(Metropolis, l, proposal_sd=2.)

Another step method class, ``AdaptiveMetropolis``, is better at handling highly-
correlated variables. If your model mixes poorly, using ``AdaptiveMetropolis``
is a sensible first thing to try.


Beyond the basics
=================

That was a brief introduction to basic PyMC usage. Many more topics are covered
in the subsequent sections, including:

* Class ``Potential``, another building block for probability models in addition
  to ``Stochastic`` and ``Deterministic``

* Normal approximations

* Using custom probability distributions

* Object architecture

* Saving traces to the disk, or streaming them to the disk during sampling

* Writing your own step methods and fitting algorithms.

Also, be sure to check out the documentation for the Gaussian process extension,
which is available on the webpage.

.. rubric:: Footnotes

.. [#] Note that the unknown variables :math:`s`, :math:`e`, :math:`l` and :math:`r`
   will all accrue samples, but :math:`D` will not because its value has been
   observed and is not updated. Hence :math:`D` has no trace and calling
   ``M.trace('D')[:]`` will raise an error.

