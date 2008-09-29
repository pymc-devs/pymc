~~~~~~~~
Tutorial
~~~~~~~~

.. default-role:: math

An example statistical model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consider the following dataset, which is a time series of recorded coal mining 
disasters in the UK from 1851 to 1962 (\cite{Jarrett:1979fr}).

.. image:: disasterts.pdf
   
Occurrences of disasters in the time series is thought to be derived from a 
Poisson process with a large rate parameter in the early part of the time 
series, and from one with a smaller rate in the later part. We are interested 
in locating the change point in the series, which perhaps is related to changes 
in mining safety regulations. 

We represent our conceptual model formally as a statistical model:

.. math::
   :label: disastermodel

    \begin{array}{ccc}
        (D_t | s, e, l) \sim \textup{Poisson}\left(r_t\right), & r_t=\left\{\begin{array}{lll}
            e &\text{for}& t< s\\ l &\text{ for}& t\ge s
            \end{array}\right.,&t\in[t_l,t_h]\\
        s\sim \textup{Uniform}(t_l, t_h)\\
        e\sim \textup{Exponential}(r_e)\\
        l\sim \textup{Exponential}(r_l)        
    \end{array}


The symbols have the following meanings:

 * `D_t`: The number of disasters in year `t`.
 * `r_t`: The rate parameter of the Poisson distribution of disasters in year `t`.
 * `s`:   The year in which the rate parameter changes.
 * `e`:   The rate parameter before the switchpoint `s`.
 * `l`:   The rate parameter after the switchpoint.
 * `t_l` and `t_h`: The lower and upper boundaries of time `t`.
 * `r_e` and `r_l`: Prior parameters.


Because we have defined `D` by its dependence on `s`, `e` and `l`, the latter 
three are known as the *parents* of `D` and `D` is called their *child*. 
Similarly, the parents of `s` are `t_l` and `t_h`, and `s` is the child of `t_l`
and `t_h`.


Conditionally stochastic and conditionally deterministic variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At the model-specification stage (before the data have been observed),
`D`,  `s`,  `e`, `r`  and  `l` are  all  random  variables. Under  the
Bayesian  interpretation of probability,  *random* variables  have not
necessarily   arisen  from   a  physical   random   process.  Instead,
probability distributions are used  to characterize our uncertainty in
the true parameter values. Random variables are represented in PyMC by
``Stochastic``  and  ``Deterministic`` classes [1]_



There is a difference between `r`  and the other variables: if we knew
the  values  of `r`'s  parents,  we could  compute  the  value of  `r`
exactly.  This  variable,  represented by  the  ``Deterministic``
class, is defined  by a mathematical function which  returns its value
given values  for its parents.  This nomenclature is a  bit confusing,
because  these  objects usually  represent  random  variables; if  the
parents  of `r` are  random, `r`  is random  also. A  more descriptive
(though    more   awkward)    name   for    this   class    would   be
``DeterminedByValuesOfParents``.

On the  other hand, even  if the values  of the parents  of parameters
`s`, `D`, `e` or `l` were  known, we would still be uncertain of their
values. These variables are characterized by probability distributions
that express  how plausible their  candidate values are,  given values
for  their  parents. The  ``Stochastic``  class represents  these
variables.   A    better   name    for   these   objects    might   be
``RandomEvenGivenValuesOfParents``.

We can represent model :eq:`disastermodel` in a file called 
:file:`DisasterModel.py` as follows. First, we import the PyMC and 
NumPy namespaces and enter the actual data values into an array::

	
   # Import from modules
   from pymc import Exponential, deterministic, DiscreteUniform, Poisson
   from numpy import array

   D_array =   array([ 4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                       3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                       2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
                       1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                       0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                       3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                       0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])


Next, we create the switchpoint variable `s`::

   s = DiscreteUniform('s', lower=0, upper=110)   


``DiscreteUniform`` is a subclass of ``Stochastic`` that represents uniformly-distributed discrete variables. Use of this distribution suggests that we have no information *a priori* regarding the location of the switchpoint; all values are equally likely. Now we create the exponentially-distributed variables `e` and `l`::

   e = Exponential('e', beta=1)
   l = Exponential('l', beta=1)   


Next, we define the variable `r`, which selects the early rate `e` for times before `s` and the late rate `l` for times after `s`. We create `r` using the ``deterministic`` decorator, which converts the ordinary Python function `r` into a ``Deterministic`` object.::


   @deterministic
   def r(s=s, e=e, l=l):
      # Create an empty array of size 111
      out = np.empty(111)
      # Assign e to the first s elements
      out[:s] = e
      # ... and l to the rest.
      out[s:] = l
      return out


The last step is to define the number of disasters `D`. This is done the same way as for stochastic variables, except that we set the argument ``isdata` to ``True`` (it is set to ``False`` by default). This tells PyMC that this object has a fixed value and does not need to be sampled::

   D = Poisson('D', mu=r, value=D_array, isdata=True)



.. rubric:: Footnotes

.. [1] Both ``Stochastic`` and  ``Deterministic`` are subclasses  of the generic ``Variable`` class.


.. rubric :: Why are data and unknown variables represented by the same object?


Since its represented by a ``Stochastic`` object, `D` is defined by its dependence on its parents `s`, `e` and `l` even though its value is fixed. This isn't just a quirk of PyMC's syntax; Bayesian hierarchical notation itself makes no distinction between random variables and data. The reason is simple: to use Bayes' theorem to compute the posterior `p(e,s,l|D)` of model \ref{disastermodel}, we need to use the likelihood `p(D|e,s,l)`. Even though `D`'s value is known and fixed, we need to formally assign it a probability distribution as if it were a random variable.

This point can be counterintuitive at first, as many peoples' instinct is to regard data as fixed a priori and unknown variables as dependent on the data. One way to understand this is to think of statistical models like (:eq:`disastermodel`) as predictive models for data, or as models of the processes that gave rise to data. Before observing the value of `D`, we could have sampled from its prior predictive distribution `p(D)` (*i.e.* the marginal distribution of the data) as follows:

#. Sample `e`, `s` and `l` from their priors.
#. Sample `D` conditional on these values.

Even after we observe the value of `D`, we need to use this process model to make inferences about `e`, `s` and `l`; it's the only information we have about how the variables are related.


To look at the issue another way, we could, in principle, have written a model equivalent to (:eq:`disastermodel`) such that `D` depended on nothing and everything else depended on `D`, for example

.. math::

    s|e,l,D\sim\cdot

    e|l,D\sim\cdot

    l|D\sim\cdot

    D=D_*


In one respect, this would have been more natural because we would have the unknown stochastic variables depending on the data. However, if we could write down that model using standard distributions we could trivially compute and sample from the posterior,

.. math::

    p(s,e,l|D) = p(s|e, l, D) p(e|l, D) p(l|D),

and we would have no use for MCMC or any other fitting method. Bayesian methods, and statistics in general, are needed when it's feasible to write down the data's dependence on the unknown variables but not vice versa.


Parents and children
~~~~~~~~~~~~~~~~~~~~

We have created a PyMC probability model: a linked collection of variables. To see the nature of the links, import or run ``DisasterModel.py`` and examine `s`'s ``parents`` attribute from the Python prompt\footnote{If you do not recognize this prompt, it is because we are using the IPython shell, rather than the standard shell.}::

   In [2]: s.parents
   Out[2]: {'lower': 0, 'upper': 110}

The ``parents`` dictionary shows us the distributional parameters of `s`. Now try examining `D`'s parents::

   In [3]: D.parents
   Out[3]: {'mu': <pymc.PyMCObjects.Deterministic 'r' at 0x3e51a70>}

We are using `r` as a distributional parameter of `D`, so `r` is `D`'s parent. `D` labels `r` as ``mu``, meaning it plays the role of the rate parameter in `D`'s Poisson distribution. Now examine `r`'s ``children`` attribute::

   In [3]: r.children
   Out[3]: set([<pymc.distributions.Poisson 'D' at 0x3e51290>])

Because `D` considers `r` its parent, `r` considers `D` its child. Unlike ``parents``, ``children`` is a set; variables do not associate their children with any particular distributional role. Try examining the ``parents`` and ``children`` attributes of the other parameters in the model.

Variables' values and log-probabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All PyMC variables have an attribute called ``value``. Try examining `D`'s value, and you'll see the initial value we provided for it::

   In [4]: D.value
   Out[4]: 
   array([4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6, 3, 3, 5, 4, 5, 3, 1,
          4, 4, 1, 5, 5, 3, 4, 2, 5, 2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3,
          0, 0, 1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1, 0, 1, 0, 1, 0,
          0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2, 3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2,
          0, 0, 1, 4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])


If you check `e`'s, `s`'s and `l`'s values, you'll see random initial values generated by PyMC::

   In [5]: s.value
   Out[5]: 44

   In [6]: e.value
   Out[6]: 0.33464706250079584

   In [7]: l.value
   Out[7]: 2.6491936762267811


Of course, since these are Stochastic elements, your value will be different than these. If you check `r`'s value, you'll see an array whose first `s` elements are ``e.value``, and whose remaining elements are ``l.value``::

   In [8]: r.value
   Out[8]: 
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


To compute its value, `r` calls the funtion we used to create it, passing in the values of its parents.

Stochastic objects can evaluate their probability mass or density functions at their current values given the values of their parents. The log of a stochastic object's probability mass or density can be accessed via the ``logp`` attribute. For vector-valued variables like `D`, the ``logp`` attribute returns the log of the joint probability or density of all elements of the value. Try examining `s`'s and `D`'s log-probabilities and `e`'s and `l`'s log-densities::

   In [9]: s.logp
   Out[9]: -4.7095302013123339

   In [10]: D.logp
   Out[10]: -1080.5149888046033

   In [11]: e.logp
   Out[11]: -0.33464706250079584

   In [12]: l.logp
   Out[12]: -2.6491936762267811


Stochastic objects need to call an internal function to compute their ``logp`` attributes, as `r` needed to call an internal function to compute its value. Just as we created `r` by decorating a function that computes its value, it's possible to create custom ``Stochastic`` objects by decorating functions that compute their log-probabilities or densities (see chapter \ref{chap:modelbuilding}). 

.. rubric:: Using ``Variables`` as parents of ``Variables``

Let's take a closer look at our definition of `r`::

   @deterministic
   def r(s=s, e=e, l=l):
      # Create an empty array of size 111
      out = np.empty(111)
      # Assign e to the first s elements
      out[:s] = e
      # ... and l to the rest.
      out[s:] = l
      return out


The arguments are ``Stochastic`` objects, not numbers. Why aren't errors raised when we attempt to slice array ``out`` up to a ``Stochastic`` object?

Whenever a variable is used as a parent for a child variable, PyMC replaces it with its ``value`` attribute when the child's value or log-probability is computed. When `r`'s value is recomputed, ``s.value`` is passed to the function as argument ``s``. To see the values of the parents of `r` all together, look at ``r.parents.value``.

Fitting the model with MCMC
~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyMC provides several objects that fit probability models (linked collections of variables) like ours. The primary such object, ``MCMC``, fits models with the Markov chain Monte Carlo algorithm. See chapter \ref{chap:MCMC} for an introduction to the algorithm itself. To create an ``MCMC`` object to handle our model, import \module{DisasterModel.py} and use it as an argument for ``MCMC``::

   import DisasterModel
   from pymc import MCMC
   M = MCMC(DisasterModel)


To run the sampler, call the MCMC object's ``isample()`` (or ``sample()``) method, either from \module{DisasterModel.py} or the prompt::

   M.isample(iter=10000, burn=1000)


After a few seconds, you should see that sampling has finished normally. The model has been fitted.



.. rubric:: What does it mean to fit a model?


The MCMC sampler runs for the specified number of iterations. If the run is sufficiently long, the model will have converged to the posterior distribution of interest, and all subsequent samples can be considered samples from that distribution, and used for inference. The specified ``burn`` interval should be large enough to ensure that no pre-convergent samples are included in the sample used for generating summary statistics.

The output of the MCMC algorithm is a *trace*, the sequence of retained samples for each variable in the model. These traces are stored as attributes of the variables themselves and can be accessed using the ``trace()`` method. For example::

   In [2]: s.trace()
   Out[2]: array([41, 40, 40, ..., 43, 44, 44])


The unknown variables `s`, `e`, `l` and `r` will all accrue samples, but `D` will not because its value has been observed and is not updated.



.. rubric:: Sampling output

You can examine the marginal posterior of any variable by plotting a histogram of its trace::

   In [3]: from pylab import hist
   In [4]: hist(l.trace())
   Out[4]: 
   (array([   8,   52,  565, 1624, 2563, 2105, 1292,  488,  258,   45]),
    array([ 0.52721865,  0.60788251,  0.68854637,  0.76921023,  0.84987409,
           0.93053795,  1.01120181,  1.09186567,  1.17252953,  1.25319339]),
    <a list of 10 Patch objects>)


You should see something like this:

.. image:: ltrace.pdf

PyMC has its own plotting functionality, via the optional matplotlib module as noted in the installation notes. The ``Matplot`` module includes a ``plot`` function that takes the model (or a single parameter) as an argument::

   In [5]: from pymc.Matplot import plot
   In [6]: plot(M)

You will see several figures like the following:

.. image::spost.pdf


The left-hand pane of this figure shows the temporal series of the samples from `s`, while the right-hand pane shows a histogram of the trace. The trace is useful for evaluating and diagnosing the algorithm's performance [\textbf{ref}]. If the trace looks good, the right-hand pane is useful for visualizing the posterior. The posterior of `s` seems to be bimodal, which is interesting.

For a non-graphical summary of the posterior, simply call ``M.stats()``.


Fine-tuning the MCMC algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MCMC objects handle individual variables via 'step methods', which determine how parameters are updated at each step of the MCMC algorithm. By default, step methods are automatically assigned to variables by PyMC. To see which step methods `M` is using, look at its ``step_method_dict`` attribute with respect to each parameter::

   In [5]: M.step_method_dict[s]
   Out[5]: [<pymc.StepMethods.DiscreteMetropolis object at 0x3e8cb50>]
   
   In [8]: M.step_method_dict[e]
   Out[8]: [<pymc.StepMethods.Metropolis object at 0x3e8cbb0>]

   In [9]: M.step_method_dict[l]
   Out[9]: [<pymc.StepMethods.Metropolis object at 0x3e8ccb0>]

The value of ``step_method_dict`` corresponding to a particular variable is a list of the step methods `M` is using to handle that variable. 

You can force `M` to use a particular step method by calling ``M.use_step_method`` before telling it to sample. The following call will cause `M` to handle `l` with a standard ``Metropolis`` step method, but with proposal standard deviation equal to `2`::

   M.use_step_method(Metropolis, l, sig=2.)


Another step method class, ``AdaptiveMetropolis``, is better at handling highly-correlated variables. If your model mixes poorly, using ``AdaptiveMetropolis`` is a sensible first thing to try.

You can see all the step method classes that have been defined (including user-defined step methods) in the list ``StepMethodRegistry``, which is on the PyMC namespace::

   In [12]: pymc.StepMethodRegistry
   Out[12]: 
   [<class 'pymc.StepMethods.StepMethod'>,
    <class 'pymc.StepMethods.NoStepper'>,
    <class 'pymc.StepMethods.Metropolis'>,
    <class 'pymc.StepMethods.Gibbs'>,
    <class 'pymc.StepMethods.NoStepper'>,
    <class 'pymc.StepMethods.DiscreteMetropolis'>,
    <class 'pymc.StepMethods.BinaryMetropolis'>,
    <class 'pymc.StepMethods.AdaptiveMetropolis'>,
    <class 'pymc.StepMethods.IIDSStepper'>,
    <class 'pymc.GP.PyMC_objects.GPParentMetropolis'>,
    <class 'pymc.GP.PyMC_objects.GPMetropolis'>,
    <class 'pymc.GP.PyMC_objects.GPNormal'>]

See the docstrings of the individual classes for details on how to use them.

Beyond the basics
~~~~~~~~~~~~~~~~~

That's all there is to basic PyMC usage. Many more topics are covered in the reference manual (all chapters after \ref{chap:MCMC}), including:

#. Class ``Potential``, another building block for probability models in addition to ``Stochastic`` and ``Deterministic``
#. Normal approximations
#. How to use custom probability distributions
#. The inner workings of the objects
#. How to save traces to the disk, or stream them to the disk during sampling
#. How to write your own step methods and fitting algorithms.

Also, be sure to check out the documentation for the Gaussian process extension, located in folder ``gp`` in the source directory. 


MCMC is a surprisingly difficult and bug-prone algorithm to implement by hand. We find PyMC makes it much easier and less stressful. PyMC also makes our work more dynamic; getting hand-coded MCMC's working used to be so much work that we were reluctant to change anything, but with PyMC changing models is a breeze. We hope it does the same for you!
