~~~~~~~~~~~~~~
Extending pymc
~~~~~~~~~~~~~~

PyMC is designed to make standard things easy, but keep weird things possible. Its openness, combined with Python's flexibility, invite extensions from using new step methods to ``Stochastics`` valued as exotic stochastic processes (see the Gaussian process module). This chapter briefly reviews the primary ways PyMC can be extended without actually reading its source code. 

\hypertarget{nonstandard}{}
Nonstandard Stochastics} \label{nonstandard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

\pdfbookmark[0]{Nonstandard Stochastics}{nonstandard}

The simplest way to create a ``Stochastic`` object with a nonstandard distribution is to simply use the medium or long decorator syntax. See chapter \ref{chap:modelbuilding}. If you want to create many stochastics with the same nonstandard distribution, the decorator syntax can become cumbersome. An actual subclass of ``Stochastic`` can be created using the class factory ``stochastic_from_distribution``. This function takes the following arguments:
\begin{itemize}
   \item The name of the new class,
   \item A ``logp`` function,
   \item A ``random`` function,
   \item The dtype of the new class,
   \item A flag indicating whether the resulting class represents a vector-valued variable.
\end{itemize}
The necessary parent labels are read from the ``logp`` function, and a docstring for the new class is automatically generated. Instances of the new class can be created in one line.

Full subclasses of ``Stochastic`` may be necessary to provide nonstandard behaviors (see ``gp.GP``).

\hypertarget{custom-stepper}{}
User-defined step methods} \label{custom-stepper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

\pdfbookmark[0]{User-defined step methods}{custom-stepper}
The ``StepMethod`` class is meant to be subclassed. There are an enormous number of MCMC step methods in the literature, whereas PyMC provides only about a dozen. Most user-defined step methods will be either Metropolis-Hastings or Gibbs step methods, and these should subclass ``Metropolis`` or ``Gibbs`` respectively. More unusual step methods should subclass ``StepMethod`` directly.

\hypertarget{user-gen}{}
.. rubric:: General step methods} \label{user-gen


\pdfbookmark[1]{General step methods}{user-gen}

All step methods must implement the following methods:
\begin{description}
   \item[``step()``:] Updates the values of ``self.stochastics``.
   \item[``tune()``:] Tunes the jumping strategy based on performance so far. A default method is available that increases ``self._asf`` (see below) when acceptance rate is high, and decreases it when acceptance rate is low. This method should return ``True`` if additional tuning will be required later, and ``False`` otherwise.
   \item[``competence(s):``] A static method that examines stochastic variable `s` and returns a value from 0 to 3 expressing the step method's ability to handle the variable. This method is used by ``MCMC`` instances when automatically assigning step methods. Conventions are:
   \begin{description}
      \item[0] I cannot safely handle this variable. 
      \item[1] I can handle the variable about as well as the standard ``Metropolis`` step method.
      \item[2] I can do better.
      \item[3] I am the best step method you are likely to find for this variable in most cases.
   \end{description}
   \item[``current_state()``:] This method is easiest to explain by showing the code:
   \begin{verbatim}
state = {}
for s in self._state:
    state[s] = getattr(self, s)
return state      
   \end{verbatim}
   ``self._state`` should be a list containing the names of the attributes needed to reproduce the current jumping strategy. If an ``MCMC`` object writes its state out to a database, these attributes will be preserved. If an ``MCMC`` object restores its state from the database later, the corresponding step method will have these attributes set to their saved values.
\end{description}

Step methods should also maintain the following attributes:
\begin{description}
   \item[``_id``:] A string that can identify each step method uniquely (usually something like ``<class_name>_<stochastic_name>``).
   \item[``_accepted``:] A running tally of the number of jumps accepted.
   \item[``_rejected``:] A running tally of the number of jumps rejected.   
   \item[``_asf``:] An `adaptive scale factor'. This attribute is only needed if the default ``tune()`` method is used.
\end{description}

All step methods have a property called ``loglike``, which returns the sum of the log-probabilities of the union of the extended children of ``self.stochastics``. This quantity is one term in the log of the Metropolis-Hastings acceptance ratio.


\hypertarget{user-metro}{}
.. rubric:: Metropolis-Hastings step methods} \label{user-metro


\pdfbookmark[1]{Metropolis-Hastings step methods}{user-metro}
A Metropolis-Hastings step method must implement the following methods, which are called by ``step()``:
\begin{description}

   \item[``reject()``:] Usually just
   \begin{verbatim}
[s.value = s.last_value for s in self.stochastics]
   \end{verbatim}
   \item[``propose():``] Sets the values of all ``self.stochastics`` to new, proposed values.
\end{description}
Metropolis-Hastings step methods with asymmetric jumping distributions may implement a method called ``hastings_factor()``, which returns the log of the ratio of the `reverse' and `forward' proposal probabilities. Note that no ``accept()`` method is needed or used.

\hypertarget{user-gibbs}{}
.. rubric:: Gibbs step methods} \label{user-gibbs


\pdfbookmark[1]{Gibbs step methods}{user-gibbs}

Gibbs step methods handle conjugate submodels. These models usually have two components: the `parent' and the `children'. For example, a gamma-distributed variable serving as the precision of several normally-distributed variables is a conjugate submodel; the gamma variable is the parent and the normal variables are the children. 

This section describes PyMC's current scheme for Gibbs step methods. It is meant to be as generic as possible to minimize code duplication, but it is admittedly complicated. Feel free to subclass StepMethod directly when writing Gibbs step methods if you prefer.

Gibbs step methods that subclass PyMC's ``StandardGibbs`` should define the following class attributes:
\begin{description}
   \item[``child_class``:] The class of the children in the submodels the step method can handle.
   \item[``parent_class``:] The class of the parent.
   \item[``parent_label``:] The label the children would apply to the parent in a conjugate submodel. In the gamma-normal example, this would be ``tau``.
   \item[``linear_OK``:] A flag indicating whether the children can use linear combinations involving the parent as their actual parent without destroying the conjugacy.
\end{description}

A subclass of ``StandardGibbs`` that defines these attributes need only implement a ``propose()`` method. The resulting step method will be able to handle both conjugate and non-conjugate cases. The conjugate case corresponds to an actual conjugate submodel. In the nonconjugate case all the children are of the required class, but the parent is not. In this case the parent's value is proposed from the likelihood and accepted based on its prior. The acceptance rate in the nonconjugate case will be less than one.

\hypertarget{custom-model}{}
New fitting algorithms} \label{custom-model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

\pdfbookmark[0]{New fitting algorithms}{custom-model}

PyMC provides a convenient platform for non-MCMC fitting algorithms in addition to MCMC. The primary advantage of implementing fitting algorithms in PyMC is that the same model can be fit with the new algorithm or one of the existing algorithms with virtually no re-coding.

All fitting algorithms should be implemented by subclasses of ``Model``. There are virtually no restrictions on fitting algorithms, but many of ``Model``'s behaviors may be useful. See chapter \ref{chap:modelfitting}. 

\hypertarget{custom-MC}{}
.. rubric:: Monte Carlo fitting algoriths} \label{custom-MC


\pdfbookmark[1]{Monte Carlo fitting algoriths}{custom-MC}

Unless there is a good reason to do otherwise, Monte Carlo fitting algorithms should be implemented by subclasses of ``Sampler`` to take advantage of the interactive sampling feature and database backends. Subclasses using the standard ``sample()`` and ``isample()`` methods must define one of two methods:
\begin{description}
   \item[``draw()``:] If it is possible to generate an independent sample from the posterior at every iteration, the ``draw`` method should do so. The default ``_loop`` method can be used in this case.
   \item[``_loop()``:] If it is not possible to implement a ``draw()`` method, but you want to take advantage of the interactive sampling option, you should override ``_loop()``. This method is responsible for generating the posterior samples and calling ``tally()`` when it is appropriate to save the model's state. In addition, ``_loop`` should monitor the sampler's ``status`` attribute at every iteration and respond appropriately. The possible values of ``status`` are:
   \begin{description}
      \item[``'ready'``:] Ready to sample.
      \item[``'running'``:] Sampling should continue as normal.
      \item[``'halt'``:] Sampling should halt as soon as possible. ``_loop`` should call the ``halt()`` method and return control. ``_loop`` can set the status to ``'halt'`` itself if appropriate (eg the database is full or a ``KeyboardInterrupt`` has been caught).
      \item[``'paused'``:] Sampling should pause as soon as possible. ``_loop`` should return, but should be able to pick up where it left off next time it's called.
   \end{description}
\end{description}

Samplers may alternatively want to override the default ``sample()`` method. In that case, they should call the ``tally()`` method whenever it is appropriate to save the current model state. Like custom ``_loop()`` methods, custom ``sample()`` methods should handle ``KeyboardInterrupts`` and call the ``halt()`` method when sampling terminates to finalize the traces.

\hypertarget{dont-update-indepth}{}
Don't update stochastic variables' values in-place} \label{dont-update-indepth
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

\pdfbookmark[0]{Don't update stochastic variables' values in-place}{dont-update-indepth}

If you're going to implement a new step method, fitting algorithm or exotic ``Stochastic`` subclass, you should understand the issues related to in-place updates of ``Stochastic`` objects' values. Fitting methods should never update variables' values in-place for two reasons:
\begin{itemize}
   \item In algorithms that involve accepting and rejecting proposals, the `pre-proposal' value needs to be preserved uncorrupted. It would be possible to make a copy of the pre-proposal value and then allow in-place updates, but in PyMC we have chosen to store the pre-proposal value as ``Stochastic.last_value`` and require proposed values to be new objects. In-place updates would corrupt ``Stochastic.last_value``, and this would cause problems.
   \item ``LazyFunction``'s caching scheme checks current values against its internal cache by reference.
\end{itemize}

However, a ``Stochastic`` object's value can make in-place updates to itself if the updates don't change its identity. For example, the ``Stochastic`` subclass ``gp.GP`` is valued as a ``gp.Realization`` object. GP realizations represent random functions, which are infinite-dimensional stochastic processes, as literally as possible. The strategy they employ is to `self-discover' on demand: when they are evaluated, they generate the required value conditional on previous evaluations and then make an internal note of it. This is an in-place update, but it is done to provide the same behavior as a single random function whose value everywhere has been determined since it was created.
