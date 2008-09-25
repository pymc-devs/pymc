PyMC probability models are linked collections of nodes. These nodes are only informed by the value of their parents. ``Deterministic`` instances can compute their values given their parents' values, ``Stochastic`` instances can compute their log-probabilities or draw new values, and ``Potential`` instances can compute their log-probabilities. Fitting probability models requires larger-scale coordination and communication.

All objects capable of fitting probability models are subclasses of the ``Model`` class. All objects that fit probability models using some kind of Monte Carlo method are descended from the ``Model`` subclass ``Sampler``. ``Sampler`` provides a generic sampling loop method and database support for storing large sets of joint samples. %Sampling loops can optionally be run interactively, meaning the user can pause sampling at any time, return to the Python prompt, check progress, and make adjustments.

PyMC provides three Sampler subclasses for fitting models:
\begin{itemize}
    \item ``MCMC``, which coordinates Markov chain Monte Carlo algorithms. The actual work of updating stochastic variables conditional on the rest of the model is done by ``StepMethod`` instances, which are described in this chapter.
    \item ``MAP``, which computes maximum *a posteriori* estimates.
    \item ``NormApprox``, which computes the `normal approximation' \cite{gelman}: the joint distribution of all stochastic variables in a model is approximated as normal using local information at the maximum *a posteriori* estimate.
\end{itemize}

\hypertarget{model}{}
The ``Model`` class} \label{sec:Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

\pdfbookmark[1]{The Model class}{model}
This class serves as a container for probability models and as a base class for the classes responsible for model fitting, such as ``MCMC``.

Models' useful methods are:
\begin{description}
    \item[``draw_from_prior()``:] Sets all stochastic variables' values to new random values, which would be a sample from the joint distribution if all data and ``Potential`` instances' log-probability attributes were set to zero.
    \item[``seed()``:] Same as ``draw_from_prior``, but only stochastic variables with an ``rseed`` attribute are changed.
    \item[``find_generations():``] Sets the ``generations`` attribute. This attribute is a list whose elements are sets of stochastic variables. The zeroth set has no extended parents in the model, the first set only has extended parents in the zeroth set, and so on.
\end{description}

The helper functions ``weight`` and ``graph`` act on models. ``weight`` computes Bayes' factors (posterior probabilities of model correctness) for lists of models using the ``draw_from_prior`` method, and ``graph`` produces graphical representations. The ``weight`` function's algorithm can only be expected to perform well when the dimension of the parameter space is small (less than about 10).

Models have the following important attributes:
\begin{itemize}
    \item ``variables``
    \item ``stochastics``
    \item ``potentials``
    \item ``deterministics``
    \item ``data_stochastics``
    \item ``step_methods``
    \item ``value``
\end{itemize}

In addition, models expose each node they contain as an attribute. For instance, if model ``M`` were produced from model (\ref{disastermodel}) ``M.s`` would return the switchpoint variable. It's a good idea to give each variable a unique name if you want to access them this way.


.. rubric:: Creation of models} \label{sec:ModelInstantiation


The ``Model`` class's init method takes the following arguments:
\begin{description}
    \item[``input``:] Some collection of PyMC nodes defining a probability model. These may be stored in a list, set, tuple, dictionary, array, module, or any object with a ``__dict__`` attribute. If ``input`` is ``None`` (the default), all the nodes on the main namespace and the ``Model`` object's class's dictionary are collected.
    \item[``output_path`` (optional):] A string indicating where all of the files produced by the model should be saved (defaults to current directory).
    \item[``verbose`` (optional):] An integer controlling the verbosity of the model's output.
\end{description}
The ``input`` argument can be just about anything; once you have defined the nodes that make up your model, you shouldn't even have to think about how to wrap them in a ``Model`` instance. Some examples of model instantiation, using nodes ``a``, ``b`` and ``c``:
\begin{itemize}
    %\item ``M = Model(a,b,c)`` THIS ONE DOES NOT WORK -- JUST TRIED IT
    \item ``M = Model(set([a,b,c]))``
    \item ``M = Model(\{`a': a, `d': [b,c]\``)}
    \item ``M = Model([[a,b],c])``
    \item File ``MyModule`` containing the definitions of ``a``, ``b`` and ``c``:\begin{verbatim}
import MyModule
M = Model(MyModule)
    \end{verbatim}
    \item `Model factory' function:
    \begin{verbatim}
def make_model(x):
    a = Exponential('a',.5,beta=x)
    
    @deterministic
    def b(a=a):
        return 100-a
    
    @stochastic
    def c(value=.5, a=a, b=b);
        return (value-a)**2/b
        
    return locals()
    
M = Model(make_model(3))
    \end{verbatim}
    \item Model subclasses are inspected for nodes:
    \begin{verbatim}
class MyModel(Model):
    a = Exponential('a',.5,beta=x)

    @deterministic
    def b(a=a):
        return 100-a

    @stochastic
    def c(value=.5, a=a, b=b);
        return (value-a)**2/b        
        
M = MyModel()
    \end{verbatim}    
    \item If no input argument is provided, the main namespace is inspected for nodes:
    \begin{verbatim}
    a = Exponential('a',.5,beta=x)

    @deterministic
    def b(a=a):
        return 100-a

    @stochastic
    def c(value=.5, a=a, b=b);
        return (value-a)**2/b        
    
    M = Model()
    \end{verbatim}

\end{itemize}

\hypertarget{sampler}{}
The ``Sampler`` class} \label{sec:Sampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

\pdfbookmark[1]{The Sampler class}{sampler}
Samplers fit models with Monte Carlo fitting methods, which characterize the posterior distribution by approximate samples from it. They are initialized as follows: ``Sampler(input, db=`ram', output\_path=None, verbose=0)``. The ``db`` argument indicates which database backend should be used to store the samples (see chapter \ref{chap:database}), and the other three arguments are the same as for ``Model``. Samplers have the following important methods:
\begin{description}
    \item[``sample(iter, length=None, verbose=0)``:] Samples from the joint distribution. The ``iter`` argument controls how many times the sampling loop will be run, and the ``length`` argument controls the initial size of the database that will be used to store the samples.
    \item[``isample(iter, length=None, verbose=0)``:] The same as ``sample``, but the sampling is done interactively: you can pause sampling at any point and be returned to the Python prompt to inspect progress and adjust fitting parameters. While sampling is paused, the following methods are useful: 
    \begin{description}
        \item[``icontinue()``:] Continue interactive sampling.
        \item[``halt()``:] Truncate the database and clean up.
    \end{description}
    \item[``tally()``:] Write all variables' current values to the database.
    %\item[``draw()``:] Not currently used. In future Monte Carlo fitting methods that aren't MCMC, such as importance samplers, the ``draw()`` method will be responsible for drawing approximate samples from the joint distribution (by setting the values of all the stochastic variables in the model).
    \item[``save\_state()``:] Saves the current state of the sampler, including all stochastics, to the database. This allows the sampler to be reconstituted at a later time to resume sampling.
    \item[``restore\_state()``:] Restores the sampler to the state stored in the database.
	 \item[``stats()``:] Generate summary statistics for all nodes in the model.
    \item[``remember(trace\_index)``:] Set all variables' values from frame ``trace\_index`` in the database.
\end{description}

In addition, the sampler attribute ``deviance`` is a deterministic variable valued as the model's deviance at its current state.

\hypertarget{MAP}{}
Maximum a posteriori estimates} \label{sec:MAP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

\pdfbookmark[1]{Maximum a posteriori estimates}{model}

The ``MAP`` class sets all stochastic variables to their maximum a posteriori values using functions in SciPy's ``optimize`` package. SciPy must be installed to use it. A ``MAP`` instance ``M`` can be created as follows:
\begin{verbatim}
M = MAP(input, eps=.001, diff_order = 5)    
\end{verbatim}
The parameters ``eps`` and ``diff_order`` control numerical differentiation. ``diff_order``, which must be an integer, specifies the order of the numerical approximation (see the SciPy function ``derivative``). The step size for numerical derivatives is controlled by ``eps``, which may be either a single value or a dictionary of values whose keys are variables (actual objects, not names). ``MAP`` requires all stochastic variables in ``input`` to be either float-valued or array-valued with dtype float, unlike PyMC in general.

``MAP`` has two useful methods:
\begin{description}
    \item[``fit(method ='fmin', iterlim=1000, tol=.0001)``:] The optimization method may be ``fmin``, ``fmin_l_bfgs_b``, ``fmin_ncg``, ``fmin_cg``, or ``fmin_powell``. See the documentation of SciPy's optimize package for the details of these methods. The ``tol`` and ``iterlim`` parameters are passed to the optimization function under the appropriate names.
    \item[``revert_to_max()``:] If the values of the constituent stochastic variables change after fitting, this function will reset them to their maximum a posteriori values.
\end{description}

The useful attributes of ``MAP`` are:
\begin{description}
    \item[``logp``:] The joint log-probability of the model.
    \item[``logp_at_max``:] The maximum joint log-probability of the model.
    \item[``len``:] The total number of elements in all the stochastic variables in the model with ``isdata=False``.
    \item[``data_len``:] The total number number of elements in all the stochastic variables in the model with ``isdata=True``.
    \item[``AIC``:] Akaike's information criterion for this model \cite{Akaike:1973aj,Burnham:2002ic}.
    \item[``BIC``:] The Bayesian information criterion for this model \cite{Schwarz:1978ud}.
\end{description}

One use of the ``MAP`` class is finding reasonable initial states for MCMC chains. Note that multiple ``Model`` subclasses can handle the same collection of nodes.

\hypertarget{norm-approx}{}
Normal approximations} \label{sec:norm-approx
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

\pdfbookmark[1]{Normal approximations}{norm-approx}

The ``NormApprox`` class extends the ``MAP`` class by approximating the posterior covariance of the model using the Fisher information matrix, or the Hessian of the joint log probability at the maximum. In addition to the methods and attributes of ``MAP``, it provides the following methods inherited from ``Sampler``:
\begin{description}
    \item[``sample(iter)``:] Samples from the approximate posterior distribution are drawn and stored.
    \item[``isample(iter)``:] An `interactive' version of ``sample()``: sampling can be paused, returning control to the user.
\end{description}
It provides the following additional attributes:
\begin{description}
    \item[mu:] A special dictionary-like object that can be keyed with multiple variables. ``N.mu[p1, p2, p3]`` would return the approximate posterior mean values of stochastic variables ``p1``, ``p2`` and ``p3``, ravelled and concatenated to form a vector.
    \item[C:] Another special dictionary-like object. ``N.C[p1, p2, p3]`` would return the approximate posterior covariance matrix of stochastic variables ``p1``, ``p2`` and ``p3``. As with ``mu``, these variables' values are ravelled and concatenated before their covariance matrix is constructed.
\end{description}

\hypertarget{mcmc}{}
Markov chain Monte Carlo: the ``MCMC`` class} \label{sec:mcmc
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

\pdfbookmark[1]{The MCMC class}{mcmc}
 ``MCMC`` is a subclass of ``Sampler``. At the beginning of a sampling loop, it assigns a ``StepMethod`` instance (section \ref{sec:stepmethod}) to each stochastic variable for which the user has not created one. Step methods are assigned as follows: each ``StepMethod`` subclass in existence is allowed to inspect the variable in question and determine its competence to handle the variable, on a scale of 0 to 3. An instance of the highest bidder is created to handle the variable.

MCMC samplers have the following methods, in addition to those of ``Sampler``:
\begin{description}
    \item[``sample(iter, burn=0, thin=1, tune\_interval=1000, verbose=0)``:] The ``iter`` argument controls the total number of MCMC iterations. No tallying will be done during the first ``burn`` iterations; these samples will be forgotten. After this burn-in period, tallying will be done each ``thin`` iterations. Tuning will be done each ``tune\_interval`` iterations, even after burn-in is complete \cite{tuning,Haario:2001lr}.
    \item[``isample(iter, burn=0, thin=1, tune\_interval=1000, verbose=0)``:] Interactive sampling; see ``Sampler.isample``.
    \item[``use_step_method(method, *args, **kwargs)``:] Creates an instance of step method class ``method`` to handle some stochastic variables. The extra arguments are passed to the init method.
    \item[``assign_step_methods()``:] Assigns step methods now. This method is called whenever ``sample`` or ``isample`` is called, but it can be useful to call it directly to see what the default step methods will be.
    \item[``tune()``:] Each step method's ``tune`` method is called. This method is called periodically throughout the sampling loop.
    \item[``goodness()``:] Calculates goodness-of-fit (GOF) statistics according to \cite{Brooks:2000il}.
\end{description}

MCMC samplers' step methods can be accessed via the ``\textbf{step_method_dict``} attribute. ``M.step_method_dict[x]`` returns a list of the step methods ``M`` will use to handle the stochastic variable ``x``.


\hypertarget{step-method}{}
Step methods} \label{sec:stepmethod
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

\pdfbookmark[0]{Step methods}{step-method}


Step method objects handle individual stochastic variables, or sometimes groups of them. They are responsible for making the variables they handle take single MCMC steps conditional on the rest of the model. Each subclass of ``StepMethod`` implements a method called ``step()``, which does this. Step methods with adaptive tuning parameters can optionally implement a method called ``tune()``, which causes them to assess performance so far and adjust.

The major subclasses of ``StepMethod`` are ``Metropolis`` and ``Gibbs``. PyMC provides several flavors of the basic Metropolis steps, but the Gibbs steps are in the sandbox as of the current release. However, because it is feasible to write Gibbs step methods for particular applications, the ``Gibbs`` class will be documented here.

\hypertarget{metropolis}{}
.. rubric:: Metropolis step methods} \label{metropolis


\pdfbookmark[1]{Metropolis step methods}{metropolis}

``Metropolis`` and subclasses implement Metropolis-Hastings steps. ``Metropolis`` itself handles float-valued variables, and subclasses ``DiscreteMetropolis`` and ``BinaryMetropolis`` handle integer- and boolean-valued variables, respectively. Subclasses of ``Metropolis`` must implement the following methods:
\begin{description}
    \item[``propose()``:] Sets the values of the stochastic variables handled by the step method to new values.
    \item[``reject()``:] If the Metropolis acceptance test fails, reset the values of the stochastic variables to their values before ``propose()`` was called.
\end{description}
Note that there is no ``accept()`` method; if a proposal is accepted, the variables' values are simply left alone. Subclasses that use proposal distributions other than symmetric random-walk may specify the `Hastings factor' by changing the \textbf{``hastings_factor``} method.

Metropolis step methods have the following useful attributes:
\begin{description}
    \item[``dist``:] A string indicating which distribution should be used for proposals. Current options are ``'Normal'`` and ``'Prior'``.
    \item[``proposal\_sig``:] Proportional to the standard deviation of the proposal distribution (if it is ``'Normal'``).
    \item[``\_asf``:] The `adaptive scale factor'. When ``tune()`` is called, the acceptance ratio of the step method is examined and this scale factor is updated accordingly. If the proposal distribution is normal, proposals will have standard deviation ``self.proposal\_sig * self.\_asf``. It is usually OK to keep tuning throughout the MCMC loop even though the resulting chain is not actually Markov \cite{tuning}. % This attribute is hidden, and should not be altered manually by the user. S'OK if they alter it, I do it sometimes.
\end{description}

Metropolis step methods can be created as follows:
\begin{verbatim}
M = Metropolis(stochastic, scale=1., sig=None, dist=None, verbose=0)
\end{verbatim}
The ``scale`` and ``sig`` arguments determine ``proposal\_sig``. If ``sig`` is provided, ``proposal\_sig`` is set to ``sig``. Otherwise ``sig`` is computed from ``scale`` as follows:
\begin{verbatim}
if all(self.stochastic.value != 0.):
    self.proposal_sig = ones(shape(self.stochastic.value)) * abs(self.stochastic.value) 
* scale
else:
    self.proposal_sig = ones(shape(self.stochastic.value)) * scale
\end{verbatim}

The ``dist`` argument specifies the proposal distribution and may be either of the following strings:
\begin{itemize}
    \item ``"Normal"``: A random-walk normal proposal distribution is used.
    \item ``"Prior"``: The variable's value is proposed from its prior using its ``random`` method, if possible.
\end{itemize}
If ``dist=None``, the proposal distribution is chosen automatically.

\subsubsection{The ``DiscreteMetropolis`` class}
This class is just like ``Metropolis``, but specialized to handle ``Stochastic`` instances with dtype ``int``.

\subsubsection{The ``BinaryMetropolis`` class} 
This class is specialized to handle ``Stochastic`` instances with dtype ``bool``, which are Bernoulli random variables conditional on their parents. 

For scalar-valued variables, ``BinaryMetropolis`` behaves like a Gibbs sampler, since this requires no additional expense. The ``p_jump`` and ``_asf`` parameters are not used in this case.

For array-valued variables, ``BinaryMetropolis`` can be set to propose from the prior by passing in ``dist="Prior"``. Otherwise, the argument ``p_jump`` of the init method specifies how probable a change is when proposing a new value for array-valued variables. Like ``Metropolis``' attribute ``proposal_sig``, ``p_jump`` is tuned throughout the sampling loop via ``_asf``.

\subsubsection{The ``AdaptiveMetropolis`` class} 
The ``AdaptativeMetropolis`` (AM) sampling algorithm works like a regular Metropolis step method, with the exception that stochastic parameters are block-updated using a multivariate jump distribution whose covariance is tuned during sampling. Although the chain is non-Markovian, it has correct ergodic properties (see \cite{Haario:2001lr}).

``AdaptativeMetropolis``' init method takes the following arguments:
cov=None, delay=1000, scales=None, interval=200, greedy=True,verbose=0)
\begin{description}
   \item[``stochastics``:] The stochastic variables to handle. These will be updated jointly.
   \item[``cov`` (optional):] An initial covariance matrix.
   \item[``delay`` (optional):] The number of iterations to delay before computing the empirical covariance matrix.
   \item[``scales`` (optional):] The initial covariance matrix will be diagonal, and its diagonal elements will be set to ``scales`` times the stochastics' values, squared.
   \item[``interval`` (optional):] The number of iterations between updates of the covariance matrix.
   \item[``greedy`` (optional):] If ``True``, only accepted jumps will be counted toward the delay before the covariance is first computed.
   \item[``verbose``:] An integer from 0 to 3 controlling the verbosity of the step method.   
\end{description}
 
\hypertarget{gibbs}{}
.. rubric:: Gibbs step methods} \label{gibbs


\pdfbookmark[1]{Gibbs step methods}{gibbs}

Conjugate submodels (see \href{http://en.wikipedia.org/wiki/Conjugate_prior}{http://en.wikipedia.org/wiki/Conjugate_prior} ) can be handled by Gibbs step methods rather than the default Metropolis methods. Gibbs step methods are Metropolis methods whose acceptance rate is always 1. They can be convenient because they relieve the user from having to worry about tuning the acceptance rate, but they can be computationally expensive. When variables are highly dependent on one another, better mixing can often be obtained by using ``AdaptiveMetropolis`` even when Gibbs step methods are available.

Alpha versions of Gibbs step methods handling the following conjugate submodels are available in the ``sandbox`` module:
\begin{itemize}
    \item Gamma-Gamma
    \item Gamma-Exponential
    \item Gamma-Poisson
    \item Gamma-Normal
    \item Beta-Geometric
    \item Beta-Binomial
    \item Wishart-Multivariate Normal (represented by the ``MvNormal`` class, which is parameterized by precision)
    \item Dirichlet-Multinomial.
    \item Normal-Normal (or Normal-MvNormal, etc.) (requires ``cvxopt``, \href{http://abel.ee.ucla.edu/cvxopt}{http://abel.ee.ucla.edu/cvxopt} )
\end{itemize}

Gibbs step methods have the following class attributes:
\begin{itemize}
    \item ``child_class``: The step method can handle variables whose children are all of this class. ``GammaNormal.child_class`` is ``Normal``, for example.
    \item ``parent_label``: The target variable's children must refer to it by this label. ``GammaNormal.parent_label`` is ``'mu'``.
    \item ``target_class``: The target variable should be of this class for the submodel to be fully conjugate. ``GammaNormal.target_class`` is ``Gamma``.
    \item ``linear_OK``: A flag indicating whether the variable's children can depend on a multiple of the variable. Such multiples must be implemented via the ``Deterministic`` subclass ``LinearCombination``.
\end{itemize}

A Gibbs step method can handle variables that are not of their target class, as long as all their children are of the appropriate class. If this is the case, the step method's ``conjugate`` attribute will be set to ``False`` and its acceptance rate will no longer be 1.

Gibbs step methods can are easy to use manually. To tell an ``MCMC`` object `M` to handle a variable `x` using the ``GammaNormal`` class, simply use the call
\begin{verbatim}
    M.use_step_method(GammaNormal, x)
\end{verbatim}

To indicate a general preference for Gibbs step methods vs. Metropolis step methods, set the following global integer values:
\begin{itemize}
    \item ``pymc.conjugate_Gibbs_competence``: Applicable Gibbs step methods' competence functions will return this value for variables that are not of their target classes. The default value is 0, meaning that these methods will never be assigned automatically. Set this value to 3 to ensure that Gibbs step methods are always be assigned to conjugate submodels, or to 1.5 to set their priorities between those of ``Metropolis`` and ``AdaptiveMetropolis``.
    \item ``pymc.nonconjugate_Gibbs_competence``: Applicable Gibbs step methods' competence functions will return this value for variables that are of their target classes. The default value is 0, meaning that these methods are never assigned automatically.
\end{itemize}


.. rubric:: Granularity of step methods: one-at-a-time vs. block updating

 
There is currently no way for a stochastic variable to cache individual terms of its log-probability; when this is recomputed, it is recomputed from scratch. This means that updating the elements of a array-valued variable individually is inefficient, so all existing step methods update array-valued variables together, in a block update.

To update an array-valued variable's elements individually, simply break it up into an array of scalar-valued variables. Instead of this:
\begin{verbatim}
A = Normal('A', value = zeros(100), mu=0., tau=1.)    
\end{verbatim}
do this:
\begin{verbatim}
A = [Normal('A_%i'%i, 0., mu=0., tau=1.) for i in xrange(100)]
\end{verbatim}
An individual step method will be assigned to each element of ``A`` in the latter case, and the elements will be updated individually. Note that ``A`` can be broken up into larger blocks if desired.

.. rubric:: Automatic assignment of step methods

 
Every step method subclass (including user-defined ones) adds itself to a list called ``StepMethodRegistry`` in the PyMC namespace. If you create a step method is created by the user to handle a stochastic variable, no other step method will be created to handle that variable by ``MCMC`` (though you can create multiple step methods for the same variable if desired). 

If you have not created any step method to handle a stochastic variable, each class in ``StepMethodRegistry`` is allowed to examine the variable. More specifically, each step method implements a static method called ``competence(stochastic)``, whose only argument is a single stochastic variable. These methods return values from 0 to 3; 0 meaning the step method cannot safely handle the variable and 3 meaning it will most likely perform well for variables like this. ``MCMC`` objects assign the step method that returns the highest competence value to each stochastic variable.