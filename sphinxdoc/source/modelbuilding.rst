Bayesian inference begins with specification of a probability model relating unknown variables to data. PyMC provides three basic building blocks for Bayesian probability models: ``Stochastic``, ``Deterministic`` and ``Potential``. 

A ``Stochastic`` object represents a variable whose value is not completely determined by its parents, and a ``Deterministic`` object represents a variable that is entirely determined by its parents. In object-oriented programming parlance, ``Stochastic`` and ``Deterministic`` are subclasses of the ``Variable`` class, which is essentially a template for more specific subclasses that are actually implemented in models. The third basic class, representing `factor potentials' (\cite{dawidmarkov,Jordan:2004p5439}), represents an arbitrary log-probability term. ``Potential`` and ``Variable``, in turn, are subclasses of ``Node``.

% TODO: Need a better description of what a Potential is. Given the description of Stochastic and Deterministic we have given, its not clear where Potential fits in, as it classifies the world into 2 things -- completely determined by parents and not.

% PyMC also provides container classes for variables to make it easier to program of certain dependency situations, such as when a variable is defined by its dependence on an entire Markov chain.

\medskip
PyMC probability models are simply linked groups of ``Stochastic``, ``Deterministic`` and ``Potential`` objects. These objects have very limited awareness of the models in which they are embedded and do not themselves possess methods for updating their values in fitting algorithms. Objects responsible for fitting probability models are described in chapter \ref{chap:modelfitting}.
 

\hypertarget{stochastic}{}
The ``Stochastic`` class \label{stochastic}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A stochastic variable has the following major attributes: 
\begin{description}
    \item[``value``:] The variable's current value.
    \item[``logp``:] The log-probability of the variable's current value given the values of its parents.
\end{description}
A stochastic variable can optionally be endowed with a method called ``\bfseries random``, which draws a value for the variable given the values of its parents\footnote{Note that the ``random`` method does not provide a Gibbs sample unless the variable has no children.}. Stochastic objects have the following additional attributes that are generally specified automatically, or only specified under particular circumstances:
\begin{description}
    \item[``parents``:] A dictionary containing the variable's parents. The keys of the dictionary correspond to the names assigned to the variable's parents by the variable, and the values correspond to the actual parents. For example, the keys of `s`'s parents dictionary in model (\ref{disastermodel}) would be ``'t_l'`` and ``'t_h'``. Thanks to Python's dynamic typing, the actual parents (*i.e.* the values of the dictionary) may be of any class or type.
    \item[``children``:] A set containing the variable's children. This set is produced automatically; the user doesn't need to worry about filling it.
    \item[``extended_parents``:] A set containing all the stochastic variables on which the variable depends either directly or via a sequence of deterministic variables. If the value of any of these variables changes, the variable will need to recompute its log-probability. This set is produced automatically.
    \item[``extended_children``:] A set containing all the stochastic variables and potentials that depend on the variable either directly or via a sequence of deterministic variables. If the variable's value changes, all of these variables will need to recompute their log-probabilities. This set is produced automatically.
    \item[``coparents``:] A set containing all the stochastic variables that share extended children with the variable.
    \item[``moral_neighbors``:] A set containing the union of the variable's extended parents, extended children and coparents, with Potential objects removed.
    \item[``markov_blanket``:] A set containing self and self's moral neighbors.
    \item[``isdata``:] A flag (boolean) indicating whether the variable's value has been observed (is fixed).
    \item[``dtype``:] A Numpy dtype object (such as ``numpy.int``) that specifies the type of the variable's value to fitting methods. If this is ``None`` (default) then no type is enforced.
    % \item[``__name__``:] The name of the variable, should be unique.
    %    \item[``__doc__``:] The docstring of the variable.
\end{description}


.. rubric:: Creation of stochastic variables


There are three main ways to create stochastic variables, called the \textbf{automatic}, \textbf{decorator}, and \textbf{direct} interfaces.

\begin{description}    
    \item[Automatic] Stochastic variables with standard distributions provided by PyMC (see chapter \ref{chap:distributions} ) can be created in a single line using special subclasses of ``Stochastic``. For example, the uniformly-distributed discrete variable `s` in (\ref{disastermodel}) could be created using the automatic interface as follows::

        s = DiscreteUniform('s', 1851, 1962, value=1900)


    In addition to the classes in chapter \ref{chap:distributions}, ``scipy.stats.distributions``' random variable classes are wrapped as ``Stochastic`` subclasses if SciPy is installed. These distributions are in the submodule ``pymc.SciPyDistributions``

    Users can call the class factory ``stochastic_from_dist`` to produce ``Stochastic`` subclasses of their own from probability distributions not included with PyMC.%  These classes' init methods take the following arguments:
    % \begin{description}
    %     \item[``name``:] The name of the variable.
    %     \item[``value``:] An initial value for the variable.
    %     \item[``parents``:] Keyword arguments specifying the parents of the variable.
    %     \item[``isdata`` (optional)]
    %     \item[``doc`` (optional):] The docstring of the variable.
    %     \item[``verbose`` (optional):] An integer from 0 to 3.
    %     \item[``trace`` (optional):] A boolean indicating whether a trace should be kept for this variable in Monte Carlo fitting methods.
    %     \item[``cache_depth``:] See section \ref{sec:caching}. 
    % \end{description}
    
    
    \item[Decorator] Uniformly-distributed discrete stochastic variable `s` in (\ref{disastermodel}) could be created as follows::

	@stochastic(dtype=int)
	def s(value=1900, t_l=1851, t_h=1962):
	    """The switchpoint for the rate of disaster occurrence."""
	    if value > t_h or value < t_l:
	        return -Inf
	    else:
	        return -log(t_h - t_l + 1) 

Note that this is a simple Python function, preceded by a Python expression called a \textbf{decorator}, here called ``@stochastic``. Generally, decorators enhance functions with additional properties or functionality. The ``Stochastic`` object produced by the ``@stochastic`` decorator will evaluate its log-probability using the function ``s``. The ``value`` argument, which is required, provides an initial value for the variable. The remaining arguments will be assigned as parents of ``s`` (*i.e.* they will populate the ``parents`` dictionary).

The ``value`` and parents of stochastic variables may be any objects, provided their log-probability functions return a real number (Numpy ``float``). PyMC and SciPy both provide fast implementations of several standard probability distributions that may be helpful for creating custom stochastic variables.

    The decorator ``stochastic`` can take several arguments: 
    \begin{itemize}
        \item A flag called ``trace``, which signals to ``MCMC`` instances whether an MCMC trace should be kept for the stochastic variable. ``@stochastic(trace = False)`` would turn tracing off. Defaults to ``True``.
        \item A flag called ``plot``, which signals to ``MCMC`` instances whether summary plots should be produced for this variable. Defaults to ``True``.
        \item An integer-valued argument called ``verbose`` that controls the amount of output the variable prints to the screen. The default is `0`, no output; the maximum value is `3`. 
        \item A Numpy datatype called ``dtype``. Decorating a log-probability function with ``@stochastic(dtype=int)`` would produce a discrete random variable. Such a variable will cast its value to either an integer or an array of integers. The default dtype is ``float``.
    \end{itemize} 

    The decorator interface has a slightly more complex implementation which allows you to specify a ``random`` method for sampling the stochastic variable's value conditional on its parents.
::
  
	@stochastic(dtype=int)
	def s(value=1900, t_l=1851, t_h=1962):
	    """The switchpoint for the rate of disaster occurrence."""

	    def logp(value, t_l, t_h):
	        if value > t_h or value < t_l:
	            return -Inf
	        else:
	            return -log(t_h - t_l + 1) 
	            
	    def random(t_l, t_h):
	        return round( (t_l - t_h) * random() ) + t_l

	    rseed = 1.

The stochastic variable again gets its name, docstring and parents from function `s`, but in this case it will evaluate its log-probability using the ``logp`` function. The ``random`` function will be used when ``s.random()`` is called. Note that ``random`` doesn't take a ``value`` argument, as it generates values itself. The optional ``rseed`` variable provides a seed for the random number generator. The stochastic's ``value`` argument is optional when a ``random`` method is provided; if no initial value is provided, it will be drawn automatically using the ``random`` method.

    \item[Direct] It's possible to instantiate ``Stochastic`` directly::

	def s_logp(value, t_l, t_h):
	    if value > t_h or value < t_l:
	        return -Inf
	    else:
	        return -log(t_h - t_l + 1) 

	def s_rand(t_l, t_h):
	    return round( (t_l - t_h) * random() ) + t_l

	s = Stochastic( logp = s_logp, 
	                doc = 'The switchpoint for the rate of disaster occurrence.',
	                name = 's', 
	                parents = {'t_l': 1851, 't_h': 1962},
	                random = s_rand,                 
	                trace = True,                 
	                value = 1900,
	                dtype=int,
	                rseed = 1., 
	                isdata = False,
	                cache_depth = 2,
	                plot=True,
	                verbose = 0)

Notice that the log-probability and random variate functions are specified externally and passed to ``Stochastic`` as arguments. This is a rather awkward way to instantiate a stochastic variable; consequently, such implementations should be rare.

\end{description}


\hypertarget{sub:warning}{}

.. rubric:: Don't update stochastic variables' values in-place} \label{sub:warning


\pdfbookmark[0]{Don't update stochastic variables' values in-place}{sub:warning}

``Stochastic`` objects' values should not be updated in-place. This confuses PyMC's caching scheme and corrupt the process used for accepting or rejecting proposed values in the MCMC algorithm. The only way a stochastic variable's value should be updated is using statements of the following form::

    A.value = new_value

The following are in-place updates and should *never* be used:
\begin{itemize}
    \item ``A.value += 3``
    \item ``A.value[2,1] = 5``
    \item ``A.value.attribute = new_attribute_value``.
\end{itemize}

This restriction becomes onerous if a step method proposes values for the elements of an array-valued variable separately. In this case, it may be preferable to partition the variable into several variables stored in an array or list.




Data \label{data
~~~~~~~~~~~~~~~~

\pdfbookmark[0]{Data}{data}

Although the data `D` is represented by a random variable in the model, we have fixed its value by observing it. Such variables are represented by ``Stochastic`` objects whose ``isdata`` attribute is set to ``True``. If a stochastic variable's ``isdata`` flag is ``True``, its value cannot be changed.

.. rubric:: Declaring stochastic variables to be data



In the short and long interfaces, a ``Stochastic`` object's ``isdata`` flag can be set to true by stacking a ``@data`` decorator on top of the ``@stochastic`` decorator::

	@data
	@stochastic(dtype=int)
	def D(value = count_array, switchpoint = s, early_rate = e, late_rate = l):
	    """The observed annual disaster counts."""
	    logp = sum(-value[:switchpoint]) + early_rate * log(value[:switchpoint]) \
	            - gammaln(early_rate))
	    logp += sum(-value[switchpoint:] + late_rate * log(value[switchpoint:]) \
	            - gammaln(late_rate))
	    return logp

In the automatic and direct interfaces, the ``isdata`` argument can be simply set to ``True``.


\hypertarget{deterministic}{}
The ``Deterministic`` class} \label{deterministic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

\pdfbookmark[0]{The Deterministic class}{deterministic}

The ``Deterministic`` class represents variables whose values are completely determined by the values of their parents. For example, in model (\ref{disastermodel}), `r` is a deterministic variable. Recall it was defined by

.. math::

   r_t=\left\{\begin{array}{ll}
   e & t\le s\\ 
   l & t>s \end{array}\right.,


so `r`'s value can be computed exactly from the values of its parents `e`, `l` and `s`.

A deterministic variable's most important attribute is ``\bfseries value``, which gives the current value of the variable given the values of its parents. Like ``Stochastic``'s ``logp`` attribute, this attribute is computed on-demand and cached for efficiency.

A Deterministic variable has the following additional attributes:
\begin{description}
    \item[``parents``:] A dictionary containing the variable's parents. The keys of the dictionary correspond to the names assigned to the variable's parents by the variable, and the values correspond to the actual parents. Thanks to Python's dynamic typing, parents may be of any class or type.
    \item[``children``:] A set containing the variable's children, which must be nodes. This set is produced automatically; the user doesn't need to worry about filling it.
    % \item[``__name__``:] The name of the variable, should be unique.
    %     \item[``__doc__``:] The docstring of the variable.
\end{description}
Deterministic variables have no methods.


.. rubric:: Creation of deterministic variables


Deterministic variables are less complicated than stochastic variables, and have similar \textbf{automatic}, \textbf{decorator}, and \textbf{direct} interfaces:
\begin{description}
   \item[Automatic] A handful of common functions have been wrapped in Deterministic objects. These are brief enough to list:
   \begin{description}
      \item[``LinearCombination``:] Has two parents `x` and `y`, both of which must be iterable (*i.e.* vector-valued). This function returns:
      \[
      \sum_i x_i^{\prime} y_i.
      \]
      \item[``Index``:] Has three parents `x`, `y` and ``index``. `x` and `y` must be iterables, ``index`` must be valued as an integer. Index returns the dot product of `x` and `y` for the elements specified by \mathttt{index}:
      \[
      x[\mathtt{index}]^T y[\mathtt{index}].
      \]
      ``Index`` is useful for implementing dynamic models, in which the parent-child connections change.
      \item[``Lambda``:] Converts an anonymous function (in Python, called \textbf{lambda functions}) to a ``Deterministic`` instance on a single line.
      \item[``CompletedDirichlet``:] PyMC represents Dirichlet variables of length `k` by the first `k-1` elements; since they must sum to 1, the `k^{th}` element is determined by the others. ``CompletedDirichlet`` appends the `k^{th}` element to the value of its parent `D`.      
      \item[``Logit``, ``InvLogit``, ``StukelLogit``, ``StukelInvLogit``:] Various common link functions for generalized linear models.
   \end{description}
   It's a good idea to use these classes when feasible, because certain fitting methods (Gibbs step methods in particular) implicitly know how to take them into account.

    \item[Decorator] A deterministic variable can be created via a decorator in a way very similar to ``Stochastic``'s decorator interface:
\begin{verbatim}
@deterministic
def r(switchpoint = s, early_rate = e, late_rate = l):
    """The rate of disaster occurrence."""
    value = zeros(N)
    value[:switchpoint] = early_rate
    value[switchpoint:] = late_rate
    return value
\end{verbatim}
Notice that rather than returning the log-probability, as is the case for Stochastic objects, the function returns the value of the deterministic object, given its parents. This return value may be of any type, as is suitable for the problem at hand. Arguments' keys and values are converted into a parent dictionary as with ``Stochastic``'s short interface. The ``deterministic`` decorator can take ``trace`` and ``verbose`` arguments, like the ``stochastic`` decorator.

Of course, since deterministic nodes are not expected to generate random variates, the longer implementation of the decorator interface available to ``Stochastic`` objects is not relevant here.

    \item[Direct] Deterministic objects can also be instantiated directly, by passing the evaluation function to the Deterministic class as an argument:
\begin{verbatim}
def r_eval(switchpoint = s, early_rate = e, late_rate = l):
    value = zeros(N)
    value[:switchpoint] = early_rate
    value[switchpoint:] = late_rate
    return value

r = Deterministic(  eval = r_eval, 
                    name = 'r',
                    parents = {'switchpoint': s, 'early_rate': e, 'late_rate': l}),
                    doc = 'The rate of disaster occurrence.',
                    trace = True,
                    verbose = 0,
                    cache_depth = 2)
\end{verbatim}
The ``trace`` flag signals to ``Model`` whether to keep a trace for the variable, as with stochastic variables.
\end{description}

Note that deterministic variables have no ``isdata`` flag. If a deterministic variable's value were known, its parents would be restricted to the inverse image of that value under the deterministic variable's evaluation function. This usage would be extremely difficult to support in general, but it can be implemented for particular applications at the ``StepMethod`` level.

\hypertarget{container}{}
Containers} \label{container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

\pdfbookmark[0]{Containers}{container}

In some situations, such as a state-space model, it would be inconvenient to assign a unique label to each parent of `y`:

.. math::

   x_0 &\sim \textup N(0,\tau_x)

   x_{i+1}|x_i &\sim \textup{N}(x_i, \tau_x)

   y|x &\sim \textup N\left(\sum_{i=0}^{N-1}x_i^2,\tau_y\right)

   &i=0,\ldots, N-2


Here, `y` depends on every element of the Markov chain `x`, but we wouldn't want to manually enter `N` parent labels ```x_0'``, ```x_1'``, etc.

This situation can be handled naturally in PyMC:
\begin{verbatim}
x_0 = Normal(`x_0', mu=0, tau=1)

x = [x_0]
last_x = x_0

for i in range(1,N):          
   x_now = Normal(`x_%i' % i, mu=last_x, tau=1)        
   last_x = x_now 
   x.append(x_now)

@data
@stochastic
def y(value = 1, mu = x, tau = 100):
    mu_sum = 0
    for i in range(N):
        mu_sum += mu[i] ** 2
    return normal_like(value, mu_sum, tau)
\end{verbatim}
PyMC automatically wraps list ``x`` in an appropriate ``Container`` class. The python expression ```x_\%i' \% i`` labels each Normal object in the container with the appropriate index `i`.

Containers, like variables, have an attribute called ``value``. This attribute returns a copy of the (possibly nested) iterable that was passed into the container function, but with each variable inside replaced with its corresponding value. 

Containers can currently be constructed from lists, tuples, dictionaries, Numpy arrays, modules, sets or any object with a ``__dict__`` attribute. Variables and non-variables can be freely mixed in these containers, and different types of containers can be nested\footnote{Nodes whose parents are containers make private shallow copies of those containers. This is done for technical reasons rather than to protect users from accidental misuse.}. Containers attempt to behave like the objects they wrap. All containers are subclasses of ``ContainerBase``. 

Containers have the following useful attributes in addition to ``value``:
\begin{itemize}
    \item``variables``
    \item``stochastics``
    \item``potentials``
    \item``deterministics``
    \item``data_stochastics``
    \item``step_methods``.
\end{itemize}
Each of these attributes is a set containing all the objects of each type in a container, and within any containers in the container.


\hypertarget{potential}{}
The Potential class} \label{potential
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

\pdfbookmark[0]{The Potential class}{potential}

% WE PROBABLY NEED TO GIVE A GOOD EXAMPLE OF WHERE A POTENTIAL IS DIFFERENT FROM A DETERMINISTIC;
% THIS PROBABLY WONT BE CLEAR TO EVERYONE. THE KEY DIFFERENCE IS THAT A POTENTIAL IS PART OF THE
% JOINT POSTERIOR, NO?
% 

The joint density corresponding to model (\ref{disastermodel}) can be written as follows:

.. math::

   p(D,s,l,e) = p(D|s,l,e) p(s) p(l) p(e).


Each factor in the joint distribution is a proper, normalized probability distribution for one of the variables conditional on its parents. Such factors are contributed by ``Stochastic`` objects.

In some cases, it's nice to be able to modify the joint density by incorporating terms that don't correspond to probabilities of variables conditional on parents, for example:

.. math::

   p(x_0, x_2, \ldots x_{N-1}) \propto \prod_{i=0}^{N-2} \psi_i(x_i, x_{i+1}).


Arbitrary factors such as `\psi` are contributed by objects of class ``Potential`` (\cite{dawidmarkov} and \cite{Jordan:2004p5439} call these terms `factor potentials'). Bayesian hierarchical notation (cf model (\ref{disastermodel})) doesn't accomodate these potentials. They are most useful in cases where there is no natural dependence hierarchy, such as Markov random fields. They are also useful for expressing `soft data' \citep{Christakos:2002p5506}.

Even when there is a definite dependence hierarchy, potentials can provide a useful shorthand. Consider a new example: we have a dataset `t` consisting of the days on which several marked animals were recaptured. We believe that the probability `S` that an animal is not recaptured on any given day can be explained by a covariate vector `x`. We model this situation as follows:

.. math::

   t_i|S_i \sim \textup{Geometric}(S_i), & i=1\ldots N

   S_i = \textup{logit}^{-1}(\beta x_i), &i=1\ldots N

   \beta\sim \textup{N}(\mu_\beta, V_\beta).


So far, so good. Now suppose we have some knowledge of other related experiments and we have a good idea of what `S` will be before seeing the data. It's not obvious how to work this prior information in, because as we've written the model `S` is completely determined by `\beta`. There are three options within the strict Bayesian hierarchical framework:
\begin{itemize}
    \item Work the prior information into the prior on `\beta`.
    \item Incorporate the data from the previous experiments explicitly into the model.
    \item Refactor the model so that `S` is at the bottom of the hierarchy, and assign the prior directly.
\end{itemize}

Factor potentials provide a convenient way to incorporate the prior information without the need for such major modifications. We can simply modify the joint distribution from

.. math::

   p(t|S(x,\beta)) p(\beta)


to

.. math::

   \gamma(S,a,b) p(t|S(x,\beta)) p(\beta),


where `\gamma` expresses the prior information. It's a good idea to check the induced priors on `S` and `\beta` for sanity. This can be done in PyMC by fitting the model with the data `t` removed.

\bigskip
Potentials have one important attribute, ``\bfseries logp``, the log of their current probability or probability density value given the values of their parents. The only other additional attribute of interest is ``parents``, a dictionary containing the potential's parents. Potentials have no methods. They have no ``trace`` attribute, because they are not variables. They cannot serve as parents of variables (for the same reason), so they have no ``children`` attribute.


.. rubric:: Creation of ``Potentials``


There are two ways to create potentials:
\begin{description}
    \item[Decorator] A potential can be created via a decorator in a way very similar to ``Deterministic``'s decorator interface:
\begin{verbatim}
@potential
def psi_i(x_lo = x[i], x_hi = x[i+1]):
    """A pair potential"""
    return -(xlo - xhi)**2
\end{verbatim}
The function supplied should return a Numpy ``float``. The ``potential`` decorator can take ``verbose`` and ``cache_depth`` arguments like the ``stochastic`` decorator.
    \item[Direct] The same potential could be created directly as follows:
\begin{verbatim}
def psi_i_logp(x_lo = x[i], x_hi = x[i+1]):
    return -(xlo - xhi)**2
        
psi_i = Potential(  logp = psi_i_logp, 
                    name = 'psi_i',
                    parents = {'xlo': x[i], 'xhi': x[i+1]},
                    doc = 'A pair potential',
                    verbose = 0,
                    cache_depth = 2)
\end{verbatim}
\end{description}


\hypertarget{graphical}{}
Graphing models} \label{graphical
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

\pdfbookmark[0]{Graphing models}{graphical}

The function ``graph`` draws graphical representations of ``Model`` (Chapter \ref{chap:modelfitting}) instances using GraphViz via the Python package PyDot (if they are installed). See \cite{dawidmarkov} and \cite{Jordan:2004p5439} for more discussion of useful information that can be read off of graphical models. Note that these authors do not consider deterministic variables.

The symbol for stochastic variables is an ellipse. Parent-child relationships are indicated by arrows. These arrows point from parent to child and are labeled with the names assigned to the parents by the children. A graphical representation of model \ref{disastermodel} follows:
\begin{center}
    \epsfig{file=DisasterModel.pdf, width=6cm} 
\end{center} 
`D` is shaded because it is flagged as data.

PyMC's symbol for deterministic variables is a downward-pointing triangle. A graphical representation of model \ref{disastermodel} with `r` explicit follows:
\begin{center}
    \epsfig{file=DisasterModel2.pdf, width=6cm} 
\end{center}
% Note that if a deterministic variable has more than one child, its parents each inherit all of its children when it is made implicit:
% \begin{center}
%     \epsfig{file=DeterministicPreInheritance.pdf, width=3.5cm} `\Rightarrow` \epsfig{file=DeterministicPostInheritance.pdf, width=5cm}
% \end{center}
% These inherited children can be accessed via the ``extended_children`` attributes of the parents.

The symbol for factor potentials is a rectangle:
\begin{center}
    \epsfig{file=PotExample.pdf, width=10cm} 
\end{center}
Factor potentials are usually associated with *undirected* grahical models. In undirected representations, each parent of a potential is connected to every other parent by an undirected edge:
\begin{center}
    \epsfig{file=PotExampleCollapsed.pdf, width=5cm}
\end{center}

Directed or mixed graphical models can be represented in an undirected form by `moralizing', which is done by the function ``moral_graph``.


Class ``LazyFunction`` and caching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

\label{sec:caching} 

The ``logp`` attributes of stochastic variables and potentials and the ``value`` attributes of deterministic variables are wrappers for instances of class ``LazyFunction``. Lazy functions are wrappers for ordinary Python functions. A lazy function ``L`` could be created from a function ``fun`` as follows:
\begin{verbatim}
L = LazyFunction(fun, arguments)
\end{verbatim}
The argument ``arguments`` is a dictionary container; ``fun`` must accept keyword arguments only. When ``L``'s ``get()`` method is called, the return value is the same as the call 
\begin{verbatim}
fun(**arguments.value)
\end{verbatim}
Note that no arguments need to be passed to ``L.get``; lazy functions memorize their arguments.

Before calling ``fun``, ``L`` will check the values of ``arguments.variables`` against an internal cache. This comparison is done *by reference*, not by value, and this is part of the reason why stochastic variables' values cannot be updated in-place. If ``arguments.variables``' values match a frame of the cache, the corresponding output value is returned and ``fun`` is not called. If a call to ``fun`` is needed, ``arguments.variables``' values and the return value replace the oldest frame in the cache. The depth of the cache can be set using the optional init argument ``cache_depth``, which defaults to 2.

Caching is helpful in MCMC, because variables' log-probabilities and values tend to be queried multiple times for the same parental value configuration. The default cache depth of 2 turns out to be most useful in Metropolis-Hastings-type algorithms involving proposed values that may be rejected.

Lazy functions are implemented in C using Pyrex, a language for writing Python extensions.
