.. _prob_dists:

*************************
Probability Distributions
*************************

The most fundamental step in building Bayesian models is the specification of a full probability model for the problem at hand. This primarily involved assigning parametric statistical distributions to unknown quantities in the model, in addition to appropriate functional forms for likelihoods to represent the information from the data. To this end, PyMC3 includes a comprehensive set of pre-defined statistical distribution that can be used as model building blocks. 

For example, if we wish to define a particular variable as having a normal prior, we can specify that using an instance of the ``Normal`` class.

::

    with pm.Model():
    
        x = pm.Normal('x', mu=0, sd=1)
        
A variable requires at least a ``name`` argument, and zero or more model parameters, depending on the distribution. Parameter names vary by distribution, using conventional names wherever possible. The example above defines a scalar variable. To make a vector-valued variable, a ``shape`` argument should be provided; for example, a 3x3 matrix of beta random variables could be defined with:

::

    with pm.Model():
    
        p = pm.Beta('p', 1, 1, shape=(3, 3))
        
Probability distributions are all subclasses of ``Distribution``, which in turn has two major subclasses: ``Discrete`` and ``Continuous``. In terms of data types, a ``Continuous`` random variable is given whichever floating point type defined by ``theano.config.floatX``, while ``Discrete`` variables are given ``int16`` types when ``theano.config.floatX`` is ``float32``, and ``int64`` otherwise.

All distributions in ``pm.distributions`` will have two important methods: ``random()`` and ``logp()`` with the following signatures:

::

    class SomeDistribution(Continuous):
    
        def random(self, point=None, size=None, repeat=None):
            ...
            return random_samples
            
        def logp(self, value):
            ...
            return total_log_prob
            
PyMC3 expects the ``logp()`` method to return a log-probability evaluated at the passed ``value`` argument. This method is used internally by all of the inference methods to calculate the model log-probability that is used for fitting models. The ``random()`` method is used to simulate values from the variable, and is used internally for posterior predictive checks.


Custom distributions
====================

Despite the fact that PyMC3 ships with a large set of the most common probability distributions, some problems may require the use of functional forms that are less common, and not available in ``pm.distributions``. One example of this is in survival analysis, where time-to-event data is modeled using probability densities that are designed to accomodate censored data. 

An exponential survival function is defined by:

.. math::

    f(c, t) = \left\{ \begin{array}{l} \exp(\lambda t), \text{if c=1} \\
               \lambda \exp(\lambda t), \text{if c=0}  \end{array} \right.

Such a function can be implemented as a PyMC3 distribution by writing a function that specifies the log-probability, then passing that function as an argument to the ``DesnityDist`` function, which creates an instance of a PyMC3 distribution with the custom function as its log-probability.

For the exponential survival function, this is:

::

    def logp(failure, value):
        return (failure * log(λ) - λ * value).sum()

    exp_surv = pm.DensityDist('exp_surv', logp, observed={'failure':failure, 'value':t})

Similarly, if a random number generator is required, a function returning random numbers corresponding to the probability distribution can be passed as the ``random`` argument.    


Using PyMC distributions without a Model
========================================

Distribution objects, as we have defined them so far, are only usable inside of a ``Model`` context. 

::

    y = Binomial('y', n=10, p=0.5)
    
    
::

    TypeError: No context on context stack
    
    
::

    y = pm.Binomial.dist(n=10, p=0.5)
    
    
::

    >>> y.logp(4).eval()
    array(-1.5843639373779297, dtype=float32)
    
::

    >>> y.random(size=3)
    array([5, 4, 3])


::

    y = Binomial('y', n=10, p=0.5, model=model)
            
Auto-transformation
===================



::

    with pm.Model() as model:
        g = pm.Gamma('g', 1, 1)
        
::

    >>> model.vars
    [g_log__]
    
::

    >>> model.deterministics
    [g]
