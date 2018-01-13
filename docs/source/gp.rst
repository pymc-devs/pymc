******************
Gaussian Processes
******************

GP Basics
=========

Sometimes an unknown parameter or variable in a model is not a scalar value or
a fixed-length vector, but a *function*.  A Gaussian process (GP) can be used
as a prior probability distribution whose support is over the space of
continuous functions.  A GP prior on the function :math:`f(x)` is usually written,

.. math::

  f(x) \sim \mathcal{GP}(m(x), \, k(x, x')) \,.

The function values are modeled as a draw from a multivariate normal
distribution that is parameterized by the mean function, :math:`m(x)`, and the
covariance function, :math:`k(x, x')`.  Gaussian processes are a convenient
choice as priors over functions due to the marginalization and conditioning
properties of the multivariate normal distribution.  Usually, the marginal
distribution over :math:`f(x)` is evaluated during the inference step.  The
conditional distribution is then used for predicting the function values
:math:`f(x_*)` at new points, :math:`x_*`.  

The joint distribution of :math:`f(x)` and :math:`f(x_*)` is multivariate
normal,

.. math::

  \begin{bmatrix} f(x) \\ f(x_*) \\ \end{bmatrix} \sim
  \text{N}\left( 
    \begin{bmatrix} m(x)  \\ m(x_*)    \\ \end{bmatrix} \,,
    \begin{bmatrix} k(x,x')    & k(x_*, x)    \\ 
                    k(x_*, x) &  k(x_*, x_*')  \\ \end{bmatrix}
          \right) \,.

Starting from the joint distribution, one obtains the marginal distribution
of :math:`f(x)`, as :math:`\text{N}(m(x),\, k(x, x'))`.  The conditional
distribution is

.. math::

  f(x_*) \mid f(x) \sim \text{N}\left( k(x_*, x) k(x, x)^{-1} [f(x) - m(x)] + m(x_*) ,\, 
    k(x_*, x_*) - k(x, x_*) k(x, x)^{-1} k(x, x_*) \right) \,.

.. note::

  For more information on GPs, check out the book `Gaussian Processes for
  Machine Learning <http://www.gaussianprocess.org/gpml/>`_ by Rasmussen &
  Williams, or `this introduction <https://www.ics.uci.edu/~welling/teaching/KernelsICS273B/gpB.pdf>`_ 
  by D. Mackay.

PyMC3 is a great environment for working with fully Bayesian Gaussian Process
models.  GPs in PyMC3 have a clear syntax and are highly composable, and many 
predefined covariance functions (or kernels), mean functions, and several GP 
implementations are included.  GPs are treated as distributions that can be
used within larger or hierarchical models, not just as standalone regression 
models.

Mean and covariance functions
=============================

Those who have used the GPy or GPflow Python packages will find the syntax for
construction mean and covariance functions somewhat familiar.  When first
instantiated, the mean and covariance functions are parameterized, but not
given their inputs yet.  The covariance functions must additionally be provided
with the number of dimensions of the input matrix, and a list that indexes
which of those dimensions they are to operate on.  The reason for this design
is so that covariance functions can be constructed that are combinations of
other covariance functions.

For example, to construct an exponentiated quadratic covariance function that
operates on the second and third column of a three column matrix representing
three predictor variables::

    ls = [2, 5] # the lengthscales
    cov_func = pm.gp.cov.ExpQuad(input_dim=3, ls=ls, active_dims=[1, 2])

Here the :code:`ls`, or lengthscale, parameter is two dimensional, allowing the second
and third dimension to have a different lengthscale.  The reason we have to
specify :code:`input_dim`, the total number of columns of :code:`X`, and
:code:`active_dims`, which of those columns or dimensions the covariance
function will act on, is because :code:`cov_func` hasn't actually seen the
input data yet.  The :code:`active_dims` argument is optional, and defaults to
all columns of the matrix of inputs.  

Covariance functions in PyMC3 closely follow the algebraic rules for kernels,
which allows users to combine covariance functions into new ones, for example:

- The sum two covariance functions is also a covariance function::


    cov_func = pm.gp.cov.ExpQuad(...) + pm.gp.cov.ExpQuad(...)

- The product of two covariance functions is also a covariance function::


    cov_func = pm.gp.cov.ExpQuad(...) * pm.gp.cov.Periodic(...)
    
- The product (or sum) of a covariance function with a scalar is a covariance
function::

    
    cov_func = eta**2 * pm.gp.cov.Matern32(...)
    
After the covariance function is defined, it is now a function that is
evaluated by calling :code:`cov_func(x, x)` (or :code:`mean_func(x)`).  Since
PyMC3 is built on top of Theano, it is relatively easy to define and experiment
with non-standard covariance and mean functons.  For more information check out
the tutorial on covariance functions.


GP Implementations
==================

PyMC3 includes several GP implementations, including marginal and latent
variable models and also some fast approximations.  Their usage all follows a
similar pattern:  First, a GP is instantiated with a mean function and a
covariance function.  Then, GP objects can be added together, allowing for
function characteristics to be carefully modeled and separated.  Finally, one
of `prior`, `marginal_likelihood` or `conditional` methods is called on the GP
object to actually construct the PyMC3 random variable that represents the
function prior.

Using :code:`gp.Latent` for the example, the syntax to first specify the GP
is::

    gp = pm.gp.Latent(mean_func, cov_func)

The first argument is the mean function and the second is the covariance
function.  We've made the GP object, but we haven't made clear which function
it is to be a prior for, what the inputs are, or what parameters it will be
conditioned on.  

.. note::

  The :code:`gp.Marginal` class and similar don't have a :code:`prior` method.
  Instead they have a :code:`marginal_likelihood` method that is used similarly,
  but has additional required arguments, such as the observed data, noise,
  or other, depending on the implementation.  See the notebooks for examples.
  The :code:`conditional` method works similarly.

Calling the `prior` method will create a PyMC3 random variable that represents
the latent function :math:`f(x) = \mathbf{f}`::
  
	f = gp.prior("f", X)

:code:`f` is a random variable that can be used within a PyMC3 model like any
other type of random variable.  The first argument is the name of the random
variable representing the function we are placing the prior over.  
The second argument is the inputs to the function that the prior is over, 
:code:`X`.  The inputs are usually known and present in the data, but they can
also be PyMC3 random variables.  If the inputs are a Theano tensor or a 
PyMC3 random variable, the :code:`shape` needs to be given.

Usually at this point, inference is performed on the model.  The 
:code:`conditional` method creates the conditional, or predictive,
distribution over the latent function at arbitrary :math:`x_*` input points,
:math:`f(x_*)`.  To construct the conditional distribution we write::

	f_star = gp.conditional("f_star", X_star)

Additive GPs
============

The GP implementation in PyMC3 is constructed so that it is easy to define 
additive GPs and sample from individual GP components.  We can write::

    gp1 = pm.gp.Marginal(mean_func1, cov_func1)
    gp2 = pm.gp.Marginal(mean_func2, cov_func2)
    gp3 = gp1 + gp2

The GP objects have to have the same type, :code:`gp.Marginal` cannot
be added to :code:`gp.Latent`.

Consider two independent GP distributed functions, :math:`f_1(x) \sim
\mathcal{GP}\left(m_1(x),\, k_1(x, x')\right)` and :math:`f_2(x) \sim
\mathcal{GP}\left( m_2(x),\, k_2(x, x')\right)`.  The joint distribution of
:math:`f_1,\, f_1^*,\, f_2,\, f_2^*,\, f_1 + f_2 and f_1^* + f_2^*` is

.. math::

  \begin{bmatrix} f_1 \\ f_1^* \\ f_2 \\ f_2^* 
               \\ f_1 + f_2    \\ f_1^* + f_2^* \end{bmatrix} \sim
  \text{N}\left( 
    \begin{bmatrix} m_1 \\ m_1^* \\ m_2 \\ m_2^* \\
                    m_1 + m_2    \\ m_1^* + m_2^*   \\ \end{bmatrix} \,,\,
    \begin{bmatrix} 
      K_1       &  K_1^*     &   0       &    0      & K_1        & K_1^*              \\
      K_1^{*^T} &  K_1^{**}  &   0       &    0      & K_1^*      & K_1^{**}           \\
      0         &  0         & K_2       & K_2^*     & K_2        & K_2^{*}            \\
      0         &  0         & K_2^{*^T} & K_2^{**}  & K_2^{*}    & K_2^{**}           \\
      K_1       &  K_1^{*}   & K_2       & K_2^{*}   & K_1 + K_2  & K_1^{*} + K_2^{*}  \\
      K_1^{*^T} & K_1^{**} & K_2^{*^T} & K_2^{**} & K_1^{*^T}+K_2^{*^T} & K_1^{**}+K_2^{**} 
    \end{bmatrix}
  \right) \,.

Using the joint distribution to obtain the conditional distribution of :math:`f_1^*`
with the contribution due to :math:`f_1 + f_2` factored out, we get

.. math::
  f_1^* \mid f_1 + f_2 \sim \text{N}\left(
    m_1^* + K_1^{*^T}(K_1 + K_2)^{-1}\left[f_1 + f_2 - m_1 - m_2\right] \,,\,
    K_1^{**} - K_1^{*^T}(K_1 + K_2)^{-1}K_1^* \right) \,.


These equations show how to break down GP models into individual components to see how each
contributes to the data.  For more information, check out `David Duvenaud's PhD
thesis <https://www.cs.toronto.edu/~duvenaud/thesis.pdf>`_.

The GP objects in PyMC3 keeps track of these marginals automatically.  The
following code sketch shows how to define the conditional distribution of
:math:`f_2^*`.  We use `gp.Marginal` in the example, but the same works for
other implementations.  The first block fits the GP prior.  We denote
:math:`f_1 + f_2` as just :math:`f` for brevity::

    with pm.Model() as model:
        gp1 = pm.gp.Marginal(mean_func1, cov_func1)
        gp2 = pm.gp.Marginal(mean_func2, cov_func2)
        
        # gp represents f1 + f2.  
        gp = gp1 + gp2
        
        f = gp.marginal_likelihood("f", X, y, noise)
        
        trace = pm.sample(1000)


To construct the conditional distribution of :code:`gp1` or :code:`gp2`, we 
also need to include the additional arguments, :code:`X`, :code:`y`, and 
:code:`noise`::

    with model:
        # conditional distributions of f1 and f2
        f1_star = gp1.conditional("f1_star", X_star, 
                                  given={"X": X, "y": y, "noise": noise, "gp": gp})
        f2_star = gp2.conditional("f2_star", X_star, 
                                  given={"X": X, "y": y, "noise": noise, "gp": gp})

        # conditional of f1 + f2, `given` not required
        f_star = gp.conditional("f_star", X_star)

This second block produces the conditional distributions.  Notice that extra 
arguments are required for conditionals of :math:`f1` and :math:`f2`, but not
:math:`f`.  This is because those arguments are cached when 
:code:`.marginal_likelihood` is called on :code:`gp`.

.. note::
  When constructing conditionals, the additional arguments :code:`X`, :code:`y`,
  :code:`noise` and :code:`gp` must be provided as a dict called `given`!

Since the marginal likelihoood method of :code:`gp1` or :code:`gp2` weren't called, 
their conditionals need to be provided with the required inputs.  In the same 
fashion as the prior, :code:`f_star`, :code:`f1_star` and :code:`f2_star` are random
variables that can now be used like any other random variable in PyMC3.  

Check the notebooks for detailed demonstrations of the usage of GP functionality
in PyMC3.


