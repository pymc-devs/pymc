===========================
Gaussian Processes in PyMC3
===========================

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

=============================
Overview of GP usage In PyMC3
=============================

PyMC3 is a great environment for working with fully Bayesian Gaussian Process
models.  The GP functionality of PyMC3 is meant to have a clear syntax and be
highly composable.  It includes many predefined covariance functions (or
kernels), mean functions, and several GP implementations.  GPs are treated as
distributions that can be used within larger or hierarchical models, not just
as standalone regression models.

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

    lengthscales = [2, 5]
    cov_func = pm.gp.cov.ExpQuad(input_dim=3, lengthscales, active_dims=[1, 2])

Here the :code:`lengthscales` parameter is two dimensional, allowing the second
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
    
- The product (or sum) of a covariance function with a scalar is a covariance function::

    
    cov_func = eta**2 * pm.gp.cov.Matern32(...)
    
After the covariance function is defined, it is now a function that is
evaluated by calling :code:`cov_func(x, x)` (or :code:`mean_func(x)`).  For more
information on mean and covariance functions in PyMC3, check out the tutorial
on covariance functions.


:code:`gp.*` implementations
============================

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

GPs in PyMC3 are additive, which means we can write::

    gp1 = pm.gp.Latent(mean_func1, cov_func1)
    gp2 = pm.gp.Latent(mean_func2, cov_func2)
    gp3 = gp1 + gp2
   

Calling the `prior` method will create a PyMC3 random variable that represents
the latent function :math:$f(x) = \mathbf{f}$::
  
	f = gp3.prior("f", n_points, X)

:code:`f` is a random variable that can be used within a PyMC3 model like any
other type of random variable.  The first argument is the name of the random
variable representing the function we are placing the prior over.  The second
argument is :code:`n_points`, which is number of points that the GP acts on ---
the total number of dimensions.  The third argument is the inputs to the
function that the prior is over, :code:`X`.  The inputs are usually known and
present in the data, but they can also be PyMC3 random variables.

.. note::

  The :code:`n_points` argument is required because of how Theano and PyMC3
  handle the shape information of distributions.  For :code:`prior` or
  :code:`marginal_likelihood`, it is the number of rows in the inputs,
  :code:`X`.  For :code:`conditional`, it is the number of rows in the new
  inputs, :code:`X_new`.


The :code:`conditional` method creates the conditional, or predictive,
distribution over the latent function at arbitrary :math:$x_*$ input points,
:math:$f(x_*)$.  It can be called on any or all of the component GPs,
:code:`gp1`, :code:`gp2`, or :code:`gp3`.  To construct the conditional
distribution for :code:`gp3`, we write::

	f_star = gp3.conditional("f_star", n_newpoints, X_star)

To construct the conditional distribution of one of the component GPs,
:code:`gp1` or :code:`gp2`, we also need to include the original inputs
:code:`X` as an argument::

	f1_star = gp1.conditional("f1_start", n_newpoints, X_star, X=X)

The :code:`gp3` object keeps track of the inputs it used when :code:`prior` was
set.  Since the prior method of :code:`gp1` wasn't called, it needs to be
provided with the inputs :code:`X`.  In the same fashion as the prior,
:code:`f_star` and :code:`f1_star` are random variables that can now be used
like any other random variable in PyMC3.  

.. note::

  `gp.Latent` has a `prior` and a `conditional`, but not a `marginal_likelihood`,
  since that method doesn't make sense in this case.  Other GP implementations
  have `marginal_likelihood`, but not a `prior`, such as `gp.Marginal` and
  `gp.MarginalSparse`.  

There are examples demonstrating in more detail the usage
of GP functionality in PyMC3, including examples demonstrating the usage of the
different GP implementations.  Link to other GPs here in a note 


