.. _api:

*************
API Reference
*************

Distributions
-------------

Continuous
^^^^^^^^^^

.. currentmodule:: pymc3.distributions.continuous
.. autosummary::

   Uniform
   Flat
   Normal
   Beta
   Exponential
   Laplace
   StudentT
   Cauchy
   HalfCauchy
   Gamma
   Weibull
   StudentTpos
   Lognormal
   ChiSquared
   HalfNormal
   Wald
   Pareto
   InverseGamma
   ExGaussian

.. automodule:: pymc3.distributions.continuous
   :members:

Discrete
^^^^^^^^

.. currentmodule:: pymc3.distributions.discrete
.. autosummary::

   Binomial
   BetaBinomial
   Bernoulli
   Poisson
   NegativeBinomial
   ConstantDist
   ZeroInflatedPoisson
   DiscreteUniform
   Geometric
   Categorical

.. automodule:: pymc3.distributions.discrete
   :members:

Multivariate
^^^^^^^^^^^^

.. currentmodule:: pymc3.distributions.multivariate
.. autosummary::

   MvNormal
   Wishart
   LKJCorr
   Multinomial
   Dirichlet

.. automodule:: pymc3.distributions.multivariate
   :members:

Mixture
^^^^^^^

.. currentmodule:: pymc3.distributions.mixture
.. autosummary::
   Mixture
   NormalMixture

.. automodule:: pymc3.distributions.mixture
   :members:

Plots
-----

.. currentmodule:: pymc3.plots

.. automodule:: pymc3.plots
   :members:

Stats
-----

.. currentmodule:: pymc3.stats

.. automodule:: pymc3.stats
   :members:

Diagnostics
-----------

.. currentmodule:: pymc3.diagnostics

.. automodule:: pymc3.diagnostics
   :members:



Inference
---------

Sampling
^^^^^^^^

.. currentmodule:: pymc3.sampling

.. automodule:: pymc3.sampling
   :members:


Step-methods
^^^^^^^^^^^^

NUTS
""""

.. currentmodule:: pymc3.step_methods.nuts

.. automodule:: pymc3.step_methods.nuts
   :members:

Metropolis
""""""""""

.. currentmodule:: pymc3.step_methods.metropolis

.. automodule:: pymc3.step_methods.metropolis
   :members:

Slice
"""""

.. currentmodule:: pymc3.step_methods.slicer

.. automodule:: pymc3.step_methods.slicer
   :members:

Hamiltonian Monte Carlo
"""""""""""""""""""""""

.. currentmodule:: pymc3.step_methods.hmc

.. automodule:: pymc3.step_methods.hmc
   :members:


Variational
^^^^^^^^^^^

ADVI
""""

.. currentmodule:: pymc3.variational.advi

.. automodule:: pymc3.variational.advi
   :members:

ADVI minibatch
""""""""""""""

.. currentmodule:: pymc3.variational.advi_minibatch

.. automodule:: pymc3.variational.advi_minibatch
   :members:

Backends
--------

.. currentmodule:: pymc3.backends

.. automodule:: pymc3.backends
   :members:

ndarray
^^^^^^^

.. currentmodule:: pymc3.backends.ndarray

.. automodule:: pymc3.backends.ndarray
   :members:

sqlite
^^^^^^

.. currentmodule:: pymc3.backends.sqlite

.. automodule:: pymc3.backends.sqlite
   :members:

text
^^^^

.. currentmodule:: pymc3.backends.text

.. automodule:: pymc3.backends.text
   :members:

tracetab
^^^^^^^^

.. currentmodule:: pymc3.backends.tracetab

.. automodule:: pymc3.backends.tracetab
   :members:


GLM
---

.. currentmodule:: pymc3.glm.glm

.. automodule:: pymc3.glm.glm
   :members:


GP
--

.. currentmodule:: pymc3.gp.cov
.. autosummary::

   ExpQuad
   RatQuad
   Matern32
   Matern52
   Exponential
   Cosine
   Linear
   Polynomial
   WarpedInput

.. automodule:: pymc3.gp.cov
   :members:

Math
----

This submodule contains various mathematical functions. Most of them
are imported directly from theano.tensor (see there for more
details). Doing any kind of math with PyMC3 random variables, or
defining custom likelihoods or priors requires you to use these theano
expressions rather than NumPy or Python code.

.. currentmodule:: pymc3.math
.. autosummary::
   dot
   constant
   flatten
   zeros_like
   ones_like
   stack
   concatenate
   sum
   prod
   lt
   gt
   le
   ge
   eq
   neq
   switch
   clip
   where
   and_
   or_
   abs_
   exp
   log
   cos
   sin
   tan
   cosh
   sinh
   tanh
   sqr
   sqrt
   erf
   erfinv
   dot
   maximum
   minimum
   sgn
   ceil
   floor
   det
   matrix_inverse
   extract_diag
   matrix_dot
   trace
   sigmoid
   logsumexp
   invlogit
   logit

.. automodule:: pymc3.math
   :members:
