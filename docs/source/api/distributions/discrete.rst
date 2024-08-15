********
Discrete
********

.. currentmodule:: pymc
.. autosummary::
   :toctree: generated
   :template: distribution.rst

   Bernoulli
   BetaBinomial
   Binomial
   Categorical
   DiscreteUniform
   DiscreteWeibull
   Geometric
   HyperGeometric
   NegativeBinomial
   OrderedLogistic
   OrderedProbit
   Poisson

.. note::

   **OrderedLogistic and OrderedProbit:**
   The `OrderedLogistic` and `OrderedProbit` distributions expect the observed values to be 0-based, i.e., they should range from `0` to `K-1`. Using 1-based indexing (like `1, 2, 3,...K`) can result in errors.
