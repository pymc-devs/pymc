====
Math
====

This submodule contains various mathematical functions. Most of them are imported directly
from pytensor.tensor (see there for more details). Doing any kind of math with PyMC random
variables, or defining custom likelihoods or priors requires you to use these PyTensor
expressions rather than NumPy or Python code.

.. currentmodule:: pymc

Functions exposed in pymc namespace
-----------------------------------
.. autosummary::
   :toctree: generated/

   expand_packed_triangular
   logit
   invlogit
   probit
   invprobit
   logsumexp

Functions exposed in pymc.math
------------------------------

.. automodule:: pymc.math
.. autosummary::
   :toctree: generated/

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
   abs
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
   matrix_inverse
   sigmoid
   logsumexp
   invlogit
   logit
