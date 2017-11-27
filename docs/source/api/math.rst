====
Math
====

This submodule contains various mathematical functions. Most of them are imported directly 
from theano.tensor (see there for more details). Doing any kind of math with PyMC3 random 
variables, or defining custom likelihoods or priors requires you to use these theano
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
