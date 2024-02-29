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
   logaddexp
   logsumexp


Functions exposed in pymc.math
------------------------------

.. automodule:: pymc.math
.. autosummary::
   :toctree: generated/

   abs
   prod
   dot
   eq
   neq
   ge
   gt
   le
   lt
   exp
   log
   sgn
   sqr
   sqrt
   sum
   ceil
   floor
   sin
   sinh
   arcsin
   arcsinh
   cos
   cosh
   arccos
   arccosh
   tan
   tanh
   arctan
   arctanh
   cumprod
   cumsum
   matmul
   and_
   broadcast_to
   clip
   concatenate
   flatten
   or_
   stack
   switch
   where
   flatten_list
   constant
   max
   maximum
   mean
   min
   minimum
   round
   tround
   erf
   erfc
   erfcinv
   erfinv
   log1pexp
   log1mexp
   log1mexp_numpy
   logaddexp
   logsumexp
   logdiffexp
   logdiffexp_numpy
   logit
   invlogit
   probit
   invprobit
   sigmoid
   softmax
   log_softmax
   logbern
   full
   full_like
   ones
   ones_like
   zeros
   zeros_like
   kronecker
   cartesian
   kron_dot
   kron_solve_lower
   kron_solve_upper
   kron_diag
   flat_outer
   expand_packed_triangular
   batched_diag
   block_diagonal
   matrix_inverse
   logdet
