from __future__ import division
from theano.tensor import constant, flatten, zeros_like, ones_like, stack, concatenate, sum, prod, lt,gt, le, ge,eq, neq, switch, clip, where, and_, or_, abs_ 

from theano.tensor import exp,log,cos, sin, tan, cosh, sinh, tanh,sqr, sqrt,erf, erfinv, dot
from theano.tensor import maximum, minimum, sgn, ceil, floor

from theano.sandbox.linalg.ops import det, matrix_inverse, extract_diag,matrix_dot, trace
from theano.tensor.nnet import sigmoid
import theano
