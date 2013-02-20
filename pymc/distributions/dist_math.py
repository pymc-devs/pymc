'''
Created on Mar 7, 2011

@author: johnsalvatier
'''
import theano.tensor as t
from theano.tensor import switch, log, eq, neq, lt, gt, le, ge, zeros_like, cast, round

from numpy import pi, inf
from special import gammaln, factln

from theano.printing import Print
