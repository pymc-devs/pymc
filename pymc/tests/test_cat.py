from __future__ import division
import unittest

import itertools
from .checks import *
from pymc import *
from numpy import array, inf
import numpy
from numpy.linalg import inv

from scipy import integrate
import scipy.stats.distributions  as sp
import scipy.stats

from .knownfailure import *

from test_distributions import Domain, pymc_matches_scipy, Simplex, Vector

def categorical_logpdf(value, p):
    if value >= 0 and value <= len(p):
        return log(p[value]).sum()
    else:
        return -inf

def test_categorical():
    for n in [8, 12, 20]:
        yield check_categorical, n

def check_categorical(n):
    pymc_matches_scipy(
        Categorical, Domain(range(2, n, 2), 'int64'), {'p': Simplex(n)},
        lambda value, p: categorical_logpdf(value, p)
        )
