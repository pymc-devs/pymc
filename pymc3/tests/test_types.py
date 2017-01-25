import os
from copy import copy

import unittest
import theano

from .checks import close_to
from .models import simple_categorical, mv_simple, mv_simple_discrete, simple_2model
from .helpers import SeededTest
from pymc3 import df_summary, traceplot
from pymc3.sampling import assign_step_methods, sample
from pymc3.model import Model
from pymc3.step_methods import (NUTS, BinaryGibbsMetropolis, CategoricalGibbsMetropolis,
                                Metropolis, Slice, CompoundStep,
                                MultivariateNormalProposal, HamiltonianMC)
from pymc3.distributions import Binomial, Normal, Bernoulli, Categorical, InverseGamma

from numpy.testing import assert_array_almost_equal
import numpy as np
from tqdm import tqdm
from scipy import stats


class TestType(unittest.TestCase):
    samplers = (Metropolis, Slice, HamiltonianMC, NUTS)

    def setUp(self):
        # save theano config object
        self.theano_config = copy(theano.config)

    def tearDown(self):
        # restore theano config
        theano.config = self.theano_config

    def test_float64(self):
        theano.config.floatX = 'float64'
        theano.config.warn_float64 = 'ignore'

        with Model() as model:
            x = Normal('x', testval=np.array(1., dtype='float64'))
            obs = Normal('obs', mu=x, sd=1., observed=np.random.randn(5))

        assert x.dtype == 'float64'
        assert obs.dtype == 'float64'

        for sampler in self.samplers:
            with model:
                sample(10, sampler())

    def test_float32(self):
        theano.config.floatX = 'float32'
        theano.config.warn_float64 = 'warn'

        with Model() as model:
            x = Normal('x', testval=np.array(1., dtype='float32'))
            obs = Normal('obs', mu=x, sd=1., observed=np.random.randn(5).astype('float32'))

        assert x.dtype == 'float32'
        assert obs.dtype == 'float32'

        for sampler in self.samplers:
            with model:
                sample(10, sampler())
