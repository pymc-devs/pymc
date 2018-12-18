from copy import copy

import theano

from pymc3.sampling import sample
from pymc3.model import Model
from pymc3.step_methods import NUTS, Metropolis, Slice, HamiltonianMC
from pymc3.distributions import Normal
from pymc3.theanof import change_flags

import numpy as np


class TestType:
    samplers = (Metropolis, Slice, HamiltonianMC, NUTS)

    def setup_method(self):
        # save theano config object
        self.theano_config = copy(theano.config)

    def teardown_method(self):
        # restore theano config
        theano.config = self.theano_config

    @change_flags({'floatX': 'float64', 'warn_float64': 'ignore'})
    def test_float64(self):
        with Model() as model:
            x = Normal('x', testval=np.array(1., dtype='float64'))
            obs = Normal('obs', mu=x, sigma=1., observed=np.random.randn(5))

        assert x.dtype == 'float64'
        assert obs.dtype == 'float64'

        for sampler in self.samplers:
            with model:
                sample(10, sampler())

    @change_flags({'floatX': 'float32', 'warn_float64': 'warn'})
    def test_float32(self):
        with Model() as model:
            x = Normal('x', testval=np.array(1., dtype='float32'))
            obs = Normal('obs', mu=x, sigma=1., observed=np.random.randn(5).astype('float32'))

        assert x.dtype == 'float32'
        assert obs.dtype == 'float32'

        for sampler in self.samplers:
            with model:
                sample(10, sampler())
