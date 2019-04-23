from numpy import array
from unittest import TestCase, main
from pymc3.distributions.discrete import DiscreteUniform, Poisson
from pymc3.model import Model
from pymc3.step_methods.metropolis import CategoricalGibbsMetropolis
from pymc3.sampling import sample
from pymc3.stats import summary
from theano import shared
import pytest


class TestDiscreteUniform(TestCase):

    def test_DiscreteUniform(self, lower=0, upper=200000000, obs=5000000, draws=20000):
        """Define and run MCMC model with using a DiscreteUniform distribution.
        This may be useful in place of a categorical distribution for a very large support
        (i.e. large n in example below)."""
        obs = shared(obs)
        with Model() as model2:

            x = DiscreteUniform('x', lower, upper - 1)
            sfs_obs = Poisson('sfs_obs', mu=x, observed=obs)

        with model2:

            step = CategoricalGibbsMetropolis([x])
            trace = sample(draws, tune=0, step=step)
        print(summary(trace, varnames=['x']))
        return trace

    def test_bad_lower(self):
        with pytest.raises(ValueError):
            self.test_DiscreteUniform(lower=1, upper=200000001, obs=5000000, draws=20000)


if __name__ == '__main__':
    main()

