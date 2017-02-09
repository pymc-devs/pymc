from nose.plugins.attrib import attr

from . import sampler_fixtures as sf


class NUTSUniform(sf.NutsFixture, sf.UniformFixture):
    n_samples = 10000
    tune = 1000
    burn = 1000
    chains = 4
    min_n_eff = 9000
    decimals = 2


class MetropolisUniform(sf.MetropolisFixture, sf.UniformFixture):
    n_samples = 50000
    tune = 10000
    burn = 10000
    chains = 4
    min_n_eff = 10000
    decimals = 2


class SliceUniform(sf.SliceFixture, sf.UniformFixture):
    n_samples = 10000
    tune = 1000
    burn = 1000
    chains = 4
    min_n_eff = 5000
    decimals = 2


class NUTSNormal(sf.NutsFixture, sf.NormalFixture):
    n_samples = 10000
    tune = 1000
    burn = 1000
    chains = 2
    min_n_eff = 10000
    decimals = 1


class NUTSBetaBinomial(sf.NutsFixture, sf.BetaBinomialFixture):
    n_samples = 10000
    tune = 1000
    burn = 1000
    chains = 2
    min_n_eff = 2000
    decimals = 1


@attr('extra')
class NUTSStudentT(sf.NutsFixture, sf.StudentTFixture):
    n_samples = 100000
    tune = 1000
    burn = 1000
    chains = 2
    min_n_eff = 5000
    decimals = 1


@attr('extra')
class NUTSNormalLong(sf.NutsFixture, sf.NormalFixture):
    n_samples = 500000
    tune = 5000
    burn = 5000
    chains = 2
    min_n_eff = 300000
    decimals = 3
