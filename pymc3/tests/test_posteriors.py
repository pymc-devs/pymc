from nose.plugins.attrib import attr

from . import sampler_fixtures as sf


class NUTSUniform(sf.NutsFixture, sf.UniformFixture):
    n_samples = 10000
    tune = 1000
    burn = 1000
    chains = 4
    min_n_eff = 9000
    rtol = 0.1
    atol = 0.05


class MetropolisUniform(sf.MetropolisFixture, sf.UniformFixture):
    n_samples = 50000
    tune = 10000
    burn = 10000
    chains = 4
    min_n_eff = 10000
    rtol = 0.1
    atol = 0.05


class SliceUniform(sf.SliceFixture, sf.UniformFixture):
    n_samples = 10000
    tune = 1000
    burn = 1000
    chains = 4
    min_n_eff = 5000
    rtol = 0.1
    atol = 0.05


@attr('extra')
class NUTSUniform2(NUTSUniform):
    step_args = {'target_accept': 0.95, 'integrator': 'two-stage'}


class NUTSUniform3(NUTSUniform):
    step_args = {'target_accept': 0.80, 'integrator': 'two-stage'}


@attr('extra')
class NUTSUniform4(NUTSUniform):
    step_args = {'target_accept': 0.95, 'integrator': 'three-stage'}


class NUTSUniform5(NUTSUniform):
    step_args = {'target_accept': 0.80, 'integrator': 'three-stage'}


class NUTSNormal(sf.NutsFixture, sf.NormalFixture):
    n_samples = 10000
    tune = 1000
    burn = 1000
    chains = 2
    min_n_eff = 10000
    rtol = 0.1
    atol = 0.05


class NUTSBetaBinomial(sf.NutsFixture, sf.BetaBinomialFixture):
    n_samples = 2000
    ks_thin = 5
    tune = 1000
    burn = 1000
    chains = 2
    min_n_eff = 400


@attr('extra')
class NUTSStudentT(sf.NutsFixture, sf.StudentTFixture):
    n_samples = 100000
    tune = 1000
    burn = 1000
    chains = 2
    min_n_eff = 5000
    rtol = 0.1
    atol = 0.05


@attr('extra')
class NUTSNormalLong(sf.NutsFixture, sf.NormalFixture):
    n_samples = 500000
    tune = 5000
    burn = 5000
    chains = 2
    min_n_eff = 300000
    rtol = 0.01
    atol = 0.001
