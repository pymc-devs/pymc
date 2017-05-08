import pytest
from . import sampler_fixtures as sf


class TestNUTSUniform(sf.NutsFixture, sf.UniformFixture):
    n_samples = 10000
    tune = 1000
    burn = 1000
    chains = 4
    min_n_eff = 9000
    rtol = 0.1
    atol = 0.05


class TestMetropolisUniform(sf.MetropolisFixture, sf.UniformFixture):
    n_samples = 50000
    tune = 10000
    burn = 0
    chains = 4
    min_n_eff = 10000
    rtol = 0.1
    atol = 0.05


class TestSliceUniform(sf.SliceFixture, sf.UniformFixture):
    n_samples = 10000
    tune = 1000
    burn = 0
    chains = 4
    min_n_eff = 5000
    rtol = 0.1
    atol = 0.05


class TestNUTSUniform2(TestNUTSUniform):
    step_args = {'target_accept': 0.95, 'integrator': 'two-stage'}


class TestNUTSUniform3(TestNUTSUniform):
    step_args = {'target_accept': 0.80, 'integrator': 'two-stage'}


class TestNUTSUniform4(TestNUTSUniform):
    step_args = {'target_accept': 0.95, 'integrator': 'three-stage'}


class TestNUTSUniform5(TestNUTSUniform):
    step_args = {'target_accept': 0.80, 'integrator': 'three-stage'}


class TestNUTSNormal(sf.NutsFixture, sf.NormalFixture):
    n_samples = 10000
    tune = 1000
    burn = 0
    chains = 2
    min_n_eff = 10000
    rtol = 0.1
    atol = 0.05


class TestNUTSBetaBinomial(sf.NutsFixture, sf.BetaBinomialFixture):
    n_samples = 2000
    ks_thin = 5
    tune = 1000
    burn = 0
    chains = 2
    min_n_eff = 400


class TestNUTSStudentT(sf.NutsFixture, sf.StudentTFixture):
    n_samples = 100000
    tune = 1000
    burn = 0
    chains = 2
    min_n_eff = 5000
    rtol = 0.1
    atol = 0.05


@pytest.mark.skip('Takes too long to run')
class TestNUTSNormalLong(sf.NutsFixture, sf.NormalFixture):
    n_samples = 500000
    tune = 5000
    burn = 0
    chains = 2
    min_n_eff = 300000
    rtol = 0.01
    atol = 0.001


class TestNUTSLKJCholeskyCov(sf.NutsFixture, sf.LKJCholeskyCovFixture):
    n_samples = 2000
    tune = 1000
    burn = 0
    chains = 2
    min_n_eff = 200
