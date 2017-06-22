import pymc3 as pm
import numpy as np
from pymc3.step_methods import smc
from pymc3.backends.smc_text import TextStage
import pytest
from tempfile import mkdtemp
import shutil
import theano.tensor as tt
import theano

from .helpers import SeededTest


@pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32")
class TestSMC(SeededTest):

    def setup_class(self):
        super(TestSMC, self).setup_class()
        self.test_folder = mkdtemp(prefix='ATMIP_TEST')

        self.n_chains = 300
        self.n_steps = 100
        self.tune_interval = 25

        n = 4

        mu1 = np.ones(n) * (1. / 2)
        mu2 = -mu1

        stdev = 0.1
        sigma = np.power(stdev, 2) * np.eye(n)
        isigma = np.linalg.inv(sigma)
        dsigma = np.linalg.det(sigma)

        w1 = stdev
        w2 = (1 - stdev)

        def two_gaussians(x):
            log_like1 = - 0.5 * n * tt.log(2 * np.pi) \
                        - 0.5 * tt.log(dsigma) \
                        - 0.5 * (x - mu1).T.dot(isigma).dot(x - mu1)
            log_like2 = - 0.5 * n * tt.log(2 * np.pi) \
                        - 0.5 * tt.log(dsigma) \
                        - 0.5 * (x - mu2).T.dot(isigma).dot(x - mu2)
            return tt.log(w1 * tt.exp(log_like1) + w2 * tt.exp(log_like2))

        with pm.Model() as self.ATMIP_test:
            X = pm.Uniform('X',
                           shape=n,
                           lower=-2. * np.ones_like(mu1),
                           upper=2. * np.ones_like(mu1),
                           testval=-1. * np.ones_like(mu1),
                           transform=None)
            like = pm.Deterministic('like', two_gaussians(X))
            llk = pm.Potential('like_potential', like)

        self.muref = mu1

    @pytest.mark.parametrize('n_jobs', [1, 2])
    def test_sample_n_core(self, n_jobs):

        def last_sample(x):
            return x[(self.n_steps - 1)::self.n_steps]

        step = smc.SMC(
            n_chains=self.n_chains,
            tune_interval=self.tune_interval,
            model=self.ATMIP_test,
            likelihood_name=self.ATMIP_test.deterministics[0].name)

        mtrace = smc.ATMIP_sample(
            n_steps=self.n_steps,
            step=step,
            n_jobs=n_jobs,
            progressbar=True,
            homepath=self.test_folder,
            model=self.ATMIP_test,
            rm_flag=True)

        d = mtrace.get_values('X', combine=True, squeeze=True)
        x = last_sample(d)
        mu1d = np.abs(x).mean(axis=0)
        np.testing.assert_allclose(self.muref, mu1d, rtol=0., atol=0.03)

    def test_stage_handler(self):
        stage_number = -1
        stage_handler = TextStage(self.test_folder)

        step = stage_handler.load_atmip_params(stage_number, model=self.ATMIP_test)
        assert step.stage == stage_number

        corrupted_chains = stage_handler.recover_existing_results(
            stage_number, self.n_steps, step, n_jobs=1, model=self.ATMIP_test)
        assert len(corrupted_chains) == 0

    def teardown_class(self):
        shutil.rmtree(self.test_folder)
