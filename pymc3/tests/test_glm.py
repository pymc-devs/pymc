import numpy as np

from .helpers import SeededTest
from pymc3 import glm, Model, Uniform, Normal, find_MAP, Slice, sample


# Generate data
def generate_data(intercept, slope, size=700):
    x = np.linspace(-1, 1, size)
    y = intercept + x * slope
    return x, y


class TestGLM(SeededTest):
    @classmethod
    def setUpClass(cls):
        super(TestGLM, cls).setUpClass()
        cls.intercept = 1
        cls.slope = 3
        cls.sd = .05
        x_linear, cls.y_linear = generate_data(cls.intercept, cls.slope, size=1000)
        cls.y_linear += np.random.normal(size=1000, scale=cls.sd)
        cls.data_linear = dict(x=x_linear, y=cls.y_linear)

        x_logistic, y_logistic = generate_data(cls.intercept, cls.slope, size=3000)
        y_logistic = 1 / (1 + np.exp(-y_logistic))
        bern_trials = [np.random.binomial(1, i) for i in y_logistic]
        cls.data_logistic = dict(x=x_logistic, y=bern_trials)

    def test_linear_component(self):
        with Model() as model:
            y_est, _ = glm.linear_component('y ~ x', self.data_linear)
            sigma = Uniform('sigma', 0, 20)
            Normal('y_obs', mu=y_est, sd=sigma, observed=self.y_linear)
            start = find_MAP(vars=[sigma])
            step = Slice(model.vars)
            trace = sample(500, step, start, progressbar=False, random_seed=self.random_seed)

            self.assertAlmostEqual(np.mean(trace['Intercept']), self.intercept, 1)
            self.assertAlmostEqual(np.mean(trace['x']), self.slope, 1)
            self.assertAlmostEqual(np.mean(trace['sigma']), self.sd, 1)

    def test_glm(self):
        with Model() as model:
            glm.glm('y ~ x', self.data_linear)
            step = Slice(model.vars)
            trace = sample(500, step, progressbar=False, random_seed=self.random_seed)

            self.assertAlmostEqual(np.mean(trace['Intercept']), self.intercept, 1)
            self.assertAlmostEqual(np.mean(trace['x']), self.slope, 1)
            self.assertAlmostEqual(np.mean(trace['sd']), self.sd, 1)

    def test_glm_link_func(self):
        with Model() as model:
            glm.glm('y ~ x', self.data_logistic,
                    family=glm.families.Binomial(link=glm.families.logit))
            step = Slice(model.vars)
            trace = sample(1000, step, progressbar=False, random_seed=self.random_seed)

            self.assertAlmostEqual(np.mean(trace['Intercept']), self.intercept, 1)
            self.assertAlmostEqual(np.mean(trace['x']), self.slope, 1)

    def test_more_than_one_glm_is_ok(self):
        with Model():
            glm.glm('y ~ x', self.data_logistic,
                    family=glm.families.Binomial(link=glm.families.logit),
                    name='glm1')
            glm.glm('y ~ x', self.data_logistic,
                    family=glm.families.Binomial(link=glm.families.logit),
                    name='glm2')

    def test_from_xy(self):
        with Model():
            glm.glm.from_xy(
                self.data_logistic['x'],
                self.data_logistic['y'],
                family=glm.families.Binomial(link=glm.families.logit),
                name='glm1')
