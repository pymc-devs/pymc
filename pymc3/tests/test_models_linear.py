import numpy as np
import pandas as pd
from .helpers import SeededTest
from pymc3 import Model, Uniform, Normal, find_MAP, Slice, sample
from pymc3.models.linear import LinearComponent


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
        cls.data_linear = dict(x=pd.DataFrame(x_linear, columns=['x']), y=cls.y_linear)

        x_logistic, y_logistic = generate_data(cls.intercept, cls.slope, size=3000)
        y_logistic = 1 / (1 + np.exp(-y_logistic))
        bern_trials = [np.random.binomial(1, i) for i in y_logistic]
        cls.data_logistic = dict(x=pd.DataFrame(x_logistic, columns=['x']), y=bern_trials)

    def test_linear_component(self):
        with Model() as model:
            lm = LinearComponent(
                'lm',
                self.data_linear['x'],
                self.data_linear['y']
            )
            sigma = Uniform('sigma', 0, 20)
            Normal('y_obs', mu=lm.y_est, sd=sigma, observed=self.y_linear)
            start = find_MAP(vars=[sigma])
            step = Slice(model.vars)
            trace = sample(500, step, start, progressbar=False, random_seed=self.random_seed)

            self.assertAlmostEqual(np.mean(trace['lm_Intercept']), self.intercept, 1)
            self.assertAlmostEqual(np.mean(trace['lm_x']), self.slope, 1)
            self.assertAlmostEqual(np.mean(trace['sigma']), self.sd, 1)

    def test_linear_component_from_formula(self):
        with Model() as model:
            lm = LinearComponent.from_formula('', 'y ~ x', self.data_linear)
            sigma = Uniform('sigma', 0, 20)
            Normal('y_obs', mu=lm.y_est, sd=sigma, observed=self.y_linear)
            start = find_MAP(vars=[sigma])
            step = Slice(model.vars)
            trace = sample(500, step, start, progressbar=False, random_seed=self.random_seed)

            self.assertAlmostEqual(np.mean(trace['Intercept']), self.intercept, 1)
            self.assertAlmostEqual(np.mean(trace['x']), self.slope, 1)
            self.assertAlmostEqual(np.mean(trace['sigma']), self.sd, 1)