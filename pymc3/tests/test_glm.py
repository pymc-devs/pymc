import unittest
from nose import SkipTest
import numpy as np
import sys
try:
    import statsmodels.api as sm
except ImportError:
    raise SkipTest("Test requires statsmodels.")

from pymc3.examples import glm_linear, glm_robust


np.random.seed(1)
# Generate data
true_intercept = 0
true_slope = 3


def generate_data(size=700):
    x = np.linspace(-1, 1, size)
    y = true_intercept + x * true_slope
    return x, y

true_sd = .05
x_linear, y_linear = generate_data(size=1000)
y_linear += np.random.normal(size=1000, scale=true_sd)
data_linear = dict(x=x_linear, y=y_linear)

x_logistic, y_logistic = generate_data(size=3000)
y_logistic = 1 / (1 + np.exp(-y_logistic))
bern_trials = [np.random.binomial(1, i) for i in y_logistic]
data_logistic = dict(x=x_logistic, y=bern_trials)


class TestGLM(unittest.TestCase):

    @unittest.skip("Fails only on travis. Investigate")
    def test_linear_component(self):
        with Model() as model:
            y_est, coeffs = glm.linear_component('y ~ x', data_linear)
            for coeff, true_val in zip(coeffs, [true_intercept, true_slope]):
                self.assertAlmostEqual(coeff.tag.test_value, true_val, 1)
            sigma = Uniform('sigma', 0, 20)
            y_obs = Normal('y_obs', mu=y_est, sd=sigma, observed=y_linear)
            start = find_MAP(vars=[sigma])
            step = Slice(model.vars)
            trace = sample(2000, step, start, progressbar=False)

            self.assertAlmostEqual(
                np.mean(trace['Intercept']), true_intercept, 1)
            self.assertAlmostEqual(np.mean(trace['x']), true_slope, 1)
            self.assertAlmostEqual(np.mean(trace['sigma']), true_sd, 1)

    @unittest.skip("Fails only on travis. Investigate")
    def test_glm(self):
        with Model() as model:
            vars = glm.glm('y ~ x', data_linear)
            for coeff, true_val in zip(vars[1:], [true_intercept, true_slope, true_sd]):
                self.assertAlmostEqual(coeff.tag.test_value, true_val, 1)
            step = Slice(model.vars)
            trace = sample(2000, step, progressbar=False)

            self.assertAlmostEqual(
                np.mean(trace['Intercept']), true_intercept, 1)
            self.assertAlmostEqual(np.mean(trace['x']), true_slope, 1)
            self.assertAlmostEqual(np.mean(trace['sigma']), true_sd, 1)

    @unittest.skip("Was an error, then a fail, now a skip.")
    def test_glm_link_func(self):
        with Model() as model:
            vars = glm.glm('y ~ x', data_logistic,
                           family=glm.families.Binomial(link=glm.families.logit))

            for coeff, true_val in zip(vars[1:], [true_intercept, true_slope]):
                self.assertAlmostEqual(coeff.tag.test_value, true_val, 0)
            step = Slice(model.vars)
            trace = sample(2000, step, progressbar=False)

            self.assertAlmostEqual(
                np.mean(trace['Intercept']), true_intercept, 1)
            self.assertAlmostEqual(np.mean(trace['x']), true_slope, 0)
