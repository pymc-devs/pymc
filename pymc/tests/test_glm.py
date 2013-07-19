import unittest

from pymc import *

from pymc.examples import glm_linear, glm_robust

try:
    import statsmodels.api as sm
    sm_import = True
except ImportError:
    sm_import = False

np.random.seed(1)
# Generate data
true_intercept = 0
true_slope = 3

def generate_data(size=700):
    x = np.linspace(-1, 1, size)
    y = true_intercept + x*true_slope
    return x, y

true_sd = .05
x_linear, y_linear = generate_data(size=1000)
y_linear += np.random.normal(size=1000, scale=true_sd)
data_linear = dict(x=x_linear, y=y_linear)

x_logistic, y_logistic = generate_data(size=3000)
y_logistic = 1 / (1+np.exp(-y_logistic))
bern_trials = [np.random.binomial(1, i) for i in y_logistic]
data_logistic = dict(x=x_logistic, y=bern_trials)

class TestGLM(unittest.TestCase):
    def test_linear_component(self):
        with Model() as model:
            y_est, coeffs = glm.linear_component('y ~ x', data_linear, init=sm_import)
            if sm_import:
                for coeff, true_val in zip(coeffs, [true_intercept, true_slope]):
                    self.assertAlmostEqual(coeff.tag.test_value, true_val, 1)
            sigma = Uniform('sigma', 0, 20)
            y_obs = Normal('y_obs', mu=y_est, sd=sigma, observed=y_linear)
            if sm_import:
                start = find_MAP(vars=[sigma])
            else:
                start = find_MAP()

            step = Slice(model.vars)
            trace = sample(2000, step, start, progressbar=False)

            self.assertAlmostEqual(np.mean(trace.samples['Intercept'].value), true_intercept, 1)
            self.assertAlmostEqual(np.mean(trace.samples['x'].value), true_slope, 1)
            self.assertAlmostEqual(np.mean(trace.samples['sigma'].value), true_sd, 1)

    def test_glm(self):
        if not sm_import:
            return
        with Model() as model:
            vars = glm.glm('y ~ x', data_linear)
            for coeff, true_val in zip(vars[1:], [true_intercept, true_slope, true_sd]):
                self.assertAlmostEqual(coeff.tag.test_value, true_val, 1)
            step = Slice(model.vars)
            trace = sample(2000, step, progressbar=False)

            self.assertAlmostEqual(np.mean(trace.samples['Intercept'].value), true_intercept, 1)
            self.assertAlmostEqual(np.mean(trace.samples['x'].value), true_slope, 1)
            self.assertAlmostEqual(np.mean(trace.samples['sigma'].value), true_sd, 1)

    def test_glm_link_func(self):
        if not sm_import:
            return

        with Model() as model:
            vars = glm.glm('y ~ x', data_logistic,
                           family=glm.families.Binomial(link=glm.links.Logit))

            for coeff, true_val in zip(vars[1:], [true_intercept, true_slope]):
                self.assertAlmostEqual(coeff.tag.test_value, true_val, 0)
            step = Slice(model.vars)
            trace = sample(2000, step, progressbar=False)

            self.assertAlmostEqual(np.mean(trace.samples['Intercept'].value), true_intercept, 1)
            self.assertAlmostEqual(np.mean(trace.samples['x'].value), true_slope, 0)
