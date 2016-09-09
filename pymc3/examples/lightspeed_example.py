import numpy as np
import pymc3 as pm

light_speed = np.array([28, 26, 33, 24, 34, -44, 27, 16, 40, -2, 29, 22, 24, 21, 25,
                        30, 23, 29, 31, 19, 24, 20, 36, 32, 36, 28, 25, 21, 28, 29,
                        37, 25, 28, 26, 30, 32, 36, 26, 30, 22, 36, 23, 27, 27, 28,
                        27, 31, 27, 26, 33, 26, 32, 32, 24, 39, 28, 24, 25, 32, 25,
                        29, 27, 28, 29, 16, 23])

model_1 = pm.Model()

with model_1:
    # priors as specified in stan model
    # mu = pm.Uniform('mu', lower = -tt.inf, upper= np.inf)
    # sigma = pm.Uniform('sigma', lower = 0, upper= np.inf)

    # using vague priors works
    mu = pm.Uniform('mu', lower=light_speed.std() / 1000.0,
                    upper=light_speed.std() * 1000.0)
    sigma = pm.Uniform('sigma', lower=light_speed.std() /
                       1000.0, upper=light_speed.std() * 1000.0)

    # define likelihood
    y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=light_speed)

    # inference fitting the model

    # I have to use slice because the following command
    # trace = pm.sample(5000)
    # produce the error
    # ValueError: Cannot construct a ufunc with more than 32 operands
    # (requested number were: inputs = 51 and outputs = 1)valueerror


def run(n=5000):
    with model_1:
        xstart = pm.find_MAP()
        xstep = pm.Slice()
        trace = pm.sample(5000, xstep, xstart,
                          random_seed=123, progressbar=True)

        pm.summary(trace)


if __name__ == '__main__':
    run()
