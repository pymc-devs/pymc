import pymc3 as pm
import numpy as np
from ..step_methods import ATMCMC, ATMIP_sample
import theano.tensor as tt
from matplotlib import pylab as plt

test_folder = ('ATMIP_TEST')

n_chains = 500
n_steps = 100
tune_interval = 25
njobs = 1

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

with pm.Model() as ATMIP_test:
    X = pm.Uniform('X',
                   shape=n,
                   lower=-2. * np.ones_like(mu1),
                   upper=2. * np.ones_like(mu1),
                   testval=-1. * np.ones_like(mu1),
                   transform=None)
    like = pm.Deterministic('like', two_gaussians(X))
    llk = pm.Potential('like', like)

with ATMIP_test:
    step = ATMCMC(n_chains=n_chains, tune_interval=tune_interval,
                  likelihood_name=ATMIP_test.deterministics[0].name)

trcs = ATMIP_sample(
    n_steps=n_steps,
    step=step,
    njobs=njobs,
    progressbar=True,
    trace=test_folder,
    model=ATMIP_test)

pm.summary(trcs)
Pltr = pm.traceplot(trcs, combined=True)
plt.show(Pltr[0][0])
