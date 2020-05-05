#   Copyright 2020 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
from abc import ABC

import matplotlib
import numpy as np
from numpy import interp
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
import pytest
import theano
from pymc3.theanof import floatX
from packaging import version

from .helpers import SeededTest

if version.parse(matplotlib.__version__) < version.parse('3.3'):
    matplotlib.use('Agg', warn=False)
else:
    matplotlib.use('Agg')


def get_city_data():
    """Helper to get city data"""
    data = pd.read_csv(pm.get_data('srrs2.dat'))
    cty_data = pd.read_csv(pm.get_data('cty.dat'))

    data = data[data.state == 'MN']

    data['fips'] = data.stfips * 1000 + data.cntyfips
    cty_data['fips'] = cty_data.stfips * 1000 + cty_data.ctfips
    data['lradon'] = np.log(np.where(data.activity == 0, .1, data.activity))
    data = data.merge(cty_data, 'inner', on='fips')

    unique = data[['fips']].drop_duplicates()
    unique['group'] = np.arange(len(unique))
    unique.set_index('fips')
    return data.merge(unique, 'inner', on='fips')


class TestARM5_4(SeededTest):
    def build_model(self):
        data = pd.read_csv(pm.get_data('wells.dat'),
                           delimiter=' ', index_col='id',
                           dtype={'switch': np.int8})
        data.dist /= 100
        data.educ /= 4
        col = data.columns
        P = data[col[1:]]
        P -= P.mean()
        P['1'] = 1

        with pm.Model() as model:
            effects = pm.Normal('effects', mu=0, sigma=100, shape=len(P.columns))
            logit_p = tt.dot(floatX(np.array(P)), effects)
            pm.Bernoulli('s', logit_p=logit_p, observed=floatX(data.switch.values))
        return model

    def test_run(self):
        model = self.build_model()
        with model:
            pm.sample(50, tune=50)


class TestARM12_6(SeededTest):
    def build_model(self):
        data = get_city_data()

        self.obs_means = data.groupby('fips').lradon.mean().to_numpy()

        lradon = data.lradon.to_numpy()
        floor = data.floor.to_numpy()
        group = data.group.to_numpy()

        with pm.Model() as model:
            groupmean = pm.Normal('groupmean', 0, 10. ** -2.)
            groupsd = pm.Uniform('groupsd', 0, 10.)
            sd = pm.Uniform('sd', 0, 10.)
            floor_m = pm.Normal('floor_m', 0, 5. ** -2.)
            means = pm.Normal('means', groupmean, groupsd ** -2., shape=len(self.obs_means))
            pm.Normal('lr', floor * floor_m + means[group], sd ** -2., observed=lradon)
        return model

    def too_slow(self):
        model = self.build_model()
        start = {'groupmean': self.obs_means.mean(),
                 'groupsd_interval__': 0,
                 'sd_interval__': 0,
                 'means': self.obs_means,
                 'floor_m': 0.,
                 }
        with model:
            start = pm.find_MAP(start=start,
                                vars=[model['groupmean'], model['sd_interval__'], model['floor_m']])
            step = pm.NUTS(model.vars, scaling=start)
            pm.sample(50, step=step, start=start)


class TestARM12_6Uranium(SeededTest):
    def build_model(self):
        data = get_city_data()
        self.obs_means = data.groupby('fips').lradon.mean()

        lradon = data.lradon.to_numpy()
        floor = data.floor.to_numpy()
        group = data.group.to_numpy()
        ufull = data.Uppm.to_numpy()

        with pm.Model() as model:
            groupmean = pm.Normal('groupmean', 0, 10. ** -2.)
            groupsd = pm.Uniform('groupsd', 0, 10.)
            sd = pm.Uniform('sd', 0, 10.)
            floor_m = pm.Normal('floor_m', 0, 5. ** -2.)
            u_m = pm.Normal('u_m', 0, 5. ** -2)
            means = pm.Normal('means', groupmean, groupsd ** -2., shape=len(self.obs_means))
            pm.Normal('lr', floor * floor_m + means[group] + ufull * u_m, sd ** - 2.,
                      observed=lradon)
        return model

    def too_slow(self):
        model = self.build_model()
        with model:
            start = pm.Point({
                'groupmean': self.obs_means.mean(),
                'groupsd_interval__': 0,
                'sd_interval__': 0,
                'means': np.array(self.obs_means),
                'u_m': np.array([.72]),
                'floor_m': 0.,
            })

            start = pm.find_MAP(start, model.vars[:-1])
            H = model.fastd2logp()
            h = np.diag(H(start))

            step = pm.HamiltonianMC(model.vars, h)
            pm.sample(50, step=step, start=start)


def build_disaster_model(masked=False):
    disasters_data = np.array([4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                               3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                               2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
                               1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                               0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                               3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                               0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])
    if masked:
        disasters_data[[23, 68]] = -1
        disasters_data = np.ma.masked_values(disasters_data, value=-1)
    years = len(disasters_data)

    with pm.Model() as model:
        # Prior for distribution of switchpoint location
        switchpoint = pm.DiscreteUniform('switchpoint', lower=0, upper=years)
        # Priors for pre- and post-switch mean number of disasters
        early_mean = pm.Exponential('early_mean', lam=1.)
        late_mean = pm.Exponential('late_mean', lam=1.)
        # Allocate appropriate Poisson rates to years before and after current
        # switchpoint location
        idx = np.arange(years)
        rate = tt.switch(switchpoint >= idx, early_mean, late_mean)
        # Data likelihood
        pm.Poisson('disasters', rate, observed=disasters_data)
    return model


@pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32")
class TestDisasterModel(SeededTest):
    # Time series of recorded coal mining disasters in the UK from 1851 to 1962
    def test_disaster_model(self):
        model = build_disaster_model(masked=False)
        with model:
            # Initial values for stochastic nodes
            start = {'early_mean': 2., 'late_mean': 3.}
            # Use slice sampler for means (other variables auto-selected)
            step = pm.Slice([model.early_mean_log__, model.late_mean_log__])
            tr = pm.sample(500, tune=50, start=start, step=step, chains=2)
            pm.summary(tr)

    def test_disaster_model_missing(self):
        model = build_disaster_model(masked=True)
        with model:
            # Initial values for stochastic nodes
            start = {'early_mean': 2., 'late_mean': 3.}
            # Use slice sampler for means (other variables auto-selected)
            step = pm.Slice([model.early_mean_log__, model.late_mean_log__])
            tr = pm.sample(500, tune=50, start=start, step=step, chains=2)
            pm.summary(tr)


class TestGLMLinear(SeededTest):
    def build_model(self):
        size = 50
        true_intercept = 1
        true_slope = 2
        self.x = np.linspace(0, 1, size)
        self.y = true_intercept + self.x * true_slope + np.random.normal(scale=.5, size=size)
        data = dict(x=self.x, y=self.y)
        with pm.Model() as model:
            pm.GLM.from_formula('y ~ x', data)
        return model

    def test_run(self):
        with self.build_model():
            start = pm.find_MAP(method="Powell")
            pm.sample(50, pm.Slice(), start=start)


class TestLatentOccupancy(SeededTest):
    """
    From the PyMC example list
    latent_occupancy.py

    Simple model demonstrating the estimation of occupancy, using latent variables. Suppose
    a population of n sites, with some proportion pi being occupied. Each site is surveyed,
    yielding an array of counts, y:

    y = [3, 0, 0, 2, 1, 0, 1, 0, ..., ]

    This is a classic zero-inflated count problem, where more zeros appear in the data than would
    be predicted by a simple Poisson model. We have, in fact, a mixture of models; one, conditional
    on occupancy, with a poisson mean of theta, and another, conditional on absence, with mean zero.
    One way to tackle the problem is to model the latent state of 'occupancy' as a Bernoulli
    variable at each site, with some unknown probability:

    z_i ~ Bern(pi)

    These latent variables can then be used to generate an array of Poisson parameters:

    t_i = theta (if z_i=1) or 0 (if z_i=0)

    Hence, the likelihood is just:

    y_i = Poisson(t_i)

    (Note in this elementary model, we are ignoring the issue of imperfect detection.)

    Created by Chris Fonnesbeck on 2008-07-28.
    Copyright (c) 2008 University of Otago. All rights reserved.
    """
    def setup_method(self):
        super().setup_method()
        # Sample size
        n = 100
        # True mean count, given occupancy
        theta = 2.1
        # True occupancy
        pi = 0.4
        # Simulate some data data
        self.y = ((np.random.random(n) < pi) * np.random.poisson(lam=theta, size=n)).astype('int16')

    def build_model(self):
        with pm.Model() as model:
            # Estimated occupancy
            psi = pm.Beta('psi', 1, 1)
            # Latent variable for occupancy
            pm.Bernoulli('z', psi, shape=self.y.shape)
            # Estimated mean count
            theta = pm.Uniform('theta', 0, 100)
            # Poisson likelihood
            pm.ZeroInflatedPoisson('y', theta, psi, observed=self.y)
        return model

    def test_run(self):
        model = self.build_model()
        with model:
            start = {
                'psi': np.array(0.5, dtype='f'),
                'z': (self.y > 0).astype('int16'),
                'theta': np.array(5, dtype='f'),
            }
            step_one = pm.Metropolis([model.theta_interval__, model.psi_logodds__])
            step_two = pm.BinaryMetropolis([model.z])
            pm.sample(50, step=[step_one, step_two], start=start, chains=1)


@pytest.mark.skip(reason="Unknown issue in initialisation")
class TestRSV(SeededTest):
    '''
    This model estimates the population prevalence of respiratory syncytial virus
    (RSV) among children in Amman, Jordan, based on 3 years of admissions diagnosed
    with RSV to Al Bashir hospital.

    To estimate this parameter from raw counts of diagnoses, we need to establish
    the population of  1-year-old children from which the diagnosed individuals
    were sampled. This involved correcting census data (national estimate of
    1-year-olds) for the proportion of the population in the city, as well as for
    the market share of the hospital. The latter is based on expert esimate, and
    hence encoded as a prior.
    '''
    def build_model(self):
        # 1-year-old children in Jordan
        kids = np.array([180489, 191817, 190830])
        # Proportion of population in Amman
        amman_prop = 0.35
        # infant RSV cases in Al Bashir hostpital
        rsv_cases = np.array([40, 59, 65])
        with pm.Model() as model:
            # Al Bashir hospital market share
            market_share = pm.Uniform('market_share', 0.5, 0.6)
            # Number of 1 y.o. in Amman
            n_amman = pm.Binomial('n_amman', kids, amman_prop, shape=3)
            # Prior probability
            prev_rsv = pm.Beta('prev_rsv', 1, 5, shape=3)
            # RSV in Amman
            y_amman = pm.Binomial('y_amman', n_amman, prev_rsv, shape=3, testval=100)
            # Likelihood for number with RSV in hospital (assumes Pr(hosp | RSV) = 1)
            pm.Binomial('y_hosp', y_amman, market_share, observed=rsv_cases)
        return model

    def test_run(self):
        with self.build_model():
            pm.sample(50, step=[pm.NUTS(), pm.Metropolis()])


class TestMultilevelNormal(SeededTest):
    """
    Toy three-level normal model sampled using MLDA. The finest model is a
    Normal distribution with unknown mean and sigma=1.0 where we have only one
    observed datum (y = 11.0). The coarse models are the same but with the observed
    datum changed to y = 11.5 and y = 12.0. This is a very simple way to create
    a 3-level system of "approximate" coarse models.
    Normals with
    """

    def build_models(self):

        np.random.seed(1234)
        true_mean = 11.0
        y = np.array([true_mean])

        with pm.Model() as model_coarse_0:
            sigma = 1.0
            x_coeff = pm.Normal('x', true_mean, sigma=10.0)
            likelihood = pm.Normal('y', mu=x_coeff,
                                   sigma=sigma, observed=y + 1.0)

        with pm.Model() as model_coarse_1:
            sigma = 1.0
            x_coeff = pm.Normal('x', true_mean, sigma=10.0)
            likelihood = pm.Normal('y', mu=x_coeff,
                                   sigma=sigma, observed=y + 0.5)

        coarse_models = [model_coarse_0, model_coarse_1]

        with pm.Model() as model:
            sigma = 1.0
            x_coeff = pm.Normal('x', true_mean, sigma=10.0)
            likelihood = pm.Normal('y', mu=x_coeff,
                                   sigma=sigma, observed=y)

        return model, coarse_models

    def test_run(self):
        model, coarse_models = self.build_models()

        with model:
            step = pm.MLDA(subsampling_rate=2, coarse_models=coarse_models)
            trace = pm.sample(draws=50, chains=2, tune=50, step=step)


def sin_function(x, theta):
    return 2 * theta[0] * np.sin(theta[1] * 10 * x)


def my_loglik(my_model, theta, data, sigma):
    """
    This returns the log-likelihood of my_model given theta,
    datapoints, the observed data and sigma. It uses the
    model_wrapper function to do a model solve.
    """
    output = my_model.solve(theta)
    return - (0.5 / sigma ** 2) * np.sum((output - data) ** 2)


class SineModel:
    def __init__(self, resolution, datapoints):
        self.x = np.linspace(0, 1, resolution)
        self.datapoints = datapoints
        self.theta = None
        self.y = None

    def solve(self, theta):
        self.theta = theta
        self.y = sin_function(self.x, theta[0])
        return interp(self.datapoints, self.x, self.y)

    def plot(self):
        matplotlib.pyplot.plt.plot(self.x, self.y)
        matplotlib.pyplot.plt.show()


class LogLike(tt.Op):
    """
    Theano Op that wraps the log-likelihood computation.
    """
    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, my_model, loglike, data, sigma):
        """
        Initialise the Op with various things that the log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        my_model:
            A SineModel object that contains the parameters
            and functions of out model.
        loglike:
            The log-likelihood function we've defined, in this example it is
            my_loglik.
        data:
            The "observed" data that the log-likelihood function takes in. These
            are the true data generated by the finest model in this example.
        x:
            The dependent variable (aka 'x') that our model requires. This is
            the datapoints in this example.
        sigma:
            The noise standard deviation that our function requires.
        """

        # add inputs as class attributes
        self.my_model = my_model
        self.likelihood = loglike
        self.data = data
        self.sigma = sigma

    def perform(self, node, inputs, outputs):
        """
        The method that is used when calling the Op
        """
        theta = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(self.my_model, theta, self.data, self.sigma)

        outputs[0][0] = np.array(logl)  # output the log-likelihood


class TestSineWave(SeededTest):
    """
    Toy example using MLDA to find the amplitude and frequency of a sine
    wave, given some data points. It makes use of a linear interpolator with
    varying number of discretisation points.
    """

    def build_models(self):
        self.data_seed = 9999
        self.sampling_seed = 4321
        np.random.seed(self.data_seed)

        # Resolutions list (one for each level) and datapoints number
        self.resolutions = [50, 100, 150]
        self.n_datapoints = 10
        self.datapoints = np.random.uniform(size=self.n_datapoints)

        # Number of unknown parameters
        self.n_par = 2

        # Number of draws from the distribution
        self.ndraws = 100
        # Number of "burn-in points" (which we'll discard)
        self.nburn = 5
        # Tuning parameters for MLDA and Metropolis
        self.tune = True
        self.tune_interval = 100
        # Number of independent chains
        self.nchains = 2
        # Subsampling rate for MLDA
        self.nsub = 3
        # Set the sigma for inference
        self.sigma = 0.5

        # Initialise model objects for all levels
        my_models = []
        for r in self.resolutions:
            my_models.append(SineModel(r, self.datapoints))

        # Solve finest model
        my_models[-1].solve([np.random.uniform(low=0, high=3, size=self.n_par)])

        # Save true parameters of finest model
        true_parameters = my_models[-1].theta

        # Get data from the sampling points and perturb it with some noise.
        noise = np.random.normal(0, 0.001)

        # Generate data from the finest model for use in pymc3 inference
        data = sin_function(self.datapoints, true_parameters[0]) + noise

        # Create Theano Ops to wrap likelihoods of all model levels and store them in list
        logl = []
        for m in my_models:
            logl.append(LogLike(m, my_loglik, data, self.sigma))

        # Set up models in PyMC3 for each level - excluding finest model level
        coarse_models = []
        for j in range(len(my_models) - 1):
            with pm.Model() as model:
                # uniform priors on parameters
                parameters = []
                for i in range(self.n_par):
                    parameters.append(pm.Uniform('theta_' + str(i), lower=0., upper=3.))

                # convert m and c to a tensor vector
                theta = tt.as_tensor_variable(parameters)

                # use a DensityDist (use a lamdba function to "call" the Op)
                pm.DensityDist('likelihood', lambda v: logl[j](v), observed={'v': theta})

            coarse_models.append(model)

        # Set up finest model
        with pm.Model() as model:
            # uniform priors on parameters
            parameters = []
            for i in range(self.n_par):
                parameters.append(pm.Uniform('theta_' + str(i), lower=0., upper=3.))

            # convert m and c to a tensor vector
            theta = tt.as_tensor_variable(parameters)

            # use a DensityDist (use a lamdba function to "call" the Op)
            pm.DensityDist('likelihood', lambda v: logl[-1](v), observed={'v': theta})

        return model, coarse_models

    def test_run(self):
        model, coarse_models = self.build_models()

        # Perform inference with PyMC3, using the MLDA algorithm
        # and passing the coarse_models list
        method_names = []
        traces = []
        acc = []
        ess = []
        with model:
            # Initialise an MLDA step method object, passing the subsampling rate and
            # coarse models list
            # Also initialise a Metropolis step method object
            step_metropolis = pm.Metropolis(tune=True, tune_interval=self.tune_interval)
            step_mlda = pm.MLDA(subsampling_rate=self.nsub, coarse_models=coarse_models,
                                tune=self.tune, tune_interval=self.tune_interval)

            # inference
            # Metropolis
            method_names.append("Metropolis")
            traces.append(pm.sample(draws=self.ndraws, step=step_metropolis,
                                    chains=self.nchains, tune=self.nburn,
                                    random_seed=self.sampling_seed))
            # MLDA
            method_names.append("MLDA")
            traces.append(pm.sample(draws=self.ndraws, step=step_mlda,
                                    chains=self.nchains, tune=self.nburn,
                                    random_seed=self.sampling_seed))

            for i, trace in enumerate(traces):
                acc.append(trace.get_sampler_stats('accepted').mean())
                ess.append(np.array(pm.ess(trace).to_array()))
                print(f'\nSampler {method_names[i]}: {len(trace)} samples across '
                      f'{trace.nchains} chains.'
                      f'\nAcceptance rate: {acc[i]}'
                      f'\nESS list: {ess[i]}')
