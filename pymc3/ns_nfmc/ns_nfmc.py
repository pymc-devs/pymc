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

from collections import OrderedDict

import numpy as np
import theano.tensor as tt

from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from theano import function as theano_function

from pymc3.backends.ndarray import NDArray
from pymc3.model import Point, modelcontext
from pymc3.sampling import sample_prior_predictive
from pymc3.theanof import (
    floatX,
    inputvars,
    join_nonshared_inputs,
    make_shared_replacements,
)

# SINF code for fitting the normalizing flow.
from pymc3.ns_nfmc.GIS import GIS
import torch


class NS_NFMC:
    """Nested sampling with normalizing flow based density estimation and sampling."""

    def __init__(
        self,
        draws=10000,
        model=None,
        random_seed=-1,
        chain=0,
        frac_validate=0.8,
        alpha=(0,0),
        rho=0.01
    ):

        self.draws = draws
        self.model = model
        self.random_seed = random_seed
        self.chain = chain
        self.frac_validate = frac_validate
        self.alpha = alpha
        self.rho = rho
        
        self.model = modelcontext(model)

        if self.random_seed != -1:
            np.random.seed(self.random_seed)

        self.variables = inputvars(self.model.vars)
        self.log_marginal_likelihood = 0
        self.log_volume_factor = np.zeros(1)
        self.prior_weight = np.ones(self.draws) / self.draws
        self.posterior_weights = np.array([])
        self.log_evidences = np.array([])
        self.cumul_evidences = np.zeros(1)
        self.likelihood_logp_thresh = np.array([])
        
    def initialize_population(self):
        """Create an initial population from the prior distribution."""
        population = []
        var_info = OrderedDict()

        init_rnd = sample_prior_predictive(
            self.draws,
            var_names=[v.name for v in self.model.unobserved_RVs],
            model=self.model,
        )


        init = self.model.test_point

        for v in self.variables:
            var_info[v.name] = (init[v.name].shape, init[v.name].size)

        for i in range(self.draws):

            point = Point({v.name: init_rnd[v.name][i] for v in self.variables}, model=self.model)
            population.append(self.model.dict_to_array(point))

        self.nf_samples = np.array(floatX(population))
        self.live_points = np.array(floatX(population))
        self.var_info = var_info
        self.posterior = np.empty((0, np.shape(self.nf_samples)[1]))
        
    def setup_logp(self):
        """Set up the prior and likelihood logp functions."""
        shared = make_shared_replacements(self.variables, self.model)

        self.prior_logp_func = logp_forw([self.model.varlogpt], self.variables, shared)
        self.likelihood_logp_func = logp_forw([self.model.datalogpt], self.variables, shared)
        
    def get_prior_logp(self):
        """Get the prior log probabilities."""
        priors = [self.prior_logp_func(sample) for sample in self.nf_samples]

        self.prior_logp = np.array(priors).squeeze()

    def get_likelihood_logp(self):
        """Get the likelihood log probabilities."""
        likelihoods = [self.likelihood_logp_func(sample) for sample in self.nf_samples]

        self.likelihood_logp = np.array(likelihoods).squeeze()
        
    def fit_nf(self):
        """Fit the NF model to samples for the given likelihood level and draw new sample set."""
        val_idx = int((1 - self.frac_validate) * self.live_points.shape[0])
        self.nf_model = GIS(torch.from_numpy(self.live_points[:val_idx, ...].astype(np.float32)),
                            torch.from_numpy(self.live_points[val_idx:, ...].astype(np.float32)),
                            alpha=self.alpha)
        self.nf_samples, _ = self.nf_model.sample(self.draws, device=torch.device('cpu'))
        self.nf_samples = self.nf_samples.numpy().astype(np.float64)
        
    def update_likelihood_thresh(self):
        """Adaptively set the new likelihood threshold, based on the samples at the previous NS iteration."""
        self.get_likelihood_logp()
        self.likelihood_logp_thresh = np.append(self.likelihood_logp_thresh, np.quantile(self.likelihood_logp, 1 - self.rho))

    def update_weights(self):
        """Update the prior and posterior weights for the given iteration, along with the evidences and volume factors."""
        self.prior_weight = (self.likelihood_logp >= self.likelihood_logp_thresh[-1:]).astype(int) / self.draws
        self.live_points = self.nf_samples[self.prior_weight != 0]
        self.cut_idx = np.where(self.likelihood_logp < self.likelihood_logp_thresh[-1:])[0]
        log_posterior_weight = self.log_volume_factor[-1:] + self.likelihood_logp[self.cut_idx] - np.log(self.draws)   

        #weight = np.exp(self.log_volume_factor[-1:]) * np.exp(self.likelihood_logp[self.cut_idx]) / self.draws
        self.posterior_weights = np.append(self.posterior_weights, np.exp(log_posterior_weight))
        self.log_evidences = np.append(self.log_evidences, logsumexp(log_posterior_weight))
        self.posterior = np.append(self.posterior, self.nf_samples[self.cut_idx, ...], axis=0)
        self.log_volume_factor = np.append(self.log_volume_factor, self.log_volume_factor[-1:] + np.log(np.sum(self.prior_weight)))
        self.cumul_evidences = np.append(self.cumul_evidences, np.exp(logsumexp(self.log_evidences)))
        
    def resample(self):
        """Resample particles given the calculated posterior weights."""
        resampling_indexes = np.random.choice(
            np.arange(len(self.posterior_weights)), size=self.draws, p=self.posterior_weights/np.sum(self.posterior_weights)
        )

        self.posterior = self.posterior[resampling_indexes, ...]
        self.nf_samples = np.copy(self.posterior)
        self.get_prior_logp()
        self.get_likelihood_logp()
        self.posterior_logp = self.prior_logp + self.likelihood_logp
        
    def posterior_to_trace(self):
        """Save results into a PyMC3 trace."""
        lenght_pos = len(self.posterior)
        varnames = [v.name for v in self.variables]
        
        with self.model:
            strace = NDArray(name=self.model.name)
            strace.setup(lenght_pos, self.chain)
        for i in range(lenght_pos):
            value = []
            size = 0
            for var in varnames:
                shape, new_size = self.var_info[var]
                value.append(self.posterior[i][size : size + new_size].reshape(shape))
                size += new_size
            strace.record(point={k: v for k, v in zip(varnames, value)})
        return strace


def logp_forw(out_vars, vars, shared):
    """Compile Theano function of the model and the input and output variables.

    Parameters
    ----------
    out_vars: List
        containing :class:`pymc3.Distribution` for the output variables
    vars: List
        containing :class:`pymc3.Distribution` for the input variables
    shared: List
        containing :class:`theano.tensor.Tensor` for depended shared data
    """
    out_list, inarray0 = join_nonshared_inputs(out_vars, vars, shared)
    f = theano_function([inarray0], out_list[0])
    f.trust_input = True
    return f


'''

# RG: Not going to worry about simulation based inference for now - just stick with analytic likelihoods.

class PseudoLikelihood:
    """
    Pseudo Likelihood.

    epsilon: float
        Standard deviation of the gaussian pseudo likelihood.
    observations: array-like
        observed data
    function: python function
        data simulator
    params: list
        names of the variables parameterizing the simulator.
    model: PyMC3 model
    var_info: dict
        generated by ``SMC.initialize_population``
    variables: list
        Model variables.
    distance : str or callable
        Distance function.
    sum_stat: str or callable
        Summary statistics.
    size : int
        Number of simulated datasets to save. When this number is exceeded the counter will be
        restored to zero and it will start saving again.
    save_sim_data : bool
        whether to save or not the simulated data.
    save_log_pseudolikelihood : bool
        whether to save or not the log pseudolikelihood values.
    """

    def __init__(
        self,
        epsilon,
        observations,
        function,
        params,
        model,
        var_info,
        variables,
        distance,
        sum_stat,
        size,
        save_sim_data,
        save_log_pseudolikelihood,
    ):
        self.epsilon = epsilon
        self.function = function
        self.params = params
        self.model = model
        self.var_info = var_info
        self.variables = variables
        self.varnames = [v.name for v in self.variables]
        self.distance = distance
        self.sum_stat = sum_stat
        self.unobserved_RVs = [v.name for v in self.model.unobserved_RVs]
        self.get_unobserved_fn = self.model.fastfn(self.model.unobserved_RVs)
        self.size = size
        self.save_sim_data = save_sim_data
        self.save_log_pseudolikelihood = save_log_pseudolikelihood
        self.sim_data_l = []
        self.lpl_l = []

        self.observations = self.sum_stat(observations)

    def posterior_to_function(self, posterior):
        """Turn posterior samples into function parameters to feed the simulator."""
        model = self.model
        var_info = self.var_info

        varvalues = []
        samples = {}
        size = 0
        for var in self.variables:
            shape, new_size = var_info[var.name]
            varvalues.append(posterior[size : size + new_size].reshape(shape))
            size += new_size
        point = {k: v for k, v in zip(self.varnames, varvalues)}
        for varname, value in zip(self.unobserved_RVs, self.get_unobserved_fn(point)):
            if varname in self.params:
                samples[varname] = value
        return samples

    def save_data(self, sim_data):
        """Save simulated data."""
        if len(self.sim_data_l) == self.size:
            self.sim_data_l = []
        self.sim_data_l.append(sim_data)

    def get_data(self):
        """Get simulated data."""
        return np.array(self.sim_data_l)

    def save_lpl(self, elemwise):
        """Save log pseudolikelihood values."""
        if len(self.lpl_l) == self.size:
            self.lpl_l = []
        self.lpl_l.append(elemwise)

    def get_lpl(self):
        """Get log pseudolikelihood values."""
        return np.array(self.lpl_l)

    def __call__(self, posterior):
        """Compute the pseudolikelihood."""
        func_parameters = self.posterior_to_function(posterior)
        sim_data = self.function(**func_parameters)
        if self.save_sim_data:
            self.save_data(sim_data)
        elemwise = self.distance(self.epsilon, self.observations, self.sum_stat(sim_data))
        if self.save_log_pseudolikelihood:
            self.save_lpl(elemwise)
        return elemwise.sum()
'''
