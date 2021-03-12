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

from pymc3.sinf.GIS import GIS
import torch


class NF_SMC:
    """Sequential Monte Carlo with normalizing flow based sampling."""

    def __init__(
        self,
        draws=2000,
        start=None,
        threshold=0.5,
        model=None,
        random_seed=-1,
        chain=0,
        frac_validate=0.1,
        alpha=(0,0),
        k_trunc=0.25,
        epsilon=1e-3,
        verbose=False,
        n_component=None,
	interp_nbin=None,
	KDE=True,
        bw_factor=0.5,
        edge_bins=None,
        ndata_wT=None,
	MSWD_max_iter=None,
        NBfirstlayer=True,
        logit=False,
	Whiten=False,
        batchsize=None,
        nocuda=False,
        patch=False,
        shape=[28,28,1],
    ):

        self.draws = draws
        self.start = start
        self.threshold = threshold
        self.model = model
        self.random_seed = random_seed
        self.chain = chain
        self.frac_validate = frac_validate
        self.alpha = alpha
        self.k_trunc = k_trunc
        self.epsilon = epsilon
        self.verbose = verbose
        self.n_component = n_component
        self.interp_nbin = interp_nbin
        self.KDE = KDE
        self.bw_factor = bw_factor
        self.edge_bins = edge_bins
        self.ndata_wT = ndata_wT
        self.MSWD_max_iter = MSWD_max_iter
        self.NBfirstlayer = NBfirstlayer
        self.logit = logit
        self.Whiten = Whiten
        self.batchsize = batchsize
        self.nocuda = nocuda
        self.patch = patch
        self.shape = shape
        
        self.model = modelcontext(model)

        if self.random_seed != -1:
            np.random.seed(self.random_seed)

        self.beta = 0
        self.variables = inputvars(self.model.vars)
        self.weights = np.ones(self.draws) / self.draws
        self.sinf_logq = np.array([])
        self.log_marginal_likelihood = 0

    def initialize_population(self):
        """Create an initial population from the prior distribution."""
        population = []
        var_info = OrderedDict()
        if self.start is None:
            init_rnd = sample_prior_predictive(
                self.draws,
                var_names=[v.name for v in self.model.unobserved_RVs],
                model=self.model,
            )
        else:
            init_rnd = self.start

        init = self.model.test_point

        for v in self.variables:
            var_info[v.name] = (init[v.name].shape, init[v.name].size)

        for i in range(self.draws):

            point = Point({v.name: init_rnd[v.name][i] for v in self.variables}, model=self.model)
            population.append(self.model.dict_to_array(point))

        self.nf_samples = np.array(floatX(population))
        self.posterior = np.copy(self.nf_samples)
        self.var_info = var_info
        
    def setup_logp(self):
        """Set up the likelihood logp function based on the chosen kernel."""
        shared = make_shared_replacements(self.variables, self.model)

        self.prior_logp_func = logp_forw([self.model.varlogpt], self.variables, shared)
        self.likelihood_logp_func = logp_forw([self.model.datalogpt], self.variables, shared)

    def get_nf_logp(self):
        """Get the prior, likelihood and tempered posterior log probabilities, for the current NF samples."""
        priors = [self.prior_logp_func(sample) for sample in self.nf_samples]
        likelihoods = [self.likelihood_logp_func(sample) for sample in self.nf_samples]

        self.nf_prior_logp = np.array(priors).squeeze()
        self.nf_likelihood_logp = np.array(likelihoods).squeeze()
        self.nf_posterior_logp = self.nf_prior_logp + self.nf_likelihood_logp * self.beta

    def get_full_logp(self):
        """Get the prior, likelihood and tempered posterior log probabilities, for the full sample set."""
        priors = [self.prior_logp_func(sample) for sample in self.posterior]
        likelihoods = [self.likelihood_logp_func(sample) for sample in self.posterior]

        self.prior_logp = np.array(priors).squeeze()
        self.likelihood_logp = np.array(likelihoods).squeeze()
        self.posterior_logp = self.prior_logp + self.likelihood_logp * self.beta
        
    def update_weights_beta(self):
        """Calculate the next inverse temperature (beta).

        The importance weights based on current beta and tempered likelihood and updates the
        marginal likelihood estimate.
        """
        low_beta = old_beta = self.beta
        up_beta = 2.0
        rN = int(len(self.nf_likelihood_logp) * self.threshold)

        if self.beta == 0:
            self.sinf_logq = np.append(self.sinf_logq, self.nf_prior_logp)
            log_weights_q = np.ones_like(self.nf_prior_logp) / self.draws
        else:
            log_weights_q = self.nf_prior_logp + self.nf_likelihood_logp * self.beta - self.logq
            log_weights_q = np.clip(log_weights_q, a_min=None,
                                    a_max=np.log(np.mean(np.exp(log_weights_q))) + self.k_trunc * np.log(self.draws))
            log_weights_q = log_weights_q - logsumexp(log_weights_q)
        
        while up_beta - low_beta > 1e-6:
            new_beta = (low_beta + up_beta) / 2.0
            '''
            if old_beta == 0:
                log_weights_un = (new_beta - old_beta) * self.likelihood_logp
            else:
                log_weights_un = self.prior_logp + self.likelihood_logp * new_beta - self.logq
            '''
            log_weights_un = (new_beta - old_beta) * self.nf_likelihood_logp
            log_weights = log_weights_un - logsumexp(log_weights_un)
            
            ESS = int(np.exp(-logsumexp(log_weights_q + log_weights * 2)) / self.draws)
            if ESS == rN:
                break
            elif ESS < rN:
                up_beta = new_beta
            else:
                low_beta = new_beta
        if new_beta >= 1:
            new_beta = 1
            log_weights_un = (new_beta - old_beta) * self.nf_likelihood_logp
            #log_weights_un = self.prior_logp + self.likelihood_logp * new_beta - self.logq
            log_weights = log_weights_un - logsumexp(log_weights_un)

        self.log_marginal_likelihood += logsumexp(log_weights_un) - np.log(self.draws)
        self.beta = new_beta
        self.weights = np.exp(log_weights)
        
        # We normalize again to correct for small numerical errors that might build up
        self.weights /= self.weights.sum()

        self.sinf_weights = np.exp(self.prior_logp + self.likelihood_logp * self.beta - self.sinf_logq)
        self.sinf_weights = np.clip(self.sinf_weights, 0, np.mean(self.sinf_weights) * len(self.sinf_weights)**self.k_trunc)

        low_weight_idx = np.where(self.sinf_weights < np.mean(self.sinf_weights) * self.epsilon)[0]
        self.sinf_weights = np.delete(self.sinf_weights, low_weight_idx)
        self.posterior = np.delete(self.posterior, low_weight_idx, axis=0)
        self.sinf_logq = np.delete(self.sinf_logq, low_weight_idx)
        self.sinf_weights /= self.sinf_weights.sum()
        
        if old_beta == 0:
            self.raw_weights = np.exp((new_beta - old_beta) * self.nf_likelihood_logp)
        else:
            log_raw_weights = self.nf_prior_logp + self.nf_likelihood_logp * self.beta - self.logq
            log_raw_weights = np.clip(log_raw_weights, a_min=None,
                                      a_max=np.log(np.mean(np.exp(log_raw_weights))) + self.k_trunc * np.log(self.draws))
            self.raw_weights = np.exp(log_raw_weights)

    def resample(self):
        """Resample particles based on importance weights."""
        self.sinf_weights = np.exp(self.prior_logp + self.likelihood_logp * self.beta - self.sinf_logq)
        self.sinf_weights = np.clip(self.sinf_weights, 0, np.mean(self.sinf_weights) * len(self.sinf_weights)**self.k_trunc)
        self.sinf_weights /= self.sinf_weights.sum()
        resampling_indexes = np.random.choice(
            np.arange(self.posterior.shape[0]), size=self.draws, p=self.sinf_weights
        )

        self.posterior = self.posterior[resampling_indexes]
        self.prior_logp = self.prior_logp[resampling_indexes]
        self.likelihood_logp = self.likelihood_logp[resampling_indexes]
        self.posterior_logp = self.prior_logp + self.likelihood_logp * self.beta
        
    def fit_nf(self):
        """Fit an NF approximation to the current tempered posterior."""
        num_val = int(self.frac_validate * self.posterior.shape[0])
        val_idx = np.random.choice(np.arange(self.posterior.shape[0]), size=num_val, replace=False)
        fit_idx = np.delete(np.arange(self.posterior.shape[0]), val_idx)
        
        self.nf_model = GIS(torch.from_numpy(self.posterior[fit_idx, ...].astype(np.float32)),
                            torch.from_numpy(self.posterior[val_idx, ...].astype(np.float32)),
                            weight_train=torch.from_numpy(self.sinf_weights[fit_idx, ...].astype(np.float32)),
                            weight_validate=torch.from_numpy(self.sinf_weights[val_idx, ...].astype(np.float32)),
                            alpha=self.alpha, verbose=self.verbose, n_component=self.n_component,
                            interp_nbin=self.interp_nbin, KDE=self.KDE, bw_factor=self.bw_factor,
                            edge_bins=self.edge_bins, ndata_wT=self.ndata_wT, MSWD_max_iter=self.MSWD_max_iter,
                            NBfirstlayer=self.NBfirstlayer, logit=self.logit, Whiten=self.Whiten,
                            batchsize=self.batchsize, nocuda=self.nocuda, patch=self.patch, shape=self.shape)

        self.nf_samples, self.logq = self.nf_model.sample(self.draws, device=torch.device('cpu'))
        self.nf_samples = self.nf_samples.numpy().astype(np.float64)
        self.posterior = np.append(self.posterior, self.nf_samples, axis=0)
        self.logq = self.logq.numpy().astype(np.float64)
        self.sinf_logq = np.append(self.sinf_logq, self.logq)
        
    def resample_nf_iw(self):
        """Resample the NF samples at a given iteration, applying IW correction to account for
        mis-match between NF fit and the current tempered posterior."""
        self.log_mismatch_un = self.prior_logp + self.likelihood_logp * self.beta - self.logq
        self.log_mismatch = self.log_mismatch_un - logsumexp(self.log_mismatch_un)
        self.mismatch = np.exp(self.log_mismatch)
        self.mismatch /= self.mismatch.sum()
        
        resampling_indexes = np.random.choice(
            np.arange(10*self.draws), size=self.draws, p=self.mismatch
        )

        self.posterior = self.posterior[resampling_indexes]
        self.prior_logp = self.prior_logp[resampling_indexes]
        self.likelihood_logp = self.likelihood_logp[resampling_indexes]
        self.logq = self.logq[resampling_indexes]
        self.posterior_logp = self.prior_logp + self.likelihood_logp * self.beta
        
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
