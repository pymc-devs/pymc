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
from scipy.optimize import minimize
from theano import function as theano_function
import arviz as az
import jax

# This is temporary, until I've turned it into a full variational method.
from pymc3.el2o.el2o_basic import *

from pymc3.backends.ndarray import NDArray
from pymc3.model import Point, modelcontext
from pymc3.sampling import sample_prior_predictive
from pymc3.theanof import (
    floatX,
    inputvars,
    join_nonshared_inputs,
    make_shared_replacements,
    gradient,
    hessian,
)

# SINF code for fitting the normalizing flow.
from pymc3.sinf.GIS import GIS
import torch


# This is a global variable used to store the optimization steps.
# Presumably there's a nicer way to do this.
param_store = []

class NFMC:
    """Sequential type normalizing flow based sampling/global approx."""

    def __init__(
        self,
        draws=500,
        model=None,
        init_samples=None,
        pareto=False,
        random_seed=-1,
        chain=0,
        frac_validate=0.1,
        alpha=(0,0),
        optim_iter=1000,
        ftol=2.220446049250313e-9,
        gtol=1.0e-5,
        k_trunc=0.25,
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
        self.model = model
        self.init_samples = init_samples
        self.pareto = pareto,
        self.random_seed = random_seed
        self.chain = chain
        self.frac_validate = frac_validate
        self.alpha = alpha
        self.optim_iter = optim_iter
        self.ftol = ftol
        self.gtol = gtol
        self.k_trunc = k_trunc
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

        self.variables = inputvars(self.model.vars)
        
    def initialize_population(self):
        """Create an initial population from the prior distribution."""
        population = []
        var_info = OrderedDict()

        init = self.model.test_point
        
        for v in self.variables:
            var_info[v.name] = (init[v.name].shape, init[v.name].size)
        
        if self.init_samples is None:
            init_rnd = sample_prior_predictive(
                self.draws,
                var_names=[v.name for v in self.model.unobserved_RVs],
                model=self.model,
            )

            for i in range(self.draws):

                point = Point({v.name: init_rnd[v.name][i] for v in self.variables}, model=self.model)
                population.append(self.model.dict_to_array(point))

            self.prior_samples = np.array(floatX(population))

        elif self.init_samples is not None:
            
            self.prior_samples = np.copy(self.init_samples)

        print('Prior sample check ...')
        print(f'Shape of prior samples = {np.shape(self.prior_samples)}')
        self.var_info = var_info
        self.weighted_samples = np.empty((0, np.shape(self.prior_samples)[1]))
        self.importance_weights = np.array([])
        self.posterior = np.empty((0, np.shape(self.prior_samples)[1]))
        self.nf_models = []

    def setup_logp(self):
        """Set up the prior and likelihood logp functions, and derivatives."""
        shared = make_shared_replacements(self.variables, self.model)

        self.prior_logp_func = logp_forw([self.model.varlogpt], self.variables, shared)
        self.likelihood_logp_func = logp_forw([self.model.datalogpt], self.variables, shared)
        self.posterior_logp_func = logp_forw([self.model.logpt], self.variables, shared)
        self.posterior_dlogp_func = logp_forw([gradient(self.model.logpt, self.variables)], self.variables, shared)
        self.posterior_hessian_func = logp_forw([hessian(self.model.logpt, self.variables)], self.variables, shared)
        
    def get_prior_logp(self):
        """Get the prior log probabilities."""
        priors = [self.prior_logp_func(sample) for sample in self.nf_samples]

        self.prior_logp = np.array(priors).squeeze()

    def get_likelihood_logp(self):
        """Get the likelihood log probabilities."""
        likelihoods = [self.likelihood_logp_func(sample) for sample in self.nf_samples]

        self.likelihood_logp = np.array(likelihoods).squeeze()

    def get_posterior_logp(self):
        """Get the posterior log probabilities."""
        posteriors = [self.posterior_logp_func(sample) for sample in self.nf_samples]

        self.posterior_logp = np.array(posteriors).squeeze()

    def optim_target_logp(self, param_vals):
        """Optimization target function"""
        return -1.0 * self.posterior_logp_func(param_vals)

    def optim_target_dlogp(self, param_vals):
        return -1.0 * self.posterior_dlogp_func(param_vals)

    def target_logp(self, param_vals):
        logps = [self.posterior_logp_func(val) for val in param_vals]
        return np.array(logps).squeeze()

    def target_dlogp(self, param_vals):
        dlogps = [self.posterior_dlogp_func(val) for val in param_vals]
	return np.array(dlogps).squeeze()
        
    def target_hessian(self, param_vals):
        hessians = [self.posterior_hessian_func(val) for val in param_vals]
        return np.array(hessians).squeeze()

    def callback(self, xk):
        self.optim_iter_samples = np.append(self.optim_iter_samples, np.array([xk]), axis=0)
    
    def optimize(self, sample):
        """Optimize the prior samples"""
        self.optim_iter_samples = np.array([sample])
        minimize(self.optim_target_logp, x0=sample, method='L-BFGS-B',
                 options={'maxiter': self.optim_iter, 'ftol': self.ftol, 'gtol': self.gtol},
                 jac=self.optim_target_dlogp, callback=self.callback)
        return self.optim_iter_samples 
        
    def initialize_nf(self):
        """Intialize the first NF approx, by fitting to the prior samples."""
        val_idx = int((1 - self.frac_validate) * self.prior_samples.shape[0])

        self.nf_model = GIS(torch.from_numpy(self.prior_samples[:val_idx, ...].astype(np.float32)),
                            torch.from_numpy(self.prior_samples[val_idx:, ...].astype(np.float32)),
                            alpha=None, verbose=self.verbose, n_component=self.n_component,
                            interp_nbin=self.interp_nbin, KDE=self.KDE, bw_factor=self.bw_factor,
                            edge_bins=self.edge_bins, ndata_wT=self.ndata_wT, MSWD_max_iter=self.MSWD_max_iter,
                            NBfirstlayer=self.NBfirstlayer, logit=self.logit, Whiten=self.Whiten,
                            batchsize=self.batchsize, nocuda=self.nocuda, patch=self.patch, shape=self.shape)
        
        self.nf_samples, self.logq = self.nf_model.sample(self.draws, device=torch.device('cpu'))
        self.nf_samples = self.nf_samples.numpy().astype(np.float64)
        self.weighted_samples = np.append(self.weighted_samples, self.nf_samples, axis=0)
        self.get_posterior_logp()
        log_weight = self.posterior_logp - self.logq.numpy().astype(np.float64)
        self.evidence = np.mean(np.exp(log_weight))
        
        if self.pareto:
            psiw = az.psislw(log_weight)
            self.weights = np.exp(psiw[0])
        elif not self.pareto:
            self.weights = np.exp(log_weight)
            self.weights = np.clip(self.weights, 0, np.mean(self.weights) * len(self.weights)**self.k_trunc)

        self.weights = self.weights / np.sum(self.weights)
        self.importance_weights = np.append(self.importance_weights, self.weights)
        self.nf_models.append(self.nf_model)

    def logq_fr_el20(self, z, mu, Sigma):
        """Logq for full-rank Gaussian family."""
        return jax.scipy.stats.multivariate_normal.logpdf(z, mu, Sigma) 
        
    def run_el2o(self):
        """Run the EL2O algorithm, assuming you've got the MAP+Laplace solution."""

        self.mu_k = self.mu_map
        self.Sigma_k = self.Sigma_map
        self.EL2O = [1e10, 1]

        self.zk = np.random.multivariate_normal(self.mu_k, self.Sigma_k)
        #self.zk = self.zk.reshape(-1, 2)

        N = 1
        while self.EL2O[-1] > self.absEL2O and abs((self.EL2O[-1] - self.EL2O[-2]) / self.EL2O[-1]) > self.fracEL2O:

            for i in range(N):

                if i > 0:
                    self.zk = np.vstack((self.zk, np.random.multivariate_normal(self.mu_k, self.Sigma_k)))
                    #self.zk = self.zk.reshape(-1, 2)

                Nk = len(self.zk)
                self.Sigma_k = -1 * np.linalg.inv(np.sum(self.target_hessian(zk), axis=0) / Nk)

                temp = 0
                for j in range(Nk):
                    temp += np.dot(self.Sigma_k, self.target_dlogp(zk[j, :])) + zk[j, :]
                self.mu_k = temp / Nk

                self.EL2O = np.append(self.EL2O, 1 / (self.len(zk)) * (np.sum((self.target_logp(self.zk) - Lq(self.zk, self.mu_k, self.Sigma_k))**2) +
                                                                       np.sum((self.target_dlogp(zk) - 
                                                                               jax.vmap(jax.grad(self.logq_fr_el2o)(self.zk, self.mu_k, self.Sigma_k)))**2) +
                                                                       np.sum((self.target_hessian(zk) - 
                                                                               jax.vmap(jax.hessian(self.logq_fr_el2o)(self.zk, self.mu_k, self.Sigma_k)))**2)))
            N = N + 1

        return self.mu_k, self.Sigma_k, self.zk, self.EL2O

        
    def fit_nf(self):
        """Fit the NF model for a given iteration after initialization."""
        val_idx = int((1 - self.frac_validate) * self.weighted_samples.shape[0])
        
        self.nf_model = GIS(torch.from_numpy(self.weighted_samples[:val_idx, ...].astype(np.float32)),
                            torch.from_numpy(self.weighted_samples[val_idx:, ...].astype(np.float32)),
                            weight_train=torch.from_numpy(self.importance_weights[:val_idx, ...].astype(np.float32)),
                            weight_validate=torch.from_numpy(self.importance_weights[val_idx:, ...].astype(np.float32)),
                            alpha=self.alpha, verbose=self.verbose, n_component=self.n_component,
                            interp_nbin=self.interp_nbin, KDE=self.KDE, bw_factor=self.bw_factor,
                            edge_bins=self.edge_bins, ndata_wT=self.ndata_wT, MSWD_max_iter=self.MSWD_max_iter,
                            NBfirstlayer=self.NBfirstlayer, logit=self.logit, Whiten=self.Whiten,
                            batchsize=self.batchsize, nocuda=self.nocuda, patch=self.patch, shape=self.shape)
        
        self.nf_samples, self.logq = self.nf_model.sample(self.draws, device=torch.device('cpu'))
        self.nf_samples = self.nf_samples.numpy().astype(np.float64)
        self.weighted_samples = np.append(self.weighted_samples, self.nf_samples, axis=0)
        self.get_posterior_logp()
        log_weight = self.posterior_logp - self.logq.numpy().astype(np.float64)
        self.evidence = np.mean(np.exp(log_weight))

        if self.pareto:
            psiw = az.psislw(log_weight)
            self.weights = np.exp(psiw[0])
        elif not self.pareto:
            self.weights = np.exp(log_weight)
            self.weights = np.clip(self.weights, 0, np.mean(self.weights) * len(self.weights)**self.k_trunc)

        self.weights = self.weights / np.sum(self.weights)
        self.importance_weights = np.append(self.importance_weights, self.weights)
        self.nf_models.append(self.nf_model)
        
    def resample_iter(self):
        """Resample at a given NF fit iteration, to obtain samples for the next stage."""
        resampling_indexes = np.random.choice(
            np.arange(len(self.weights)), size=self.draws, p=self.weights/np.sum(self.weights)
        )
        self.nf_samples = self.nf_samples[resampling_indexes, ...]
        
        
    def resample(self):
        """Resample all the weighted samples to obtain final posterior samples with uniform weight."""
        resampling_indexes = np.random.choice(
            np.arange(len(self.importance_weights)), size=self.draws, p=self.importance_weights/np.sum(self.importance_weights)
        )
        #resampling_indexes = np.random.choice(
        #    np.arange(len(self.weights)), size=self.draws, p=self.weights/np.sum(self.weights)
        #)
        self.posterior = self.weighted_samples[resampling_indexes, ...]
        #self.posterior = self.nf_samples[resampling_indexes, ...]
        
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
def callback(xk):
    """Function used as a callback during optimization steps.
    
    Parameters
    ----------
    xk: Array
        Array containing the current parameter vector for the given optimization step.
    """
    optim_iter_samples = np.append(optim_iter_samples, np.array([xk]), axis=0)
'''


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
