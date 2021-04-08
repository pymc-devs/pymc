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
from scipy.stats import multivariate_normal, median_abs_deviation
from scipy.optimize import minimize, approx_fprime
from theano import function as theano_function
import arviz as az
import jax
import jax.numpy as jnp

# This is temporary, until I've turned it into a full variational method.
#from pymc3.el2o.el2o_basic import *

from pymc3.tuning.scaling import find_hessian
from pymc3.tuning.starting import find_MAP
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
        init_method='prior',
        init_samples=None,
        absEL2O=1e-10,
        fracEL2O=1e-2,
        pareto=False,
        local_thresh=3,
        local_step_size=0.1,
        local_grad=True,
        init_local=True,
        nf_local_iter=0,
        max_line_search=2,
        random_seed=-1,
        chain=0,
        frac_validate=0.1,
        iteration=None,
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
        self.init_method = init_method
        self.init_samples = init_samples
        self.absEL2O = 1e-10
        self.fracEL2O = 1e-2
        self.pareto = pareto,
        self.local_thresh = local_thresh
        self.local_step_size = local_step_size
        self.local_grad = local_grad
        self.init_local = init_local
        self.nf_local_iter = nf_local_iter
        self.max_line_search = max_line_search
        self.random_seed = random_seed
        self.chain = chain
        self.frac_validate = frac_validate
        self.iteration = iteration
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

    def initialize_var_info(self):
        """Extract variable info for the model instance."""
        var_info = OrderedDict()
        init = self.model.test_point
        for v in self.variables:
            var_info[v.name] = (init[v.name].shape, init[v.name].size)
        self.var_info = var_info
        
    def initialize_population(self):
        """Create an initial population from the prior distribution."""
        population = []

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

        self.weighted_samples = np.empty((0, np.shape(self.prior_samples)[1]))
        self.all_logq = np.array([])
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

    def sinf_logq(self, param_vals):
        sinf_logq = self.nf_model.evaluate_density(torch.from_numpy(param_vals.astype(np.float32))).numpy().astype(np.float64)
        return sinf_logq.item()

    def callback(self, xk):
        self.optim_iter_samples = np.append(self.optim_iter_samples, np.array([xk]), axis=0)
    
    def optimize(self, sample):
        """Optimize the prior samples"""
        self.optim_iter_samples = np.array([sample])
        minimize(self.optim_target_logp, x0=sample, method='L-BFGS-B',
                 options={'maxiter': self.optim_iter, 'ftol': self.ftol, 'gtol': self.gtol},
                 jac=self.optim_target_dlogp, callback=self.callback)
        return self.optim_iter_samples 

    def local_exploration(self, logq_func=None, dlogq_func=None):
        """Perform local exploration."""
        self.high_iw_idx = np.where(self.weights >= 1 / self.draws + self.local_thresh * median_abs_deviation(self.weights))[0]
        self.high_iw_samples = self.nf_samples[self.high_iw_idx, ...]
        self.high_weights = self.weights[self.high_iw_idx]
        print(f'Number of points we perform additional local exploration around = {len(self.high_iw_idx)}')

        self.local_samples = np.empty((0, np.shape(self.high_iw_samples)[1]))
        self.local_weights = np.array([])
        self.modified_weights = np.array([])

        for i, sample in enumerate(self.high_iw_samples):
            sample = sample.reshape(-1, len(sample))
            if self.local_grad:
                if dlogq_func is None:
                    raise Exception('Using gradient-based exploration requires you to supply dlogq_func.')
                self.log_weight_grad = self.target_dlogp(sample.astype(np.float64)) - dlogq_func(sample.astype(np.float64))
            elif not self.local_grad:
                if logq_func is None:
                    raise Exception('Gradient-free approximates gradients with finite difference. Requires you to supply logq_func.')
                self.log_weight_grad = approx_fprime(sample, self.target_logp, np.finfo(float).eps) - approx_fprime(sample, logq_func, np.finfo(float).eps)

            self.log_weight_grad = np.asarray(self.log_weight_grad).astype(np.float64)
            delta = 1.0 * self.local_step_size
            proposed_step = sample + delta * self.log_weight_grad
            line_search_iter = 0

            while self.target_logp(proposed_step) < self.target_logp(sample):
                delta = delta / 2.0
                proposed_step = sample + delta * self.log_weight_grad
                line_search_iter += 1
                if line_search_iter >= self.max_line_search:
                    break
                
            self.local_weights = np.append(self.local_weights,
                                           self.high_weights[i] * np.exp(self.target_logp(proposed_step)) / (np.exp(self.target_logp(proposed_step)) + np.exp(self.target_logp(sample))))
            self.modified_weights = np.append(self.modified_weights,
                                              self.high_weights[i] * np.exp(self.target_logp(sample)) / (np.exp(self.target_logp(proposed_step)) + np.exp(self.target_logp(sample))))
            self.local_samples = np.append(self.local_samples, proposed_step, axis=0)

        self.weights[self.high_iw_idx] = self.modified_weights
    
    def initialize_nf(self):
        """Intialize the first NF approx, by fitting to the prior samples."""
        val_idx = int((1 - self.frac_validate) * self.prior_samples.shape[0])

        self.nf_model = GIS(torch.from_numpy(self.prior_samples[:val_idx, ...].astype(np.float32)),
                            torch.from_numpy(self.prior_samples[val_idx:, ...].astype(np.float32)),
                            iteration=self.iteration, alpha=None, verbose=self.verbose, n_component=self.n_component,
                            interp_nbin=self.interp_nbin, KDE=self.KDE, bw_factor=self.bw_factor,
                            edge_bins=self.edge_bins, ndata_wT=self.ndata_wT, MSWD_max_iter=self.MSWD_max_iter,
                            NBfirstlayer=self.NBfirstlayer, logit=self.logit, Whiten=self.Whiten,
                            batchsize=self.batchsize, nocuda=self.nocuda, patch=self.patch, shape=self.shape)
        
        self.nf_samples, self.logq = self.nf_model.sample(self.draws, device=torch.device('cpu'))
        self.nf_samples = self.nf_samples.numpy().astype(np.float64)
        self.logq = self.logq.numpy().astype(np.float64)
        self.weighted_samples = np.append(self.weighted_samples, self.nf_samples, axis=0)
        self.all_logq = np.append(self.all_logq, self.logq)
        self.get_posterior_logp()
        log_weight = self.posterior_logp - self.logq
        self.evidence = np.mean(np.exp(log_weight))
        
        if self.pareto:
            psiw = az.psislw(log_weight)
            self.weights = np.exp(psiw[0])
        elif not self.pareto:
            self.weights = np.exp(log_weight)
            self.weights = np.clip(self.weights, 0, np.mean(self.weights) * len(self.weights)**self.k_trunc)

        self.weights = self.weights / np.sum(self.weights)
        self.importance_weights = np.copy(self.weights)
        if self.init_local:
            # 07/04/21 Note the arguments here are temporary until we have SINF derivatives! Modify once those are implemented.
            self.local_exploration(logq_func=None, dlogq_func=lambda x: approx_fprime(x.squeeze(), self.sinf_logq, np.finfo(float).eps))
            self.weighted_samples = np.append(self.weighted_samples, self.local_samples, axis=0)
            self.importance_weights = np.copy(self.weights)
            self.importance_weights = np.append(self.importance_weights, self.local_weights)

        self.nf_models.append(self.nf_model)

    def initialize_lbfgs(self):
        """Initialize using L-BFGS optimization and Hessian."""
        self.map_dict, self.scipy_opt = find_MAP(model=self.model, method='L-BFGS-B', return_raw=True)
        self.mu_map = []
        for v in self.variables:
            self.mu_map.append(self.map_dict[v.name])
        self.mu_map = np.array(self.mu_map).squeeze()
        print(self.mu_map)

        if self.init_method == 'lbfgs':
            self.hess_inv = self.scipy_opt.hess_inv.todense()
        if self.init_method == 'map+laplace':
            self.hess_inv = np.linalg.inv(self.target_hessian(self.mu_map.reshape(-1, len(self.mu_map))))

        self.weighted_samples = np.random.multivariate_normal(self.mu_map, self.hess_inv, size=self.draws)
        self.nf_samples = np.copy(self.weighted_samples)
        self.get_posterior_logp()
        log_weight = self.posterior_logp - multivariate_normal.logpdf(self.nf_samples, self.mu_map, self.hess_inv)
        self.evidence = np.mean(np.exp(log_weight))
        
        if self.pareto:
            psiw = az.psislw(log_weight)
            self.weights = np.exp(psiw[0])
        elif not self.pareto:
            self.weights = np.exp(log_weight)
            self.weights = np.clip(self.weights, 0, np.mean(self.weights) * len(self.weights)**self.k_trunc)

        self.weights = self.weights / np.sum(self.weights)
        self.importance_weights = np.copy(self.weights)
        if self.init_local:
            self.local_exploration(None, dlogq_func=jax.grad(lambda x: self.logq_fr_el2o(x, self.mu_map, self.hess_inv)))
            self.importance_weights = np.copy(self.weights)
            self.weighted_samples = np.append(self.weighted_samples, self.local_samples, axis=0)
            self.importance_weights = np.append(self.importance_weights, self.local_weights)
            
        self.all_logq = np.array([])
        self.nf_models = []
        
    def logq_fr_el2o(self, z, mu, Sigma):
        """Logq for full-rank Gaussian family."""
        return jnp.reshape(jax.scipy.stats.multivariate_normal.logpdf(z, mu, Sigma), ()) 

    def get_map_laplace(self):
        """Find the MAP+Laplace solution for the model."""
        self.map_dict = find_MAP(model=self.model, method='L-BFGS-B')
        self.mu_map = []
        for v in self.variables:
            self.mu_map.append(self.map_dict[v.name])
        self.mu_map = np.array(self.mu_map).squeeze()
        print(np.shape(self.mu_map))
        self.Sigma_map = np.linalg.inv(self.target_hessian(self.mu_map.reshape(-1, len(self.mu_map))))
        print(f'MAP estimate = {self.map_dict}')
        
    def run_el2o(self):
        """Run the EL2O algorithm, assuming you've got the MAP+Laplace solution."""

        self.mu_k = self.mu_map
        self.Sigma_k = self.Sigma_map
        self.EL2O = [1e10, 1]

        self.zk = np.random.multivariate_normal(self.mu_k, self.Sigma_k)
        self.zk = self.zk.reshape(-1, len(self.zk))

        while self.EL2O[-1] > self.absEL2O and abs((self.EL2O[-1] - self.EL2O[-2]) / self.EL2O[-1]) > self.fracEL2O:

            self.zk = np.vstack((self.zk, np.random.multivariate_normal(self.mu_k.squeeze(), self.Sigma_k)))
            Nk = len(self.zk)
            self.Sigma_k = np.linalg.inv(np.sum(self.target_hessian(self.zk), axis=0) / Nk)

            temp = 0
            for j in range(Nk):
                temp += np.dot(self.Sigma_k, self.target_dlogp(self.zk[j, :].reshape(-1, len(self.zk[j, :])))) + self.zk[j, :].reshape(-1, len(self.zk[j, :]))
            self.mu_k = temp / Nk

            self.EL2O = np.append(self.EL2O, 1 / (len(self.zk)) * (np.sum((self.target_logp(self.zk) -
                                                                           jax.vmap(lambda x: self.logq_fr_el2o(x, self.mu_k, self.Sigma_k), in_axes=0)(self.zk))**2) +
                                                                   np.sum((self.target_dlogp(self.zk) - 
                                                                           jax.vmap(jax.grad(lambda x: self.logq_fr_el2o(x, self.mu_k, self.Sigma_k)), in_axes=0)(self.zk))**2) +
                                                                   np.sum((self.target_hessian(self.zk) - 
                                                                           jax.vmap(jax.hessian(lambda x: self.logq_fr_el2o(x, self.mu_k, self.Sigma_k)), in_axes=0)(self.zk))**2)
            ))

        print(f'Final EL2O mu = {self.mu_k}')
        print(f'Final EL2O Sigma = {self.Sigma_k}')
        self.weighted_samples = np.random.multivariate_normal(self.mu_k.squeeze(), self.Sigma_k, size=self.draws)
        self.nf_samples = np.copy(self.weighted_samples)

        self.get_posterior_logp()
        log_weight = self.posterior_logp - multivariate_normal.logpdf(self.nf_samples, self.mu_k.squeeze(), self.Sigma_k)
        self.evidence = np.mean(np.exp(log_weight))

        if self.pareto:
            psiw = az.psislw(log_weight)
            self.weights = np.exp(psiw[0])
        elif not self.pareto:
            self.weights = np.exp(log_weight)
            self.weights = np.clip(self.weights, 0, np.mean(self.weights) * len(self.weights)**self.k_trunc)

        self.weights = self.weights / np.sum(self.weights)
        self.importance_weights = np.copy(self.weights)
        
        if self.init_local:
            self.local_exploration(logq_func=None, dlogq_func=jax.grad(lambda x: self.logq_fr_el2o(x, self.mu_k, self.Sigma_k)))
            self.weighted_samples = np.append(self.weighted_samples, self.local_samples, axis=0)
            self.importance_weights = np.copy(self.weights)
            self.importance_weights = np.append(self.importance_weights, self.local_weights)

        self.all_logq = np.array([])
        self.nf_models = []

    def fit_nf(self):
        """Fit the NF model for a given iteration after initialization."""
        val_idx = int((1 - self.frac_validate) * self.weighted_samples.shape[0])
        
        self.nf_model = GIS(torch.from_numpy(self.weighted_samples[:val_idx, ...].astype(np.float32)),
                            torch.from_numpy(self.weighted_samples[val_idx:, ...].astype(np.float32)),
                            weight_train=torch.from_numpy(self.importance_weights[:val_idx, ...].astype(np.float32)),
                            weight_validate=torch.from_numpy(self.importance_weights[val_idx:, ...].astype(np.float32)),
                            iteration=self.iteration, alpha=self.alpha, verbose=self.verbose, n_component=self.n_component,
                            interp_nbin=self.interp_nbin, KDE=self.KDE, bw_factor=self.bw_factor,
                            edge_bins=self.edge_bins, ndata_wT=self.ndata_wT, MSWD_max_iter=self.MSWD_max_iter,
                            NBfirstlayer=self.NBfirstlayer, logit=self.logit, Whiten=self.Whiten,
                            batchsize=self.batchsize, nocuda=self.nocuda, patch=self.patch, shape=self.shape)
        
        self.nf_samples, self.logq = self.nf_model.sample(self.draws, device=torch.device('cpu'))
        self.nf_samples = self.nf_samples.numpy().astype(np.float64)
        self.logq = self.logq.numpy().astype(np.float64)
        self.weighted_samples = np.append(self.weighted_samples, self.nf_samples, axis=0)
        self.all_logq =	np.append(self.all_logq, self.logq)
        self.get_posterior_logp()
        log_weight = self.posterior_logp - self.logq
        self.evidence = np.mean(np.exp(log_weight))

        if self.pareto:
            psiw = az.psislw(log_weight)
            self.weights = np.exp(psiw[0])
        elif not self.pareto:
            self.weights = np.exp(log_weight)
            self.weights = np.clip(self.weights, 0, np.mean(self.weights) * len(self.weights)**self.k_trunc)

        self.weights = self.weights / np.sum(self.weights)
        self.importance_weights = np.append(self.importance_weights, self.weights)
        if self.nf_local_iter > 0:
            # 07/04/21 Note the arguments here are temporary until we have SINF derivatives! Modify once those are implemented.
            self.local_exploration(logq_func=None, dlogq_func=lambda x: approx_fprime(x.squeeze(), self.sinf_logq, np.finfo(float).eps))
            self.weighted_samples = np.append(self.weighted_samples, self.local_samples, axis=0)
            self.importance_weights[-len(self.weights):] = self.weights
            self.importance_weights = np.append(self.importance_weights, self.local_weights)
            
        self.nf_models.append(self.nf_model)

    def reinitialize_nf(self):
        """Draw a fresh set of samples from the most recent NF fit. Used to start a set of NF fits without local exploration."""
        self.nf_samples, self.logq = self.nf_model.sample(self.draws, device=torch.device('cpu'))
        self.nf_samples = self.nf_samples.numpy().astype(np.float64)
        self.logq = self.logq.numpy().astype(np.float64)
        self.weighted_samples = np.copy(self.nf_samples)
        self.all_logq = np.copy(self.logq)
        self.get_posterior_logp()
        log_weight = self.posterior_logp - self.logq

        if self.pareto:
            psiw = az.psislw(log_weight)
            self.weights = np.exp(psiw[0])
        elif not self.pareto:
            self.weights = np.exp(log_weight)
            self.weights = np.clip(self.weights, 0, np.mean(self.weights) * len(self.weights)**self.k_trunc)

        self.weights = self.weights / np.sum(self.weights)
        self.importance_weights = np.copy(self.weights)
        
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
