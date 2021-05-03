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
import copy

import numpy as np
import theano.tensor as tt

from scipy.special import logsumexp
from scipy.stats import multivariate_normal, median_abs_deviation
from scipy.optimize import minimize, approx_fprime
from theano import function as theano_function
import arviz as az
import jax
import jax.numpy as jnp
from jax.experimental import optimizers as jax_optimizers
import pymc3.nfmc.posdef as posdef

from pymc3.tuning.scaling import find_hessian
from pymc3.tuning.starting import find_MAP
from pymc3.backends.ndarray import NDArray
from pymc3.blocking import ArrayOrdering, DictToArrayBijection
from pymc3.model import Point, modelcontext, set_data
from pymc3.distributions.distribution import draw_values, to_tuple
from pymc3.sampling import sample_prior_predictive
from pymc3.theanof import (
    floatX,
    inputvars,
    join_nonshared_inputs,
    make_shared_replacements,
    gradient,
    hessian,
)
from pymc3.util import (
    check_start_vals,
    get_default_varnames,
    get_var_name,
    update_start_vals,
)
from pymc3.vartypes import discrete_types, typefilter


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
        start=None,
        init_EL2O='adam',
        use_hess_EL2O=False,
        mean_field_EL2O=False,
        absEL2O=1e-10,
        fracEL2O=1e-2,
        scipy_map_method='L-BFGS-B',
        adam_lr=1e-3,
        adam_b1=0.9,
        adam_b2=0.999,
        adam_eps=1.0e-8,
        adam_steps=1000,
        simulator=None,
        model_data=None,
        sim_data_cov=None,
        sim_size=None,
        sim_params=None,
        sim_start=None,
        sim_optim_method='lbfgs',
        sim_tol=0.01,
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

        # Init method params.
        self.init_method = init_method
        self.init_samples = init_samples
        self.start = start
        self.init_EL2O = init_EL2O
        self.mean_field_EL2O = mean_field_EL2O
        self.use_hess_EL2O = use_hess_EL2O
        self.absEL2O = absEL2O
        self.fracEL2O = fracEL2O
        self.scipy_map_method = scipy_map_method
        self.adam_lr = adam_lr
        self.adam_b1 = adam_b1
        self.adam_b2 = adam_b2
        self.adam_eps = adam_eps
        self.adam_steps = adam_steps
        self.simulator = simulator
        self.model_data = model_data
        self.sim_data_cov = sim_data_cov
        self.sim_size = sim_size
        self.sim_params = sim_params
        self.sim_start = sim_start
        self.sim_optim_method = sim_optim_method
        self.sim_tol = sim_tol
        
        self.pareto = pareto

        # Local exploration params.
        self.local_thresh = local_thresh
        self.local_step_size = local_step_size
        self.local_grad = local_grad
        self.init_local = init_local
        self.nf_local_iter = nf_local_iter
        self.max_line_search = max_line_search

        self.random_seed = random_seed
        self.chain = chain

        # Separating out so I can keep track. These are SINF params.
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

        print(f'Prior samples = {np.shape(self.prior_samples)}')
            
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
        minimize(self.optim_target_logp, x0=sample, method=self.scipy_map_method,
                 options={'maxiter': self.optim_iter, 'ftol': self.ftol, 'gtol': self.gtol},
                 jac=self.optim_target_dlogp, callback=self.callback)
        return self.optim_iter_samples 

    def get_MAP(self, map_method='adam', map_start=None):
        """Get the MAP estimate."""
        if map_start is None:
            map_start = self.start
        if map_method == 'adam':
            self.optimization_start()
            opt_init, opt_update, get_params = jax_optimizers.adam(step_size=self.adam_lr, b1=self.adam_b1,
                                                                   b2=self.adam_b2, eps=self.adam_eps)
            opt_state = opt_init(map_start)

            for i in range(self.adam_steps):
                value, opt_state, update_params = self.update_adam(i, opt_state, opt_update, get_params)
                target_diff = np.abs((value - np.float64(self.adam_logp(floatX(update_params)))) / max(value, np.float64(self.adam_logp(floatX(update_params)))))
                if target_diff <= self.ftol:
                    print(f'ADAM converged at step {i}')
                    break
            vars = get_default_varnames(self.model.unobserved_RVs, include_transformed=True)
            map_dict = {var.name: value for var, value in zip(vars, self.model.fastfn(vars)(self.bij.rmap(update_params.squeeze())))}
        else:
            map_dict = find_MAP(start=map_start, model=self.model, method=self.scipy_map_method)
        return map_dict
            
    def get_sim_data(self, point):
        """Generate simulated data using the supplied simulator function."""
        size = to_tuple(self.sim_size)
        params = draw_values([*self.params], point=point, size=1) 
        forward_sim = self.simulator(*params)
        self.sim_data = forward_sim + np.random.multivariate_normal(mu=0, cov=self.sim_data_cov)
        self.sim_params = np.array([])
        for p in params:
            self.sim_params = np.append(self.sim_params, p)
        self.sim_params = self.sim_params.squeeze()
        
    def simulation_init(self):
        """Initialize the model using a simulation-based init (generalization of the Ensemble Kalman filter)."""
        assert self.model_data is not None

        self.data_MAP = self.get_MAP(map_method=self.sim_optim_method, start=self.start)
        self.data_map_arr = np.array([])
        for v in self.variables:
            self.data_map_arr = np.append(self.data_map_arr, self.data_MAP[v.name])
        self.data_map_arr = self.data_map_arr.squeeze()
        if self.sim_start is None:
            # Check this - only really want MAP of the hyper-params. Maybe can't have self.sim_start as None.
            self.sim_start = self.data_MAP
        self.sim_samples = np.empty((0, len(self.data_map_arr)))    

        self.sim_logp_diff = 1000
        sim_iter = 1
        while self.sim_logp_diff > self.sim_tol:
            print(f'Running simulation init iteration: {sim_iter}.')
            self.get_sim_data(point=self.sim_start)
            set_data({self.model_data.keys(): self.sim_data}, model=self.model)
            self.sim_MAP = self.get_MAP(map_method=self.sim_optim_method, start=self.sim_start)
            self.sim_map_arr = np.array([])
            for v in self.variables:
                self.sim_map_arr = np.append(self.sim_map_arr, self.sim_MAP[v.name])
            self.sim_map_arr = self.sim_map_arr.squeeze()
            self.map_diff = self.sim_map_arr - self.sim_params
            self.sim_update = self.data_map_arr + self.map_diff
            self.sim_samples = np.append(self.sim_samples, self.sim_update)
        
            set_data({self.model_data.keys(): self.sim_data}, model=self.model)
            self.old_logp = self.get_posterior_logp(self.sim_params.reshape(-1, len(self.sim_params)))
            self.new_logp = self.get_posterior_logp(self.sim_update.reshape(-1, len(self.sim_update)))
            self.sim_logp_diff = abs(self.old_logp - self.new_logp) / max(abs(self.old_logp), abs(self.new_logp), 1)
            sim_stage += 1

        self.mu_map = 1.0 * self.sim_update
        self.hess_inv = np.linalg.inv(self.target_hessian(self.mu_map.reshape(-1, len(self.mu_map))))
        self.weighted_samples = np.random.multivariate_normal(self.mu_map, self.hess_inv, size=self.draws)
        self.nf_samples = np.copy(self.weighted_samples)
        self.get_posterior_logp()
        self.log_weight = self.posterior_logp - multivariate_normal.logpdf(self.nf_samples, self.mu_map.squeeze(), self.hess_inv, allow_singular=True)
        self.log_evidence = logsumexp(self.log_weight) - np.log(len(self.log_weight))
        self.evidence = np.exp(self.log_evidence)
        self.log_weight = self.log_weight - logsumexp(self.log_weight)

        if self.pareto:
            psiw = az.psislw(self.log_weight)
            self.log_weight = psiw[0]
            self.weights = np.exp(self.log_weight)
        elif not self.pareto:
            self.log_weight = np.clip(self.log_weight, a_min=None, a_max=logsumexp(self.log_weight) + (self.k_trunc - 1) * np.log(len(self.log_weight)))
            self.log_weight = logsumexp(self.log_weight) - np.log(len(self.log_weight))
            self.weights = np.exp(self.log_weight)

        self.weights = self.weights / np.sum(self.weights)
        self.importance_weights = np.copy(self.weights)
        if self.init_local:
            self.local_exploration(None, dlogq_func=jax.grad(lambda x: self.logq_fr_el2o(x, self.mu_map, self.hess_inv)))
            self.importance_weights = np.copy(self.weights)
            self.weighted_samples = np.append(self.weighted_samples, self.local_samples, axis=0)
            self.importance_weights = np.append(self.importance_weights, self.local_weights)
            self.nf_samples = np.append(self.nf_samples, self.local_samples, axis=0)
            self.log_weight = np.append(self.log_weight, self.local_log_weight)
            self.weights = np.append(self.weights, self.local_weights)

        self.all_logq = np.array([])
        self.nf_models = []
            
    def optimization_start(self):
        """Setup for optimization starting point."""
        disc_vars = list(typefilter(self.variables, discrete_types))
        allinmodel(self.variables, self.model)
        self.start = copy.deepcopy(self.start)
        if self.start is None:
            self.start = self.model.test_point
        else:
            update_start_vals(self.start, self.model.test_point, self.model)
        check_start_vals(self.start, self.model)
        print(self.start)

        self.start = Point(self.start, model=self.model)
        print(self.start)
        self.bij = DictToArrayBijection(ArrayOrdering(self.variables), self.start)
        self.start = self.bij.map(self.start)
        self.adam_logp = self.bij.mapf(self.model.fastlogp_nojac)
        self.adam_dlogp = self.bij.mapf(self.model.fastdlogp_nojac(self.variables))
        print(self.start)
    
    def update_adam(self, step, opt_state, opt_update, get_params):
        """Jax implemented ADAM update."""
        params = np.asarray(get_params(opt_state)).astype(np.float64)
        #params = params.reshape(-1, len(params))
        value = np.float64(self.adam_logp(floatX(params.squeeze())))
        grads = -1 * jnp.asarray(np.float64(self.adam_dlogp(floatX(params.squeeze()))))
        opt_state = opt_update(step, grads, opt_state)
        update_params = np.asarray(get_params(opt_state)).astype(np.float64)
        #update_params = update_params.reshape(-1, len(update_params))
        return value, opt_state, update_params

    def adam_map_hess(self):
        """Use ADAM to find the MAP solution."""
        self.optimization_start()
        opt_init, opt_update, get_params = jax_optimizers.adam(step_size=self.adam_lr, b1=self.adam_b1,
                                                               b2=self.adam_b2, eps=self.adam_eps)
        opt_state = opt_init(self.start)

        for i in range(self.adam_steps):
            value, opt_state, update_params = self.update_adam(i, opt_state, opt_update, get_params)
            target_diff = np.abs((value - np.float64(self.adam_logp(floatX(update_params)))) / max(value, np.float64(self.adam_logp(floatX(update_params)))))
            if target_diff <= self.ftol:
                print(f'ADAM converged at step {i}')
                break
        vars = get_default_varnames(self.model.unobserved_RVs, include_transformed=True)
        print(self.variables)
        self.map_dict = {var.name: value for var, value in zip(vars, self.model.fastfn(vars)(self.bij.rmap(update_params.squeeze())))}
        self.mu_map = np.array([])
        for v in self.variables:
            self.mu_map = np.append(self.mu_map, self.map_dict[v.name])
        self.mu_map = self.mu_map.squeeze()
        
        print(f'BIJ rmap = {self.map_dict}')
        print(f'ADAM map solution = {self.mu_map}')
        self.hess_inv = np.linalg.inv(-1 * self.target_hessian(self.mu_map.reshape(-1, len(self.mu_map))))
        if not posdef.isPD(self.hess_inv):
            print(f'Autodiff Hessian is not positive semi-definite. Building Hessian with L-BFGS run starting from ADAM MAP.')
            self.scipy_opt = minimize(self.optim_target_logp, x0=self.mu_map, method='L-BFGS-B',
                                      options={'maxiter': self.optim_iter, 'ftol': self.ftol, 'gtol': self.gtol},
                                      jac=self.optim_target_dlogp)
            print(f'lbfgs Hessian inverse = {self.scipy_opt.hess_inv.todense()}')
            if posdef.isPD(self.scipy_opt.hess_inv.todense()):
                self.hess_inv = self.scipy_opt.hess_inv.todense()
            else:
                print(f'L-BFGS-B Hessian was not positive semi-definite - resorting to finding the nearest PSD matrix.')
                self.hess_inv = posdef.nearestPD(self.hess_inv)
        print(f'Final MAP solution = {self.mu_map}')
        print(f'Inverse Hessian at MAP = {self.hess_inv}')

        self.weighted_samples = np.random.multivariate_normal(self.mu_map, self.hess_inv, size=self.draws)
        self.nf_samples = np.copy(self.weighted_samples)
        self.get_posterior_logp()
        self.log_weight = self.posterior_logp - multivariate_normal.logpdf(self.nf_samples, self.mu_map.squeeze(), self.hess_inv, allow_singular=True)
        print(f'Unormalized log weights = {self.log_weight}')
        self.log_evidence = logsumexp(self.log_weight) - np.log(len(self.log_weight))
        self.evidence = np.exp(self.log_evidence)
        self.log_weight = self.log_weight - logsumexp(self.log_weight)
        print(f'Normalized log weights = {self.log_weight}')

        if self.pareto:
            psiw = az.psislw(self.log_weight)
            self.log_weight = psiw[0]
            print(f'Pareto logw = {self.log_weight}')
            self.weights = np.exp(self.log_weight)
            print(f'Pareto w = {self.weights}')
        elif not self.pareto:
            self.log_weight = np.clip(self.log_weight, a_min=None, a_max=logsumexp(self.log_weight) + (self.k_trunc - 1) * np.log(len(self.log_weight)))
            self.log_weight = self.log_weight - logsumexp(self.log_weight)
            self.weights = np.exp(self.log_weight)

        print(f'Non-zero Adam init weights = {self.weights[self.weights != 0.0]}')
        self.weights = self.weights / np.sum(self.weights)
        self.importance_weights = np.copy(self.weights)
        if self.init_local:
            self.local_exploration(None, dlogq_func=jax.grad(lambda x: self.logq_fr_el2o(x, self.mu_map, self.hess_inv)))
            self.importance_weights = np.copy(self.weights)
            self.weighted_samples = np.append(self.weighted_samples, self.local_samples, axis=0)
            self.importance_weights = np.append(self.importance_weights, self.local_weights)
            self.nf_samples = np.append(self.nf_samples, self.local_samples, axis=0)
            self.log_weight = np.append(self.log_weight, self.local_log_weight)
            self.weights = np.append(self.weights, self.local_weights)

        self.all_logq = np.array([])
        self.nf_models = []
        
    def local_exploration(self, logq_func=None, dlogq_func=None):
        """Perform local exploration."""
        self.high_iw_idx = np.where(self.log_weight >= np.log(self.local_thresh) - np.log(self.draws))[0]
        #self.high_iw_idx = np.where(self.weights >= self.local_thresh / self.draws)[0]
        self.num_local = len(self.high_iw_idx)
        self.high_iw_samples = self.nf_samples[self.high_iw_idx, ...]
        self.high_log_weight = self.log_weight[self.high_iw_idx]
        self.high_weights = self.weights[self.high_iw_idx]
        print(f'Number of points we perform additional local exploration around = {self.num_local}')

        self.local_samples = np.empty((0, np.shape(self.high_iw_samples)[1]))
        self.local_log_weight = np.array([])
        self.modified_log_weight = np.array([])
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

            local_log_w = self.high_log_weight[i] + self.target_logp(proposed_step) - np.log(np.exp(self.target_logp(proposed_step)) + np.exp(self.target_logp(sample)))
            modif_log_w = self.high_log_weight[i] + self.target_logp(sample) - np.log(np.exp(self.target_logp(proposed_step)) + np.exp(self.target_logp(sample)))
            self.local_log_weight = np.append(self.local_log_weight, local_log_w)
            self.modified_log_weight = np.append(self.modified_log_weight, modif_log_w)
            self.local_weights = np.append(self.local_weights, np.exp(local_log_w))
            self.modified_weights = np.append(self.modified_weights, np.exp(modif_log_w))
            self.local_samples = np.append(self.local_samples, proposed_step, axis=0)

        self.log_weight[self.high_iw_idx] = self.modified_log_weight
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
        self.log_weight = self.posterior_logp - self.logq
        self.log_evidence = logsumexp(self.log_weight) - np.log(len(self.log_weight))
        self.evidence = np.exp(self.log_evidence)
        self.log_weight = self.log_weight - logsumexp(self.log_weight)
        
        if self.pareto:
            psiw = az.psislw(self.log_weight)
            self.log_weight = psiw[0]
            self.weights = np.exp(self.log_weight)
        elif not self.pareto:
            self.log_weight = np.clip(self.log_weight, a_min=None, a_max=logsumexp(self.log_weight) + (self.k_trunc - 1) * np.log(len(self.log_weight)))
            self.log_weight = self.log_weight - logsumexp(self.log_weight)
            self.weights = np.exp(self.log_weight)

        self.weights = self.weights / np.sum(self.weights)
        self.importance_weights = np.copy(self.weights)
        if self.init_local:
            # 07/04/21 Note the arguments here are temporary until we have SINF derivatives! Modify once those are implemented.
            self.local_exploration(logq_func=None, dlogq_func=lambda x: approx_fprime(x.squeeze(), self.sinf_logq, np.finfo(float).eps))
            self.weighted_samples = np.append(self.weighted_samples, self.local_samples, axis=0)
            self.importance_weights = np.copy(self.weights)
            self.importance_weights = np.append(self.importance_weights, self.local_weights)
            self.nf_samples = np.append(self.nf_samples, self.local_samples, axis=0)
            self.log_weight = np.append(self.log_weight, self.local_log_weight)
            self.weights = np.append(self.weights, self.local_weights)
            
        self.nf_models.append(self.nf_model)

    def initialize_map_hess(self):
        """Initialize using scipy MAP optimization and Hessian."""
        self.map_dict, self.scipy_opt = find_MAP(start=self.start, model=self.model, method=self.scipy_map_method, return_raw=True)
        print(f'lbfgs map dict = {self.map_dict}')
        print(self.variables)
        self.mu_map = []
        for v in self.variables:
            self.mu_map.append(self.map_dict[v.name])
        self.mu_map = np.array(self.mu_map).squeeze()
        print(self.mu_map)

        if self.init_method == 'lbfgs':
            assert self.scipy_map_method == 'L-BFGS-B'
            self.hess_inv = self.scipy_opt.hess_inv.todense()
        if self.init_method == 'map+laplace':
            self.hess_inv = np.linalg.inv(self.target_hessian(self.mu_map.reshape(-1, len(self.mu_map))))

        if not posdef.isPD(self.hess_inv):
            print(f'Inverse Hessian is not positive semi-definite - resorting to nearest PSD matrix.')
            self.hess_inv = posdef.nearestPD(self.hess_inv)
            
        self.weighted_samples = np.random.multivariate_normal(self.mu_map, self.hess_inv, size=self.draws)
        self.nf_samples = np.copy(self.weighted_samples)
        self.get_posterior_logp()
        self.log_weight = self.posterior_logp - multivariate_normal.logpdf(self.nf_samples, self.mu_map, self.hess_inv, allow_singular=True)
        self.log_evidence = logsumexp(self.log_weight) - np.log(len(self.log_weight))
        self.evidence = np.exp(self.log_evidence)
        self.log_weight = self.log_weight - logsumexp(self.log_weight)
        
        if self.pareto:
            psiw = az.psislw(self.log_weight)
            self.log_weight = psiw[0]
            self.weights = np.exp(self.log_weight)
        elif not self.pareto:
            self.log_weight = np.clip(self.log_weight, a_min=None, a_max=logsumexp(self.log_weight) + (self.k_trunc - 1) * np.log(len(self.log_weight)))
            self.log_weight = self.log_weight - logsumexp(self.log_weight)
            self.weights = np.exp(self.log_weight)

        self.weights = self.weights / np.sum(self.weights)
        self.importance_weights = np.copy(self.weights)
        if self.init_local:
            self.local_exploration(None, dlogq_func=jax.grad(lambda x: self.logq_fr_el2o(x, self.mu_map, self.hess_inv)))
            self.importance_weights = np.copy(self.weights)
            self.weighted_samples = np.append(self.weighted_samples, self.local_samples, axis=0)
            self.importance_weights = np.append(self.importance_weights, self.local_weights)
            self.nf_samples = np.append(self.nf_samples, self.local_samples, axis=0)
            self.log_weight = np.append(self.log_weight, self.local_log_weight)
            self.weights = np.append(self.weights, self.local_weights)
            
        self.all_logq = np.array([])
        self.nf_models = []
        
    def logq_fr_el2o(self, z, mu, Sigma):
        """Logq for full-rank Gaussian family."""
        return jnp.reshape(jax.scipy.stats.multivariate_normal.logpdf(z, mu, Sigma), ()) 

    def get_map_laplace(self):
        """Find the MAP+Laplace solution for the model."""
        if self.init_EL2O == 'adam':
            self.optimization_start()
            opt_init, opt_update, get_params = jax_optimizers.adam(step_size=self.adam_lr, b1=self.adam_b1,
                                                                   b2=self.adam_b2, eps=self.adam_eps)
            opt_state = opt_init(self.start)

            for i in range(self.adam_steps):
                value, opt_state, update_params = self.update_adam(i, opt_state, opt_update, get_params)
                target_diff = np.abs((value - np.float64(self.adam_logp(floatX(update_params)))) / max(value, np.float64(self.adam_logp(floatX(update_params)))))
                if target_diff <= self.ftol:
                    print(f'ADAM converged at step {i}')
                    break
            vars = get_default_varnames(self.model.unobserved_RVs, include_transformed=True)
            self.map_dict = {var.name: value for var, value in zip(vars, self.model.fastfn(vars)(self.bij.rmap(update_params.squeeze())))}
        else:
            self.map_dict = find_MAP(start=self.start, model=self.model, method=self.scipy_map_method)

        self.mu_map = np.array([])
        for v in self.variables:
            self.mu_map = np.append(self.mu_map, self.map_dict[v.name])
        self.mu_map = self.mu_map.squeeze()
        self.Sigma_map = np.linalg.inv(self.target_hessian(self.mu_map.reshape(-1, len(self.mu_map))))

        print(f'MAP estimate = {self.map_dict}')
        print(f'Sigma estimate at MAP = {self.Sigma_map}')
        
    def run_el2o(self):
        """Run the EL2O algorithm, assuming you've got the MAP+Laplace solution."""

        self.mu_k = self.mu_map
        self.Sigma_k = self.Sigma_map
        self.EL2O = [1e10, 1]

        self.zk = np.random.multivariate_normal(self.mu_k, self.Sigma_k, size=len(self.mu_k))
        #self.zk = self.zk.reshape(-1, len(self.zk))

        while self.EL2O[-1] > self.absEL2O and abs((self.EL2O[-1] - self.EL2O[-2]) / self.EL2O[-1]) > self.fracEL2O:

            self.zk = np.vstack((self.zk, np.random.multivariate_normal(self.mu_k.squeeze(), self.Sigma_k)))
            Nk = len(self.zk)
            if not self.use_hess_EL2O:
                temp1 = 0
                temp2 = 0
                for k in range(Nk):
                    temp1 += np.outer(self.zk[k, :] - np.mean(self.zk, axis=0), self.zk[k, :] - np.mean(self.zk, axis=0))
                    temp2 += np.outer(self.zk[k, :] - np.mean(self.zk, axis=0), self.target_dlogp(self.zk[k, :].reshape(-1, len(self.zk[k, :]))))
                if self.mean_field_EL2O:
                    self.Sigma_k = -1 * np.diag(temp2) / np.diag(temp1)
                    self.Sigma_k = 1.0 / self.Sigma_k
                    self.Sigma_k = self.Sigma_k * np.eye(len(self.Sigma_k))
                elif not self.mean_field_EL2O:
                    self.Sigma_k = -1 * np.matmul(np.linalg.inv(temp1), temp2)
                    self.Sigma_k = np.linalg.inv(self.Sigma_k)
            elif self.use_hess_EL2O:
                self.Sigma_k = np.linalg.inv(np.sum(self.target_hessian(self.zk), axis=0) / Nk)
                if self.mean_field_EL2O:
                    self.Sigma_k = np.diag(self.Sigma_k) * np.eye(len(self.Sigma_k))

            temp = 0
            for j in range(Nk):
                temp += np.matmul(self.Sigma_k, self.target_dlogp(self.zk[j, :].reshape(-1, len(self.zk[j, :]))))
            self.mu_k = np.mean(self.zk, axis=0) + temp / Nk
            print(f'Current EL2O mu = {self.mu_k}')
            
            self.EL2O = np.append(self.EL2O, 1 / (len(self.zk)) * (np.sum((self.target_logp(self.zk) -
                                                                           jax.vmap(lambda x: self.logq_fr_el2o(x, self.mu_k, self.Sigma_k), in_axes=0)(self.zk))**2) +
                                                                   np.sum((self.target_dlogp(self.zk) -
                                                                           jax.vmap(jax.grad(lambda x: self.logq_fr_el2o(x, self.mu_k, self.Sigma_k)), in_axes=0)(self.zk))**2)
            ))

        print(f'Final EL2O mu = {self.mu_k}')
        print(f'Final EL2O Sigma = {self.Sigma_k}')
        self.weighted_samples = np.random.multivariate_normal(self.mu_k.squeeze(), self.Sigma_k, size=self.draws)
        self.nf_samples = np.copy(self.weighted_samples)

        self.get_posterior_logp()
        self.log_weight = self.posterior_logp - multivariate_normal.logpdf(self.nf_samples, self.mu_k.squeeze(), self.Sigma_k, allow_singular=True)
        self.log_evidence = logsumexp(self.log_weight) - np.log(len(self.log_weight))
        self.evidence = np.exp(self.log_evidence)
        self.log_weight = self.log_weight - logsumexp(self.log_weight)

        if self.pareto:
            psiw = az.psislw(self.log_weight)
            self.log_weight = psiw[0]
            self.weights = np.exp(self.log_weight)
        elif not self.pareto:
            self.log_weight = np.clip(self.log_weight, a_min=None, a_max=logsumexp(self.log_weight) + (self.k_trunc - 1) * np.log(len(self.log_weight)))
            self.log_weight = self.log_weight - logsumexp(self.log_weight)
            self.weights = np.exp(self.log_weight)

        self.weights = self.weights / np.sum(self.weights)
        self.importance_weights = np.copy(self.weights)
        
        if self.init_local:
            self.local_exploration(logq_func=None, dlogq_func=jax.grad(lambda x: self.logq_fr_el2o(x, self.mu_k, self.Sigma_k)))
            self.weighted_samples = np.append(self.weighted_samples, self.local_samples, axis=0)
            self.importance_weights = np.copy(self.weights)
            self.importance_weights = np.append(self.importance_weights, self.local_weights)
            self.nf_samples = np.append(self.nf_samples, self.local_samples, axis=0)
            self.log_weight = np.append(self.log_weight, self.local_log_weight)
            self.weights = np.append(self.weights, self.local_weights)
            
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
        self.log_weight = self.posterior_logp - self.logq
        self.log_evidence = logsumexp(self.log_weight) - np.log(len(self.log_weight))
        self.evidence = np.exp(self.log_evidence)
        self.log_weight = self.log_weight - logsumexp(self.log_weight)

        if self.pareto:
            psiw = az.psislw(self.log_weight)
            self.log_weight = psiw[0]
            self.weights = np.exp(self.log_weight)
        elif not self.pareto:
            self.log_weight = np.clip(self.log_weight, a_min=None, a_max=logsumexp(self.log_weight) + (self.k_trunc - 1) * np.log(len(self.log_weight)))
            self.log_weight = self.log_weight - logsumexp(self.log_weight)
            self.weights = np.exp(self.log_weight)

        self.weights = self.weights / np.sum(self.weights)
        self.importance_weights = np.append(self.importance_weights, self.weights)
        if self.nf_local_iter > 0:
            # 07/04/21 Note the arguments here are temporary until we have SINF derivatives! Modify once those are implemented.
            self.local_exploration(logq_func=None, dlogq_func=lambda x: approx_fprime(x.squeeze(), self.sinf_logq, np.finfo(float).eps))
            self.weighted_samples = np.append(self.weighted_samples, self.local_samples, axis=0)
            self.importance_weights[-len(self.weights):] = self.weights
            self.importance_weights = np.append(self.importance_weights, self.local_weights)
            self.nf_samples = np.append(self.nf_samples, self.local_samples, axis=0)
            self.log_weight = np.append(self.log_weight, self.local_log_weight)
            self.weights = np.append(self.weights, self.local_weights)
            
        self.nf_models.append(self.nf_model)

    def reinitialize_nf(self):
        """Draw a fresh set of samples from the most recent NF fit. Used to start a set of NF fits without local exploration."""
        self.nf_samples, self.logq = self.nf_model.sample(self.draws, device=torch.device('cpu'))
        self.nf_samples = self.nf_samples.numpy().astype(np.float64)
        self.logq = self.logq.numpy().astype(np.float64)
        self.weighted_samples = np.copy(self.nf_samples)
        self.all_logq = np.copy(self.logq)
        self.get_posterior_logp()
        self.log_weight = self.posterior_logp - self.logq
        self.log_evidence = logsumexp(self.log_weight) - np.log(len(self.log_weight))
        self.evidence = np.exp(self.log_evidence)
        self.log_weight = self.log_weight - logsumexp(self.log_weight)
        
        if self.pareto:
            psiw = az.psislw(self.log_weight)
            self.log_weight = psiw[0]
            self.weights = np.exp(self.log_weight)
        elif not self.pareto:
            self.log_weight = np.clip(self.log_weight, a_min=None, a_max=logsumexp(self.log_weight) + (self.k_trunc - 1) * np.log(len(self.log_weight)))
            self.log_weight = self.log_weight - logsumexp(self.log_weight)
            self.weights = np.exp(self.log_weight)

        self.weights = self.weights / np.sum(self.weights)
        self.importance_weights = np.copy(self.weights)

    def final_nf(self):
        """Final NF fit used to ensure the target distribution is the asymptotic distribution of our importance sampling."""
        if self.num_local > 0:

            print('Performing final NF fit without local exploration.')
            self.nf_local_iter = 0
            self.fit_nf()

        resampling_indexes = np.random.choice(
            np.arange(len(self.weights)), size=self.draws, p=self.weights/np.sum(self.weights)
        )
        self.posterior = self.nf_samples[resampling_indexes, ...]
            
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
        print(f'posterior to trace varnames = {varnames}')
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

def allinmodel(vars, model):
    notin = [v for v in vars if v not in model.vars]
    if notin:
        notin = list(map(get_var_name, notin))
        raise ValueError("Some variables not in the model: " + str(notin))
