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

from scipy.linalg import cholesky
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
from pymc3.backends.ndarray import NDArray, point_list_to_multitrace
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

class NFO:
    """Sequencial NF Bayesian Optimization."""
    def __init__(
        self,
        draws=500,
        init_draws=500,
        model=None,
        init_method='prior',
        init_samples=None,
        start=None,
        random_seed=-1,
        frac_validate=0.1,
        iteration=None,
        alpha=(0,0),
        k_trunc=0.25,
        verbose=False,
        n_component=None,
        interp_nbin=None,
        KDE=True,
        bw_factor_min=0.5,
        bw_factor_max=2.5,
        bw_factor_num=11,
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
        redraw=True,
    ):

        self.draws = draws
        self.init_draws = init_draws
        self.model = model

        # Init method params.
        self.init_method = init_method
        self.init_samples = init_samples
        self.start = start

        self.random_seed = random_seed

        # Set the torch seed.
        if self.random_seed != 1:
            np.random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)

        # Separating out so I can keep track. These are SINF params.
        assert 0.0 <= frac_validate <= 1.0
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
        self.bw_factors = np.linspace(bw_factor_min, bw_factor_max, bw_factor_num)
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

        #whether to redraw samples at every iteration, used for BO testing
        self.redraw = redraw

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
                self.init_draws,
                var_names=[v.name for v in self.model.unobserved_RVs],
                model=self.model,
            )

            for i in range(self.init_draws):

                point = Point({v.name: init_rnd[v.name][i] for v in self.variables}, model=self.model)
                population.append(self.model.dict_to_array(point))

            self.prior_samples = np.array(floatX(population))

        elif self.init_samples is not None:

            self.prior_samples = np.copy(self.init_samples)

        self.weighted_samples = np.copy(self.prior_samples)
        self.nf_samples = np.copy(self.weighted_samples)
        self.get_posterior_logp()
        self.get_prior_logp()
        self.log_weight = self.posterior_logp - self.prior_logp
        self.log_evidence = logsumexp(self.log_weight) - np.log(len(self.log_weight))
        self.evidence = np.exp(self.log_evidence)
        self.log_weight = self.log_weight - self.log_evidence

        #same as in fitnf but prior~q
        self.log_weight_pq_num = self.posterior_logp + 2*self.prior_logp
        self.log_weight_pq_den = 3*self.prior_logp
        self.log_evidence_pq = logsumexp(self.log_weight_pq_num) - logsumexp(self.log_weight_pq_den)
        self.evidence_pq = np.exp(self.log_evidence_pq)

        self.regularize_weights()
        self.init_weights_cleanup(None, lambda x: self.prior_dlogp(x))
        self.q_ess = self.calculate_ess(self.log_weight)
        self.total_ess = self.calculate_ess(self.sinf_logw)

        self.all_logq = np.array([])
        self.nf_models = []

    def setup_logp(self):
        """Set up the prior and likelihood logp functions, and derivatives."""
        shared = make_shared_replacements(self.variables, self.model)

        self.prior_logp_func = logp_forw([self.model.varlogpt], self.variables, shared)
        self.prior_dlogp_func = logp_forw([gradient(self.model.varlogpt, self.variables)], self.variables, shared)
        self.likelihood_logp_func = logp_forw([self.model.datalogpt], self.variables, shared)
        self.posterior_logp_func = logp_forw([self.model.logpt], self.variables, shared)
        self.posterior_dlogp_func = logp_forw([gradient(self.model.logpt, self.variables)], self.variables, shared)
        self.posterior_hessian_func = logp_forw([hessian(self.model.logpt, self.variables)], self.variables, shared)
        self.posterior_logp_nojac = logp_forw([self.model.logp_nojact], self.variables, shared)
        self.posterior_dlogp_nojac = logp_forw([gradient(self.model.logp_nojact, self.variables)], self.variables, shared)
        self.posterior_hessian_nojac = logp_forw([hessian(self.model.logp_nojact, self.variables)], self.variables, shared)

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

    def sinf_logq(self, param_vals):
        if param_vals.size == 1:
            param_vals = np.array([param_vals])
        sinf_logq = self.nf_model.evaluate_density(torch.from_numpy(param_vals.astype(np.float32))).numpy().astype(np.float64)
        return sinf_logq.item()


    def regularize_weights(self):
        """Apply clipping to importance weights."""
        inf_weights = np.isinf(np.exp(self.log_weight))
        self.log_weight = np.clip(self.log_weight, a_min=None, a_max=logsumexp(self.log_weight[~inf_weights])
                                  - np.log(len(self.log_weight[~inf_weights]))  + self.k_trunc * np.log(len(self.log_weight)))
        self.weights = np.exp(self.log_weight)

    def calculate_ess(self, logw):
        """Calculate ESS given a set of sample weights"""
        logw = logw - logsumexp(logw)
        ess = np.exp(-logsumexp(2 * logw) - np.log(logw.shape[0]))
        return ess

    def calculate_weight_variance(self):
        """Calculates the variance of importance weights for a given q."""
        return np.var(self.weight)

    def init_weights_cleanup(self, logq_func=None, dlogq_func=None):
        """Finish initializing the first importance weights (including possible local exploration)."""
        self.sinf_logw = np.copy(self.log_weight)
        self.importance_weights = np.copy(self.weights)
        if self.init_local:
            self.local_exploration(logq_func=logq_func, dlogq_func=dlogq_func,
                                   log_thresh=np.log(self.local_thresh))
            self.weighted_samples = np.append(self.weighted_samples, self.local_samples, axis=0)
            self.nf_samples = np.append(self.nf_samples, self.local_samples, axis=0)
            self.log_weight = np.append(self.log_weight, self.local_log_weight)
            self.weights = np.append(self.weights, self.local_weights)
            self.sinf_logw = np.copy(self.log_weight)
            self.importance_weights = np.copy(self.weights)


    def fit_nf(self, num_draws):
        """Fit the NF model for a given iteration after initialization."""
        bw_var_weights = []
        bw_pq_weights = []
        bw_nf_models = []
        for bw_factor in self.bw_factors:
            if self.frac_validate > 0.0:
                num_val = int(self.frac_validate * self.weighted_samples.shape[0])
                val_idx = np.random.choice(np.arange(self.weighted_samples.shape[0]), size=num_val, replace=False)
                fit_idx = np.delete(np.arange(self.weighted_samples.shape[0]), val_idx)
                self.train_ess = self.calculate_ess(self.sinf_logw[fit_idx, ...])
                self.nf_model = GIS(torch.from_numpy(self.weighted_samples[fit_idx, ...].astype(np.float32)),
                                    torch.from_numpy(self.weighted_samples[val_idx, ...].astype(np.float32)),
                                    weight_train=torch.from_numpy(self.importance_weights[fit_idx, ...].astype(np.float32)),
                                    weight_validate=torch.from_numpy(self.importance_weights[val_idx, ...].astype(np.float32)),
                                    iteration=self.iteration, alpha=self.alpha, verbose=self.verbose, n_component=self.n_component,
                                    interp_nbin=self.interp_nbin, KDE=self.KDE, bw_factor=bw_factor,
                                    edge_bins=self.edge_bins, ndata_wT=self.ndata_wT, MSWD_max_iter=self.MSWD_max_iter,
                                    NBfirstlayer=self.NBfirstlayer, logit=self.logit, Whiten=self.Whiten,
                                    batchsize=self.batchsize, nocuda=self.nocuda, patch=self.patch, shape=self.shape)
            elif self.frac_validate == 0.0:
                fit_idx = np.arange(self.weighted_samples.shape[0])
                self.train_ess = self.calculate_ess(self.sinf_logw[fit_idx, ...])
                self.nf_model = GIS(torch.from_numpy(self.weighted_samples.astype(np.float32)),
                                    weight_train=torch.from_numpy(self.importance_weights.astype(np.float32)),
                                    iteration=self.iteration, alpha=self.alpha, verbose=self.verbose, n_component=self.n_component,
                                    interp_nbin=self.interp_nbin, KDE=self.KDE, bw_factor=bw_factor,
                                    edge_bins=self.edge_bins, ndata_wT=self.ndata_wT, MSWD_max_iter=self.MSWD_max_iter,
                                    NBfirstlayer=self.NBfirstlayer, logit=self.logit, Whiten=self.Whiten,
                                    batchsize=self.batchsize, nocuda=self.nocuda, patch=self.patch, shape=self.shape)

            if(self.redraw): #do the usual thing

                self.nf_samples, self.logq = self.nf_model.sample(num_draws, device=torch.device('cpu'))
                self.nf_samples = self.nf_samples.numpy().astype(np.float64)
                self.logq = self.logq.numpy().astype(np.float64)
                self.all_logq = np.append(self.all_logq, self.logq)

                self.get_posterior_logp()
            elif(~self.redraw):
                self.train_logp = self.posterior_logp
                #compute logq because we didn't draw new samples (when it wouldn've been computed   automatically)
                self.logq = self.nf_model.evaluate_density(torch.from_numpy(self.weighted_samples[fit_idx, ...].astype(np.float32))).numpy().astype(np.float64)
                self.train_logq = self.logq

            #first estimator of evidence using E_p[1/q]
            self.log_weight = self.posterior_logp - self.logq
            self.log_evidence = logsumexp(self.log_weight) - np.log(len(self.log_weight))
            self.log_weight = self.log_weight - self.log_evidence

            #second estimator of evidence using E_q[pq]/E_q[q^2] to avoid SINF dropping low-p samples
            self.log_weight_pq_num = (self.posterior_logp+2*self.logq)
            self.log_weight_pq_den = 3*self.logq
            self.log_evidence_pq = (logsumexp(self.log_weight_pq_num) - logsumexp(self.log_weight_pq_den)) #length factor unnecessary here

            self.regularize_weights()
            bw_var_weights.append(np.var(self.weights))
            bw_pq_weights.append( sum( (np.exp(self.posterior_logp) -  np.exp(self.log_evidence_pq + self.logq)
                                       )**2
                                     )
                                ) #alternative loss for choosing bw, check for underflow?
            bw_nf_models.append(self.nf_model)

        min_var_idx = bw_var_weights.index(min(bw_var_weights))
        min_pq_idx  = bw_pq_weights.index(min(bw_pq_weights))
        self.nf_model = bw_nf_models[min_var_idx]
        self.min_var_weights = bw_var_weights[min_var_idx]
        self.min_var_bw = self.bw_factors[min_var_idx]
        self.min_pq_bw = self.bw_factors[min_pq_idx]

        if(self.redraw): #do the usual thing
            self.nf_samples, self.logq = self.nf_model.sample(num_draws, device=torch.device('cpu'))
            self.nf_samples = self.nf_samples.numpy().astype(np.float64)
            self.logq = self.logq.numpy().astype(np.float64)
            self.weighted_samples = np.append(self.weighted_samples, self.nf_samples, axis=0)
            self.all_logq = np.append(self.all_logq, self.logq)
            self.get_posterior_logp()
        elif(~self.redraw):
            self.train_logp = self.posterior_logp
            self.logq = self.nf_model.evaluate_density(torch.from_numpy(self.weighted_samples[fit_idx, ...].astype(np.float32))).numpy().astype(np.float64)
            self.train_logq = self.logq

        self.log_weight = self.posterior_logp - self.logq
        self.log_evidence = logsumexp(self.log_weight) - np.log(len(self.log_weight))
        self.log_weight = self.log_weight - self.log_evidence

        #second estimator of evidence using E[pq]/E[q^2] to avoid SINF dropping low-p samples
        #For now we don't actually end up using these weights except to get the evidence, but can later
        self.log_weight_pq_num = (self.posterior_logp+2*self.logq)
        self.log_weight_pq_den = 3*self.logq
        self.log_evidence_pq = (logsumexp(self.log_weight_pq_num) - logsumexp(self.log_weight_pq_den)) #length factor unnecessary here

        self.regularize_weights()

        self.train_logp = self.target_logp(self.weighted_samples[fit_idx, ...])
        self.train_logq = self.nf_model.evaluate_density(torch.from_numpy(self.weighted_samples[fit_idx, ...].astype(np.float32))).numpy().astype(np.float64)

        if(self.redraw):
            self.sinf_logw = np.append(self.sinf_logw, self.log_weight)
            self.importance_weights = np.append(self.importance_weights, self.weights)
        elif(~self.redraw):
            self.sinf_logw = self.log_weight
            self.importance_weights = np.exp(self.sinf_logw - logsumexp(self.sinf_logw))

        self.q_ess = self.calculate_ess(self.log_weight)
        self.total_ess = self.calculate_ess(self.sinf_logw)
        self.nf_models.append(self.nf_model)

    def nf_samples_to_trace(self):
        """Convert NF samples to a trace."""
        lenght_pos = len(self.nf_samples)
        varnames = [v.name for v in self.variables]
        with self.model:
            self.nf_strace = NDArray(name=self.model.name)
            self.nf_strace.setup(lenght_pos, self.chain)
        for i in range(lenght_pos):
            value = []
            size = 0
            for var in varnames:
                shape, new_size = self.var_info[var]
                value.append(self.nf_samples[i][size : size + new_size].reshape(shape))
                size += new_size
            self.nf_strace.record(point={k: v for k, v in zip(varnames, value)})
        self.nf_trace = point_list_to_multitrace(self.nf_strace, model=self.model)

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
