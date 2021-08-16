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

import aesara.tensor as at
import numpy as np

from aesara import function as aesara_function
from aesara.graph.basic import clone_replace
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from pymc3.aesaraf import (
    floatX,
    inputvars,
    join_nonshared_inputs,
    make_shared_replacements,
)
from pymc3.backends.ndarray import NDArray
from pymc3.blocking import DictToArrayBijection
from pymc3.model import Point, modelcontext
from pymc3.sampling import sample_prior_predictive
from pymc3.vartypes import discrete_types


class SMC:
    """Sequential Monte Carlo with Independent Metropolis-Hastings and ABC kernels."""

    def __init__(
        self,
        draws=2000,
        n_steps=25,
        start=None,
        tune_steps=True,
        p_acc_rate=0.85,
        threshold=0.5,
        model=None,
        random_seed=-1,
        chain=0,
    ):

        self.draws = draws
        self.n_steps = n_steps
        self.start = start
        self.tune_steps = tune_steps
        self.p_acc_rate = p_acc_rate
        self.threshold = threshold
        self.model = model
        self.random_seed = random_seed
        self.chain = chain

        self.model = modelcontext(model)

        if self.random_seed != -1:
            np.random.seed(self.random_seed)

        self.var_info = None
        self.posterior = None
        self.prior_logp = None
        self.likelihood_logp = None
        self.posterior_logp = None
        self.prior_logp_func = None
        self.likelihood_logp_func = None
        self.log_marginal_likelihood = 0

        self.beta = 0
        self.max_steps = n_steps
        self.proposed = draws * n_steps
        self.acc_rate = 1
        self.variables = inputvars(self.model.value_vars)
        self.weights = np.ones(self.draws) / self.draws
        self.cov = None

    def initialize_population(self):
        """Create an initial population from the prior distribution."""
        population = []
        var_info = OrderedDict()
        if self.start is None:
            init_rnd = sample_prior_predictive(
                self.draws,
                var_names=[v.name for v in self.model.unobserved_value_vars],
                model=self.model,
            )
        else:
            init_rnd = self.start

        init = self.model.initial_point

        for v in self.variables:
            var_info[v.name] = (init[v.name].shape, init[v.name].size)

        for i in range(self.draws):
            point = Point({v.name: init_rnd[v.name][i] for v in self.variables}, model=self.model)
            population.append(DictToArrayBijection.map(point).data)

        self.posterior = np.array(floatX(population))
        self.var_info = var_info

    def setup_kernel(self):
        """Set up the likelihood logp function based on the chosen kernel."""
        initial_values = self.model.initial_point
        shared = make_shared_replacements(initial_values, self.variables, self.model)

        self.prior_logp_func = logp_forw(
            initial_values, [self.model.varlogpt], self.variables, shared
        )
        self.likelihood_logp_func = logp_forw(
            initial_values, [self.model.datalogpt], self.variables, shared
        )

    def initialize_logp(self):
        """Initialize the prior and likelihood log probabilities."""
        priors = [self.prior_logp_func(sample) for sample in self.posterior]
        likelihoods = [self.likelihood_logp_func(sample) for sample in self.posterior]

        self.prior_logp = np.array(priors).squeeze()
        self.likelihood_logp = np.array(likelihoods).squeeze()

    def update_weights_beta(self):
        """Calculate the next inverse temperature (beta).

        The importance weights based on current beta and tempered likelihood and updates the
        marginal likelihood estimate.
        """
        low_beta = old_beta = self.beta
        up_beta = 2.0
        rN = int(len(self.likelihood_logp) * self.threshold)

        while up_beta - low_beta > 1e-6:
            new_beta = (low_beta + up_beta) / 2.0
            log_weights_un = (new_beta - old_beta) * self.likelihood_logp
            log_weights = log_weights_un - logsumexp(log_weights_un)
            ESS = int(np.exp(-logsumexp(log_weights * 2)))
            if ESS == rN:
                break
            elif ESS < rN:
                up_beta = new_beta
            else:
                low_beta = new_beta
        if new_beta >= 1:
            new_beta = 1
            log_weights_un = (new_beta - old_beta) * self.likelihood_logp
            log_weights = log_weights_un - logsumexp(log_weights_un)

        self.log_marginal_likelihood += logsumexp(log_weights_un) - np.log(self.draws)
        self.beta = new_beta
        self.weights = np.exp(log_weights)
        # We normalize again to correct for small numerical errors that might build up
        self.weights /= self.weights.sum()

    def resample(self):
        """Resample particles based on importance weights."""
        resampling_indexes = np.random.choice(
            np.arange(self.draws), size=self.draws, p=self.weights
        )

        self.posterior = self.posterior[resampling_indexes]
        self.prior_logp = self.prior_logp[resampling_indexes]
        self.likelihood_logp = self.likelihood_logp[resampling_indexes]
        self.posterior_logp = self.prior_logp + self.likelihood_logp * self.beta

    def update_proposal(self):
        """Update proposal based on the covariance matrix from tempered posterior."""
        cov = np.cov(self.posterior, ddof=0, rowvar=0)
        cov = np.atleast_2d(cov)
        cov += 1e-6 * np.eye(cov.shape[0])
        if np.isnan(cov).any() or np.isinf(cov).any():
            raise ValueError('Sample covariances not valid! Likely "draws" is too small!')
        self.cov = cov

    def tune(self):
        """Tune n_steps based on the acceptance rate."""
        if self.tune_steps:
            acc_rate = max(1.0 / self.proposed, self.acc_rate)
            self.n_steps = min(
                self.max_steps,
                max(2, int(np.log(1 - self.p_acc_rate) / np.log(1 - acc_rate))),
            )

        self.proposed = self.draws * self.n_steps

    def mutate(self):
        """Independent Metropolis-Hastings perturbation."""
        ac_ = np.empty((self.n_steps, self.draws))

        log_R = np.log(np.random.rand(self.n_steps, self.draws))

        # The proposal distribution is a MVNormal, with mean and covariance computed from the previous tempered posterior
        dist = multivariate_normal(self.posterior.mean(axis=0), self.cov)

        for n_step in range(self.n_steps):
            # The proposal is independent from the current point.
            # We have to take that into account to compute the Metropolis-Hastings acceptance
            proposal = floatX(dist.rvs(size=self.draws))
            proposal = proposal.reshape(len(proposal), -1)
            # To do that we compute the logp of moving to a new point
            forward = dist.logpdf(proposal)
            # And to going back from that new point
            backward = multivariate_normal(proposal.mean(axis=0), self.cov).logpdf(self.posterior)
            ll = np.array([self.likelihood_logp_func(prop) for prop in proposal])
            pl = np.array([self.prior_logp_func(prop) for prop in proposal])
            proposal_logp = pl + ll * self.beta
            accepted = log_R[n_step] < (
                (proposal_logp + backward) - (self.posterior_logp + forward)
            )
            ac_[n_step] = accepted
            self.posterior[accepted] = proposal[accepted]
            self.posterior_logp[accepted] = proposal_logp[accepted]
            self.prior_logp[accepted] = pl[accepted]
            self.likelihood_logp[accepted] = ll[accepted]

        self.acc_rate = np.mean(ac_)

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
            for varname in varnames:
                shape, new_size = self.var_info[varname]
                var_samples = self.posterior[i][size : size + new_size]
                # Round discrete variable samples. The rounded values were the ones
                # actually used in the logp evaluations (see logp_forw)
                var = self.model[varname]
                if var.dtype in discrete_types:
                    var_samples = np.round(var_samples).astype(var.dtype)
                value.append(var_samples.reshape(shape))
                size += new_size
            strace.record(point={k: v for k, v in zip(varnames, value)})
        return strace


def logp_forw(point, out_vars, vars, shared):
    """Compile Aesara function of the model and the input and output variables.

    Parameters
    ----------
    out_vars: List
        containing :class:`pymc3.Distribution` for the output variables
    vars: List
        containing :class:`pymc3.Distribution` for the input variables
    shared: List
        containing :class:`aesara.tensor.Tensor` for depended shared data
    """

    # Convert expected input of discrete variables to (rounded) floats
    if any(var.dtype in discrete_types for var in vars):
        replace_int_to_float = {}
        replace_float_to_round = {}
        new_vars = []
        for var in vars:
            if var.dtype in discrete_types:
                float_var = at.TensorType("floatX", var.broadcastable)(var.name)
                replace_int_to_float[var] = float_var
                new_vars.append(float_var)

                round_float_var = at.round(float_var)
                round_float_var.name = var.name
                replace_float_to_round[float_var] = round_float_var
            else:
                new_vars.append(var)

        replace_int_to_float.update(shared)
        replace_float_to_round.update(shared)
        out_vars = clone_replace(out_vars, replace_int_to_float, strict=False)
        out_vars = clone_replace(out_vars, replace_float_to_round)
        vars = new_vars

    out_list, inarray0 = join_nonshared_inputs(point, out_vars, vars, shared)
    f = aesara_function([inarray0], out_list[0])
    f.trust_input = True
    return f
