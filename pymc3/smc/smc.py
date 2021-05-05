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


class SMC:
    """Sequential Monte Carlo with Independent Metropolis-Hastings and ABC kernels."""

    def __init__(
        self,
        draws=2000,
        kernel="metropolis",
        n_steps=25,
        start=None,
        tune_steps=True,
        p_acc_rate=0.85,
        threshold=0.5,
        save_sim_data=False,
        save_log_pseudolikelihood=True,
        model=None,
        random_seed=-1,
        chain=0,
    ):

        self.draws = draws
        self.kernel = kernel.lower()
        self.n_steps = n_steps
        self.start = start
        self.tune_steps = tune_steps
        self.p_acc_rate = p_acc_rate
        self.threshold = threshold
        self.save_sim_data = save_sim_data
        self.save_log_pseudolikelihood = save_log_pseudolikelihood
        self.model = model
        self.random_seed = random_seed
        self.chain = chain

        self.model = modelcontext(model)

        if self.random_seed != -1:
            np.random.seed(self.random_seed)

        self.beta = 0
        self.max_steps = n_steps
        self.proposed = draws * n_steps
        self.acc_rate = 1
        self.variables = inputvars(self.model.vars)
        self.weights = np.ones(self.draws) / self.draws
        self.log_marginal_likelihood = 0
        self.sim_data = []
        self.log_pseudolikelihood = []

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

        self.posterior = np.array(floatX(population))
        self.var_info = var_info

    def setup_kernel(self):
        """Set up the likelihood logp function based on the chosen kernel."""
        shared = make_shared_replacements(self.variables, self.model)

        if self.kernel == "abc":
            factors = [var.logpt for var in self.model.free_RVs]
            factors += [tt.sum(factor) for factor in self.model.potentials]
            self.prior_logp_func = logp_forw([tt.sum(factors)], self.variables, shared)
            simulator = self.model.observed_RVs[0]
            distance = simulator.distribution.distance
            sum_stat = simulator.distribution.sum_stat
            self.likelihood_logp_func = PseudoLikelihood(
                simulator.distribution.epsilon,
                simulator.observations,
                simulator.distribution.function,
                [v.name for v in simulator.distribution.params],
                self.model,
                self.var_info,
                self.variables,
                distance,
                sum_stat,
                self.draws,
                self.save_sim_data,
                self.save_log_pseudolikelihood,
            )
        elif self.kernel == "metropolis":
            self.prior_logp_func = logp_forw([self.model.varlogpt], self.variables, shared)
            self.likelihood_logp_func = logp_forw([self.model.datalogpt], self.variables, shared)

    def initialize_logp(self):
        """Initialize the prior and likelihood log probabilities."""
        priors = [self.prior_logp_func(sample) for sample in self.posterior]
        likelihoods = [self.likelihood_logp_func(sample) for sample in self.posterior]

        self.prior_logp = np.array(priors).squeeze()
        self.likelihood_logp = np.array(likelihoods).squeeze()

        if self.kernel == "abc" and self.save_sim_data:
            self.sim_data = self.likelihood_logp_func.get_data()

        if self.kernel == "abc" and self.save_log_pseudolikelihood:
            self.log_pseudolikelihood = self.likelihood_logp_func.get_lpl()

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
        if self.kernel == "abc" and self.save_sim_data:
            self.sim_data = self.sim_data[resampling_indexes]
        if self.kernel == "abc" and self.save_log_pseudolikelihood:
            self.log_pseudolikelihood = self.log_pseudolikelihood[resampling_indexes]

    def update_proposal(self):
        """Update proposal based on the covariance matrix from tempered posterior."""
        cov = np.cov(self.posterior, ddof=0, aweights=self.weights, rowvar=0)
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

            if self.kernel == "abc" and self.save_sim_data:
                self.sim_data[accepted] = self.likelihood_logp_func.get_data()[accepted]

            if self.kernel == "abc" and self.save_log_pseudolikelihood:
                self.log_pseudolikelihood[accepted] = self.likelihood_logp_func.get_lpl()[accepted]

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
