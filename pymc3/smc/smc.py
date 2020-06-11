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
from scipy.special import logsumexp
from fastprogress.fastprogress import progress_bar
import multiprocessing as mp
import warnings
from theano import function as theano_function

from ..model import modelcontext, Point
from ..parallel_sampling import _cpu_count
from ..theanof import inputvars, make_shared_replacements
from ..vartypes import discrete_types
from ..sampling import sample_prior_predictive
from ..theanof import floatX, join_nonshared_inputs
from ..step_methods.arraystep import metrop_select
from ..step_methods.metropolis import MultivariateNormalProposal
from ..backends.ndarray import NDArray
from ..backends.base import MultiTrace

EXPERIMENTAL_WARNING = (
    "Warning: SMC-ABC methods are experimental step methods and not yet"
    " recommended for use in PyMC3!"
)


class SMC:
    def __init__(
        self,
        draws=1000,
        kernel="metropolis",
        n_steps=25,
        parallel=False,
        start=None,
        cores=None,
        tune_steps=True,
        p_acc_rate=0.99,
        threshold=0.5,
        epsilon=1.0,
        dist_func="absolute_error",
        sum_stat="Identity",
        progressbar=False,
        model=None,
        random_seed=-1,
    ):

        self.draws = draws
        self.kernel = kernel
        self.n_steps = n_steps
        self.parallel = parallel
        self.start = start
        self.cores = cores
        self.tune_steps = tune_steps
        self.p_acc_rate = p_acc_rate
        self.threshold = threshold
        self.epsilon = epsilon
        self.dist_func = dist_func
        self.sum_stat = sum_stat
        self.progressbar = progressbar
        self.model = model
        self.random_seed = random_seed

        self.model = modelcontext(model)

        if self.random_seed != -1:
            np.random.seed(self.random_seed)

        if self.cores is None:
            self.cores = _cpu_count()

        self.beta = 0
        self.max_steps = n_steps
        self.proposed = draws * n_steps
        self.acc_rate = 1
        self.acc_per_chain = np.ones(self.draws)
        self.model.marginal_log_likelihood = 0
        self.variables = inputvars(self.model.vars)
        dimension = sum(v.dsize for v in self.variables)
        self.scalings = np.ones(self.draws) * min(1, 2.38 ** 2 / dimension)
        self.discrete = np.concatenate(
            [[v.dtype in discrete_types] * (v.dsize or 1) for v in self.variables]
        )
        self.any_discrete = self.discrete.any()
        self.all_discrete = self.discrete.all()

    def initialize_population(self):
        """
        Create an initial population from the prior distribution
        """
        population = []
        var_info = OrderedDict()
        if self.start is None:
            init_rnd = sample_prior_predictive(
                self.draws, var_names=[v.name for v in self.model.unobserved_RVs], model=self.model,
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
        """
        Set up the likelihood logp function based on the chosen kernel
        """
        shared = make_shared_replacements(self.variables, self.model)
        self.prior_logp = logp_forw([self.model.varlogpt], self.variables, shared)

        if self.kernel.lower() == "abc":
            warnings.warn(EXPERIMENTAL_WARNING)
            if len(self.model.observed_RVs) != 1:
                warnings.warn("SMC-ABC only works properly with models with one observed variable")
            simulator = self.model.observed_RVs[0]
            self.likelihood_logp = PseudoLikelihood(
                self.epsilon,
                simulator.observations,
                simulator.distribution.function,
                [v.name for v in simulator.distribution.params],
                self.model,
                self.var_info,
                self.variables,
                self.dist_func,
                self.sum_stat,
            )
        elif self.kernel.lower() == "metropolis":
            self.likelihood_logp = logp_forw([self.model.datalogpt], self.variables, shared)

    def initialize_logp(self):
        """
        initialize the prior and likelihood log probabilities
        """
        if self.parallel and self.cores > 1:
            self.pool = mp.Pool(processes=self.cores)
            priors = self.pool.starmap(self.prior_logp, [(sample,) for sample in self.posterior])
            likelihoods = self.pool.starmap(
                self.likelihood_logp, [(sample,) for sample in self.posterior]
            )
        else:
            priors = [self.prior_logp(sample) for sample in self.posterior]
            likelihoods = [self.likelihood_logp(sample) for sample in self.posterior]

        self.priors = np.array(priors).squeeze()
        self.likelihoods = np.array(likelihoods).squeeze()

    def update_weights_beta(self):
        """
        Calculate the next inverse temperature (beta), the importance weights based on current beta
        and tempered likelihood and updates the marginal likelihood estimation
        """
        low_beta = old_beta = self.beta
        up_beta = 2.0
        rN = int(len(self.likelihoods) * self.threshold)

        while up_beta - low_beta > 1e-6:
            new_beta = (low_beta + up_beta) / 2.0
            log_weights_un = (new_beta - old_beta) * self.likelihoods
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
            log_weights_un = (new_beta - old_beta) * self.likelihoods
            log_weights = log_weights_un - logsumexp(log_weights_un)

        ll_max = np.max(log_weights_un)
        self.model.marginal_log_likelihood += ll_max + np.log(
            np.exp(log_weights_un - ll_max).mean()
        )
        self.beta = new_beta
        self.weights = np.exp(log_weights)

    def resample(self):
        """
        Resample particles based on importance weights
        """
        resampling_indexes = np.random.choice(
            np.arange(self.draws), size=self.draws, p=self.weights
        )
        self.posterior = self.posterior[resampling_indexes]
        self.priors = self.priors[resampling_indexes]
        self.likelihoods = self.likelihoods[resampling_indexes]
        self.tempered_logp = self.priors + self.likelihoods * self.beta
        self.acc_per_chain = self.acc_per_chain[resampling_indexes]
        self.scalings = self.scalings[resampling_indexes]

    def update_proposal(self):
        """
        Update proposal based on the covariance matrix from tempered posterior
        """
        cov = np.cov(self.posterior, bias=False, rowvar=0)
        cov = np.atleast_2d(cov)
        cov += 1e-6 * np.eye(cov.shape[0])
        if np.isnan(cov).any() or np.isinf(cov).any():
            raise ValueError('Sample covariances not valid! Likely "draws" is too small!')
        self.proposal = MultivariateNormalProposal(cov)

    def tune(self):
        """
        Tune scaling and n_steps based on the acceptance rate.
        """
        ave_scaling = np.exp(np.log(self.scalings.mean()) + (self.acc_per_chain.mean() - 0.234))
        self.scalings = 0.5 * (
            ave_scaling + np.exp(np.log(self.scalings) + (self.acc_per_chain - 0.234))
        )

        if self.tune_steps:
            acc_rate = max(1.0 / self.proposed, self.acc_rate)
            self.n_steps = min(
                self.max_steps, max(2, int(np.log(1 - self.p_acc_rate) / np.log(1 - acc_rate))),
            )

        self.proposed = self.draws * self.n_steps

    def mutate(self):
        """
        Perform mutation step, i.e. apply selected kernel
        """
        parameters = (
            self.proposal,
            self.scalings,
            self.any_discrete,
            self.all_discrete,
            self.discrete,
            self.n_steps,
            self.prior_logp,
            self.likelihood_logp,
            self.beta,
        )
        if self.parallel and self.cores > 1:
            results = self.pool.starmap(
                metrop_kernel,
                [
                    (
                        self.posterior[draw],
                        self.tempered_logp[draw],
                        self.priors[draw],
                        self.likelihoods[draw],
                        draw,
                        *parameters,
                    )
                    for draw in range(self.draws)
                ],
            )
        else:
            iterator = range(self.draws)
            if self.progressbar:
                iterator = progress_bar(iterator, display=self.progressbar)
            results = [
                metrop_kernel(
                    self.posterior[draw],
                    self.tempered_logp[draw],
                    self.priors[draw],
                    self.likelihoods[draw],
                    draw,
                    *parameters,
                )
                for draw in iterator
            ]
        posterior, acc_list, priors, likelihoods = zip(*results)
        self.posterior = np.array(posterior)
        self.priors = np.array(priors)
        self.likelihoods = np.array(likelihoods)
        self.acc_per_chain = np.array(acc_list)
        self.acc_rate = np.mean(acc_list)

    def posterior_to_trace(self):
        """
        Save results into a PyMC3 trace
        """
        lenght_pos = len(self.posterior)
        varnames = [v.name for v in self.variables]

        with self.model:
            strace = NDArray(self.model)
            strace.setup(lenght_pos, 0)
        for i in range(lenght_pos):
            value = []
            size = 0
            for var in varnames:
                shape, new_size = self.var_info[var]
                value.append(self.posterior[i][size : size + new_size].reshape(shape))
                size += new_size
            strace.record({k: v for k, v in zip(varnames, value)})
        return MultiTrace([strace])


def metrop_kernel(
    q_old,
    old_tempered_logp,
    old_prior,
    old_likelihood,
    draw,
    proposal,
    scalings,
    any_discrete,
    all_discrete,
    discrete,
    n_steps,
    prior_logp,
    likelihood_logp,
    beta,
):
    """
    Metropolis kernel
    """
    deltas = np.squeeze(proposal(n_steps) * scalings[draw])

    accepted = 0
    for n_step in range(n_steps):
        delta = deltas[n_step]

        if any_discrete:
            if all_discrete:
                delta = np.round(delta, 0).astype("int64")
                q_old = q_old.astype("int64")
                q_new = (q_old + delta).astype("int64")
            else:
                delta[discrete] = np.round(delta[discrete], 0)
                q_new = floatX(q_old + delta)
        else:
            q_new = floatX(q_old + delta)

        ll = likelihood_logp(q_new)
        pl = prior_logp(q_new)

        new_tempered_logp = pl + ll * beta

        q_old, accept = metrop_select(new_tempered_logp - old_tempered_logp, q_new, q_old)

        if accept:
            accepted += 1
            old_prior = pl
            old_likelihood = ll
            old_tempered_logp = new_tempered_logp

    return q_old, accepted / n_steps, old_prior, old_likelihood


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
    Pseudo Likelihood
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
    ):
        """
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
        distance : str or callable
            Distance function. The only available option is ``gaussian_kernel``
        sum_stat: str or callable
            Summary statistics. Available options are ``indentity``, ``sorted``, ``mean``,
            ``median``. The user can pass any valid Python function
        """
        self.epsilon = epsilon
        self.function = function
        self.params = params
        self.model = model
        self.var_info = var_info
        self.variables = variables
        self.varnames = [v.name for v in self.variables]
        self.unobserved_RVs = [v.name for v in self.model.unobserved_RVs]
        self.get_unobserved_fn = self.model.fastfn(self.model.unobserved_RVs)

        if sum_stat == "identity":
            self.sum_stat = lambda x: x
        elif sum_stat == "sorted":
            self.sum_stat = np.sort
        elif sum_stat == "mean":
            self.sum_stat = np.mean
        elif sum_stat == "median":
            self.sum_stat = np.median
        elif hasattr(sum_stat, "__call__"):
            self.sum_stat = sum_stat
        else:
            raise ValueError(f"The summary statistics {sum_stat} is not implemented")

        self.observations = self.sum_stat(observations)

        if distance == "gaussian_kernel":
            self.distance = self.gaussian_kernel
        elif hasattr(distance, "__call__"):
            self.distance = distance
        else:
            raise ValueError(f"The distance metric {distance} is not implemented")

    def posterior_to_function(self, posterior):
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

    def gaussian_kernel(self, obs_data, sim_data):
        return np.sum(-0.5 * ((obs_data - sim_data) / self.epsilon) ** 2)

    def __call__(self, posterior):
        func_parameters = self.posterior_to_function(posterior)
        sim_data = self.sum_stat(self.function(**func_parameters))
        return self.distance(self.observations, sim_data)
