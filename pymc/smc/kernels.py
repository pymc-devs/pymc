#   Copyright 2024 - present The PyMC Developers
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
import abc
import warnings

from abc import ABC
from typing import TypeAlias

import numpy as np
import pytensor.tensor as pt

from pytensor import shared
from pytensor.graph.replace import clone_replace
from pytensor.link.jax import JAXLinker
from pytensor.tensor.random.type import RandomGeneratorType
from rich.progress import TextColumn
from rich.table import Column
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from pymc.blocking import DictToArrayBijection
from pymc.initial_point import make_initial_point_expression
from pymc.model import modelcontext
from pymc.pytensorf import (
    compile,
    floatX,
    join_nonshared_inputs,
    make_shared_replacements,
)
from pymc.sampling.forward import draw
from pymc.step_methods.metropolis import MultivariateNormalProposal
from pymc.vartypes import discrete_types

SMCStats: TypeAlias = dict[str, int | float]
SMCSettings: TypeAlias = dict[str, int | float]


class SMC_KERNEL(ABC):
    """Base class for the Sequential Monte Carlo kernels.

    To create a new SMC kernel you should subclass from this.

    Before sampling, the following methods are called once in order:

        initialize_population
            Choose initial population of SMC particles. Should return a dictionary
            with {var.name : numpy array of size (draws, var.size)}. Defaults
            to sampling from the prior distribution, except for parameters which have custom
            `initval`, in which case that value is used for all SMC particles.
            This method is only called if `start` is not specified.

        _initialize_kernel : default
            Creates initial population of particles in the variable
            `self.tempered_posterior` and populates the `self.var_info` dictionary
            with information about model variables shape and size as
            {var.name : (var.shape, var.size)}.

            The functions `self.prior_logp_func` and `self.likelihood_logp_func` are
            created in this step. These expect a 1D numpy array with the summed
            sizes of each raveled model variable (in the order specified in
            :meth:`pymc.Model.initial_point`).

            Finally, this method computes the log prior and log likelihood for
            the initial particles, and saves them in `self.prior_logp` and
            `self.likelihood_logp`.

            This method should not be modified.

        setup_kernel : optional
            May include any logic that should be performed before sampling
            starts.

    During each sampling stage the following methods are called in order:

        update_beta_and_weights : default
            The inverse temperature self.beta is updated based on the `self.likelihood_logp`
            and `threshold` parameter.

            The importance `self.weights` of each particle are computed from the old and newly
            selected inverse temperature.

            The iteration number stored in `self.iteration` is updated by this method.

            Finally the model `log_marginal_likelihood` of the tempered posterior
            is updated from these weights.

        resample : default
            The particles in `self.posterior` are sampled with replacement based
            on `self.weights`, and the used resampling indexes are saved in
            `self.resampling_indexes`.

            The arrays `self.prior_logp` and `self.likelihood_logp` are rearranged according
            to the order of the resampled particles. `self.tempered_posterior_logp`
            is computed from these and the current `self.beta`.

        tune : optional
            May include logic that should be performed before every mutation step.

        mutate : REQUIRED
            Mutate particles in `self.tempered_posterior`.

            This method is further responsible to update the `self.prior_logp`,
            `self.likelihod_logp` and `self.tempered_posterior_logp`, corresponding
            to each mutated particle.

        sample_stats : default
            Returns important sampling_stats at the end of each stage in a dictionary
            format. This will be saved in the final InferenceData object under `sample_stats`.

    Finally, at the end of sampling the following methods are called:

        _posterior_to_trace : default
            Convert final population of particles to a posterior trace object.
            This method should not be modified.

        sample_settings : default
            Returns important sample_settings at the end of sampling in a dictionary
            format. This will be saved in the final InferenceData object under `sample_stats`.

    """

    stats_dtypes_shapes: dict[str, tuple[type, list]] = {
        "log_marginal_likelihood": (float, []),
        "beta": (float, []),
    }

    def __init__(
        self,
        draws=2000,
        start=None,
        model=None,
        random_seed=None,
        threshold=0.5,
        compile_kwargs: dict | None = None,
    ):
        """
        Initialize the SMC_kernel class.

        Parameters
        ----------
        draws : int, default 2000
            The number of samples to draw from the posterior (i.e. last stage). Also the number of
            independent chains. Defaults to 2000.
        start : dict, or array of dict, default None
            Starting point in parameter space. It should be a list of dict with length `chains`.
            When None (default) the starting point is sampled from the prior distribution, except
            for parameters with a custom `initval`, in which case that value is used.
        model : Model (optional if in ``with`` context).
        random_seed : int, array_like of int, RandomState or Generator, optional
            Value used to initialize the random number generator.
        threshold : float, default 0.5
            Determines the change of beta from stage to stage, i.e.indirectly the number of stages,
            the higher the value of `threshold` the higher the number of stages. Defaults to 0.5.
            It should be between 0 and 1.
        compile_kwargs: dict, optional
            Keyword arguments passed to pytensor.function

        Attributes
        ----------
        self.var_info : dict
            Dictionary that contains information about model variables shape and size.

        """
        self.draws = draws
        self.start = start
        if threshold < 0 or threshold > 1:
            raise ValueError(f"Threshold value {threshold} must be between 0 and 1")
        self.threshold = threshold

        model = modelcontext(model)
        self.rng = np.random.default_rng(seed=random_seed)

        self.variables = model.value_vars

        self.var_info: dict[str, tuple] = {}
        self.tempered_posterior: np.ndarray
        self.prior_logp: np.ndarray | None = None
        self.likelihood_logp: np.ndarray | None = None
        self.tempered_posterior_logp: np.ndarray | None = None
        self.log_marginal_likelihood: float = 0.0
        self.beta = 0.0
        self.iteration = 0
        self.resampling_indexes: np.ndarray | None = None
        self.weights = np.ones(self.draws) / self.draws

        initial_point = model.initial_point(random_seed=self.rng.integers(2**30))

        for v in self.variables:
            self.var_info[v.name] = (initial_point[v.name].shape, initial_point[v.name].size)

        shared = make_shared_replacements(initial_point, self.variables, model)
        compile_kwargs = compile_kwargs if compile_kwargs is not None else {}

        # If a model has no observed variables, the likelihood_logp will have unused inputs, which can be safely
        # ignored.
        compile_kwargs.update({"on_unused_input": "ignore"})

        self.prior_logp_func = _logp_forw(
            initial_point, [model.varlogp], self.variables, shared, compile_kwargs
        )
        self.likelihood_logp_func = _logp_forw(
            initial_point, [model.datalogp], self.variables, shared, compile_kwargs
        )

        prior_expression = make_initial_point_expression(
            free_rvs=model.free_RVs,
            rvs_to_transforms=model.rvs_to_transforms,
            initval_strategies={
                **model.rvs_to_initial_values,
            },
            default_strategy="prior",
            return_transformed=True,
        )

        self._prior_expression = prior_expression
        self._prior_var_names = [model.rvs_to_values[rv].name for rv in model.free_RVs]

    def set_rng(self, rng: np.random.Generator):
        """
        Copy compiled functions, updating their random number generators.

        This is necessary because these functions were compiled once at initialization, then pickled
        and sent to worker processes. Each worker needs its own RNG state to ensure independent sampling,
        so we replace the shared RNGs in the compiled functions with new ones created from the provided `rng`.

        This method copies the functions, so it is expensive and should only be called once per worker!
        """

        def make_rng_swaps(fn, rng):
            shared_rngs = [
                var for var in fn.get_shared() if isinstance(var.type, RandomGeneratorType)
            ]
            n_shared_rngs = len(shared_rngs)
            if n_shared_rngs > 0 and isinstance(fn.maker.linker, JAXLinker):
                raise NotImplementedError(
                    f"JAX rngs cannot be replaced after compilation. {self}.set_rng will fail to "
                    f"properly update random seeds between chains, resulting in non-independent "
                    f"sampling."
                )

            return {
                old_shared_rng: shared(new_rng, borrow=True)
                for old_shared_rng, new_rng in zip(
                    shared_rngs, rng.spawn(n_shared_rngs), strict=True
                )
            }

        self.rng = rng
        self.prior_logp_func = self.prior_logp_func.copy(
            swap=make_rng_swaps(self.prior_logp_func, self.rng)
        )
        self.likelihood_logp_func = self.likelihood_logp_func.copy(
            swap=make_rng_swaps(self.likelihood_logp_func, self.rng)
        )

    def initialize_population(self) -> dict[str, np.ndarray]:
        """Create an initial population from the prior distribution."""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning, message="The effect of Potentials"
            )

            prior_values = draw(self._prior_expression, draws=self.draws, random_seed=self.rng)
            dict_prior = dict(zip(self._prior_var_names, prior_values))

        return dict_prior

    def _initialize_kernel(self):
        """Initialize particles and compute their prior and likelihood logp.

        This method should not be overwritten. If needed, use `setup_kernel` instead.
        """
        if self.start:
            init_rnd = self.start
        else:
            init_rnd = self.initialize_population()

        population = []
        for i in range(self.draws):
            point = {v.name: init_rnd[v.name][i] for v in self.variables}
            population.append(DictToArrayBijection.map(point).data)

        self.tempered_posterior = np.array(floatX(population))

        # Evaluate prior and likelihood for initial particles
        priors = [self.prior_logp_func(sample) for sample in self.tempered_posterior]
        likelihoods = [self.likelihood_logp_func(sample) for sample in self.tempered_posterior]

        self.prior_logp = np.array(priors).squeeze()
        self.likelihood_logp = np.array(likelihoods).squeeze()

    def setup_kernel(self):
        """Perform setup logic once before sampling starts."""
        pass

    def update_beta_and_weights(self):
        """Calculate the next inverse temperature (beta).

        The importance weights based on two successive tempered likelihoods (i.e.
        two successive values of beta) and updates the marginal likelihood estimate.

        ESS is calculated for importance sampling. BDA 3rd ed. eq 10.4
        """
        self.iteration += 1

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

        self.beta = new_beta
        self.weights = np.exp(log_weights)
        # We normalize again to correct for small numerical errors that might build up
        self.weights /= self.weights.sum()
        self.log_marginal_likelihood += logsumexp(log_weights_un) - np.log(self.draws)

    def resample(self):
        """Resample particles based on importance weights."""
        self.resampling_indexes = systematic_resampling(self.weights, self.rng)

        self.tempered_posterior = self.tempered_posterior[self.resampling_indexes]
        self.prior_logp = self.prior_logp[self.resampling_indexes]
        self.likelihood_logp = self.likelihood_logp[self.resampling_indexes]

        self.tempered_posterior_logp = self.prior_logp + self.likelihood_logp * self.beta

    def tune(self):
        """Tuning logic performed before every mutation step."""
        pass

    @abc.abstractmethod
    def mutate(self):
        """Apply kernel-specific perturbation to the particles once per stage."""
        pass

    @abc.abstractmethod
    def sample_stats(self) -> SMCStats:
        """Stats to be saved at the end of each stage.

        These stats will be saved under `sample_stats` in the final InferenceData object.
        """
        pass

    def step(self) -> SMCStats:
        """Perform a single SMC stage: resample, tune, and mutate."""
        self.resample()
        self.tune()
        self.mutate()

        return self.sample_stats()

    @abc.abstractmethod
    def sample_settings(self) -> SMCSettings:
        """SMC_kernel settings to be saved once at the end of sampling.

        These stats will be saved under `sample_stats` in the final InferenceData object.
        """
        pass

    def _reset_state(self):
        """Reset the sampling state for a new run."""
        self.tempered_posterior = np.empty(0)
        self.prior_logp = None
        self.likelihood_logp = None
        self.tempered_posterior_logp = None
        self.log_marginal_likelihood = 0.0
        self.beta = 0.0
        self.iteration = 0
        self.resampling_indexes = None
        self.weights = np.ones(self.draws) / self.draws

    def initialize(self, start: dict | None, rng: np.random.Generator) -> None:
        """Initialize the kernel for sampling.

        Parameters
        ----------
        start : dict or None
            Starting point in parameter space, or None to sample from prior.
        rng : np.random.Generator
            Random number generator for this chain.
        """
        self.start = start
        self.rng = rng
        self._reset_state()
        self.set_rng(rng)
        self._initialize_kernel()
        self.setup_kernel()

    @staticmethod
    def _progressbar_config(n_chains=1):
        """Configure progress bar columns for SMC sampling.

        Returns columns to display and initial stats values.
        """
        columns = [
            TextColumn("{task.fields[beta]:.4f}", table_column=Column("Beta", ratio=1)),
        ]

        stats = {
            "beta": [0.0] * n_chains,
        }

        return columns, stats

    @staticmethod
    def _make_progressbar_update_functions():
        """Create functions to update progress bar statistics."""

        def update_stats(stats):
            return {
                "beta": stats.get("beta", 0.0),
            }

        return (update_stats,)


class IMH(SMC_KERNEL):
    """Independent Metropolis-Hastings SMC_kernel."""

    stats_dtypes_shapes: dict[str, tuple[type, list]] = {
        "log_marginal_likelihood": (float, []),
        "beta": (float, []),
        "accept_rate": (float, []),
    }

    def __init__(self, *args, correlation_threshold=0.01, **kwargs):
        """
        Create the Independent Metropolis-Hastings SMC kernel object.

        Parameters
        ----------
        correlation_threshold : float, default 0.01
            The lower the value, the higher the number of IMH steps computed automatically.
            Defaults to 0.01. It should be between 0 and 1.
        **kwargs : dict, optional
            Keyword arguments passed to the SMC_kernel.  Refer to SMC_kernel documentation for a
        list of all possible arguments.

        """
        super().__init__(*args, **kwargs)
        self.correlation_threshold = correlation_threshold

        self.proposal_dist = None
        self.acc_rate = None

    def tune(self):
        # Update MVNormal proposal based on the mean and covariance of the
        # tempered posterior.
        cov = np.cov(self.tempered_posterior, ddof=0, rowvar=0)
        cov = np.atleast_2d(cov)
        cov += 1e-6 * np.eye(cov.shape[0])
        if np.isnan(cov).any() or np.isinf(cov).any():
            raise ValueError('Sample covariances not valid! Likely "draws" is too small!')
        mean = np.average(self.tempered_posterior, axis=0)
        self.proposal_dist = multivariate_normal(mean, cov)

    def mutate(self):
        """Independent Metropolis-Hastings perturbation."""
        self.n_steps = 1
        old_corr = 2
        corr = Pearson(self.tempered_posterior)
        ac_ = []
        while True:
            log_R = np.log(self.rng.random(self.draws))
            # The proposal is independent from the current point.
            # We have to take that into account to compute the Metropolis-Hastings acceptance
            # We first compute the logp of proposing a transition to the current points.
            # This variable is updated at the end of the loop with the entries from the accepted
            # transitions, which is equivalent to recomputing it in every iteration of the loop.
            proposal = floatX(self.proposal_dist.rvs(size=self.draws, random_state=self.rng))
            proposal = proposal.reshape(len(proposal), -1)
            # To do that we compute the logp of moving to a new point
            forward_logp = self.proposal_dist.logpdf(proposal)
            # And to going back from that new point
            backward_logp = self.proposal_dist.logpdf(self.tempered_posterior)
            ll = np.array([self.likelihood_logp_func(prop) for prop in proposal])
            pl = np.array([self.prior_logp_func(prop) for prop in proposal])
            proposal_logp = pl + ll * self.beta
            accepted = log_R < (
                (proposal_logp + backward_logp) - (self.tempered_posterior_logp + forward_logp)
            )

            self.tempered_posterior[accepted] = proposal[accepted]
            self.tempered_posterior_logp[accepted] = proposal_logp[accepted]
            self.prior_logp[accepted] = pl[accepted]
            self.likelihood_logp[accepted] = ll[accepted]
            ac_.append(accepted)
            self.n_steps += 1

            pearson_r = corr.get(self.tempered_posterior)
            if np.mean((old_corr - pearson_r) > self.correlation_threshold) > 0.9:
                old_corr = pearson_r
            else:
                break

        self.acc_rate = np.mean(ac_)

    def sample_stats(self) -> SMCStats:
        return {
            "log_marginal_likelihood": self.log_marginal_likelihood if self.beta == 1 else np.nan,
            "beta": self.beta,
            "accept_rate": self.acc_rate,
        }

    def sample_settings(self) -> SMCSettings:
        return {
            "_n_draws": self.draws,
            "threshold": self.threshold,
            "_n_tune": self.n_steps,
            "correlation_threshold": self.correlation_threshold,
        }


class Pearson:
    def __init__(self, a):
        self.l = a.shape[0]
        self.am = a - np.sum(a, axis=0) / self.l
        self.aa = np.sum(self.am**2, axis=0) ** 0.5

    def get(self, b):
        bm = b - np.sum(b, axis=0) / self.l
        bb = np.sum(bm**2, axis=0) ** 0.5
        ab = np.sum(self.am * bm, axis=0)
        return np.abs(ab / (self.aa * bb))


class MH(SMC_KERNEL):
    """Metropolis-Hastings SMC_kernel."""

    stats_dtypes_shapes: dict[str, tuple[type, list]] = {
        "log_marginal_likelihood": (float, []),
        "beta": (float, []),
        "mean_accept_rate": (float, []),
        "mean_proposal_scale": (float, []),
    }

    def __init__(self, *args, correlation_threshold=0.01, **kwargs):
        """
        Create a Metropolis-Hastings SMC kernel.

        Parameters
        ----------
        correlation_threshold : float, default 0.01
            The lower the value, the higher the number of MH steps computed automatically.
            Defaults to 0.01. It should be between 0 and 1.
        **kwargs : dict, optional
            Keyword arguments passed to the SMC_kernel.  Refer to SMC_kernel documentation for a
            list of all possible arguments.

        """
        super().__init__(*args, **kwargs)
        self.correlation_threshold = correlation_threshold

        self.proposal_dist = None
        self.proposal_scales = None
        self.chain_acc_rate = None

    def setup_kernel(self):
        """Proposal dist is just a Multivariate Normal with unit identity covariance.

        Dimension specific scaling is provided by `self.proposal_scales` and set in `self.tune()`.
        """
        ndim = self.tempered_posterior.shape[1]
        self.proposal_scales = np.full(self.draws, min(1, 2.38**2 / ndim))

    def resample(self):
        super().resample()
        if self.iteration > 1:
            self.proposal_scales = self.proposal_scales[self.resampling_indexes]
            self.chain_acc_rate = self.chain_acc_rate[self.resampling_indexes]

    def tune(self):
        """Update proposal scales for each particle dimension and update number of MH steps."""
        if self.iteration > 1:
            # Rescale based on distance to 0.234 acceptance rate
            chain_scales = np.exp(np.log(self.proposal_scales) + (self.chain_acc_rate - 0.234))
            # Interpolate between individual and population scales
            self.proposal_scales = 0.5 * (chain_scales + chain_scales.mean())

        # Update MVNormal proposal based on the covariance of the tempered posterior.
        cov = np.cov(self.tempered_posterior, ddof=0, rowvar=0)
        cov = np.atleast_2d(cov)
        cov += 1e-6 * np.eye(cov.shape[0])
        if np.isnan(cov).any() or np.isinf(cov).any():
            raise ValueError('Sample covariances not valid! Likely "draws" is too small!')
        self.proposal_dist = MultivariateNormalProposal(cov)

    def mutate(self):
        """Metropolis-Hastings perturbation."""
        self.n_steps = 1
        old_corr = 2
        corr = Pearson(self.tempered_posterior)
        ac_ = []
        while True:
            log_R = np.log(self.rng.random(self.draws))
            proposal = floatX(
                self.tempered_posterior
                + self.proposal_dist(num_draws=self.draws, rng=self.rng)
                * self.proposal_scales[:, None]
            )
            ll = np.array([self.likelihood_logp_func(prop) for prop in proposal])
            pl = np.array([self.prior_logp_func(prop) for prop in proposal])

            proposal_logp = pl + ll * self.beta
            accepted = log_R < (proposal_logp - self.tempered_posterior_logp)

            self.tempered_posterior[accepted] = proposal[accepted]
            self.prior_logp[accepted] = pl[accepted]
            self.likelihood_logp[accepted] = ll[accepted]
            self.tempered_posterior_logp[accepted] = proposal_logp[accepted]
            ac_.append(accepted)
            self.n_steps += 1

            pearson_r = corr.get(self.tempered_posterior)
            if np.mean((old_corr - pearson_r) > self.correlation_threshold) > 0.9:
                old_corr = pearson_r
            else:
                break

        self.chain_acc_rate = np.mean(ac_, axis=0)

    def sample_stats(self) -> SMCStats:
        return {
            "log_marginal_likelihood": self.log_marginal_likelihood if self.beta == 1 else np.nan,
            "beta": self.beta,
            "mean_accept_rate": self.chain_acc_rate.mean(),
            "mean_proposal_scale": self.proposal_scales.mean(),
        }

    def sample_settings(self) -> SMCSettings:
        return {
            "_n_draws": self.draws,
            "threshold": self.threshold,
            "_n_tune": self.n_steps,
            "correlation_threshold": self.correlation_threshold,
        }


def systematic_resampling(weights, rng):
    """
    Systematic resampling.

    Parameters
    ----------
    weights :
        The weights should be probabilities and the total sum should be 1.

    Returns
    -------
    new_indices: array
        A vector of indices in the interval 0, ..., len(normalized_weights)
    """
    lnw = len(weights)
    arange = np.arange(lnw)
    uniform = (rng.random(1) + arange) / lnw

    idx = 0
    weight_accu = weights[0]
    new_indices = np.empty(lnw, dtype=int)
    for i in arange:
        while uniform[i] > weight_accu:
            idx += 1
            weight_accu += weights[idx]
        new_indices[i] = idx

    return new_indices


def _logp_forw(point, out_vars, in_vars, shared, compile_kwargs=None):
    """Compile PyTensor function of the model and the input and output variables.

    Parameters
    ----------
    out_vars : list
        Containing Distribution for the output variables
    in_vars : list
        Containing Distribution for the input variables
    shared : list
        Containing TensorVariable for depended shared data
    compile_kwargs: dict, optional
        Additional keyword arguments passed to pytensor.function
    """
    if compile_kwargs is None:
        compile_kwargs = {}

    # Replace integer inputs with rounded float inputs
    if any(var.dtype in discrete_types for var in in_vars):
        replace_int_input = {}
        new_in_vars = []
        for in_var in in_vars:
            if in_var.dtype in discrete_types:
                float_var = pt.TensorType("floatX", in_var.type.shape)(in_var.name)
                new_in_vars.append(float_var)
                replace_int_input[in_var] = pt.round(float_var).astype(in_var.dtype)
            else:
                new_in_vars.append(in_var)

        out_vars = clone_replace(out_vars, replace_int_input, rebuild_strict=False)
        in_vars = new_in_vars

    out_list, inarray0 = join_nonshared_inputs(
        point=point, outputs=out_vars, inputs=in_vars, shared_inputs=shared
    )
    f = compile([inarray0], out_list[0], **compile_kwargs)
    f.trust_input = True
    return f
