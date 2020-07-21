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

import numpy as np
import warnings
import logging
from typing import Union, List, Optional, Type

from .arraystep import ArrayStepShared, metrop_select, Competence
from .compound import CompoundStep
from .metropolis import Proposal, Metropolis, DEMetropolisZ, delta_logp
from ..model import Model
import pymc3 as pm

__all__ = ["MetropolisMLDA", "DEMetropolisZMLDA", "RecursiveDAProposal", "MLDA"]


class MetropolisMLDA(Metropolis):
    """
    Metropolis-Hastings sampling step tailored for use as base sampler in MLDA
    """

    name = "metropolis_mlda"

    def reset_tuning(self):
        """Does not reset sampler parameters. Allows continuation with
        the same settings when MetropolisMLDA steps are done in chunks
        under MLDA."""
        return


class DEMetropolisZMLDA(DEMetropolisZ):
    """
    DEMetropolisZ sampling step tailored for use as base sampler in MLDA
    """

    name = "DEMetropolisZ_mlda"

    def __init__(self, *args, **kwargs):
        """Initialise DEMetropolisZMLDA by setting a local variable
        and calling the parent class __init__()"""

        # flag used for signaling the end of tuning
        self.tuning_end_trigger = False

        super().__init__(*args, **kwargs)

    def reset_tuning(self):
        """Skips resetting of tuned sampler parameters
        and history to their initial values. Allows
        continuation with the same settings when
        DEMetropolisZMLDA steps are done in chunks
        under MLDA."""
        return

    def stop_tuning(self):
        """At the end of the tuning phase, this method
        removes the first x% of the history so future
        proposals are not informed by unconverged tuning
        iterations. Runs only once after the end of tuning,
        when the self.tuning_end_trigger flag is set to True.
        """
        if self.tuning_end_trigger:
            it = len(self._history)
            n_drop = int(self.tune_drop_fraction * it)
            self._history = self._history[n_drop:]
        return super().stop_tuning()


class MLDA(ArrayStepShared):
    """
    Multi-Level Delayed Acceptance (MLDA) sampling step that uses coarse
    approximations of a fine model to construct proposals in multiple levels.

    MLDA creates a hierarchy of MCMC chains. Chains sample from different
    posteriors that ideally should be approximations of the fine (top-level)
    posterior and require less computational effort to evaluate their likelihood.

    Each chain runs for a fixed number of iterations (subsampling_rate) and then
    the last sample generated is used as a proposal for the chain in the level
    above. The bottom-level chain is a MetropolisMLDA or DEMetropolisZMLDA sampler.
    The algorithm achieves higher acceptance rate and effective sample sizes
    than other samplers if the coarse models are sufficiently good approximations
    of the fine one.

    Parameters
    ----------
    coarse_models : list
        List of coarse (multi-level) models, where the first model
        is the coarsest one (level=0) and the last model is the
        second finest one (level=L-1 where L is the number of levels).
        Note this list excludes the model passed to the model
        argument above, which is the finest available.
    vars : list
        List of variables for sampler
    base_sampler : string
        Sampler used in the base (coarsest) chain. Can be 'Metropolis' or
        'DEMetropolisZ'. Defaults to 'DEMetropolisZ'.
    base_S : standard deviation of base proposal covariance matrix
        Some measure of variance to parameterize base proposal distribution
    base_proposal_dist : function
        Function that returns zero-mean deviates when parameterized with
        S (and n). Defaults to normal. This is the proposal used in the
        coarsest (base) chain, i.e. level=0.
    base_scaling : scalar or array
        Initial scale factor for base proposal. Defaults to 1 if base_sampler
        is 'Metropolis' and to 0.001 if base_sampler is 'DEMetropolisZ'.
    tune : bool
        Flag for tuning in the base proposal. If base_sampler is 'Metropolis' it
        should be True or False and defaults to True. Note that
        this is overidden by the tune parameter in sample(). For example when calling
        step=MLDA(tune=False, ...) and then sample(step=step, tune=200, ...),
        tuning will be activated for the first 200 steps. If base_sampler is
        'DEMetropolisZ', it should be True. For 'DEMetropolisZ', there is a separate
        argument base_tune_target which allows modifying the type of tuning.
    base_tune_target: string
        Defines the type of tuning that is performed when base_sampler is
        'DEMetropolisZ'. Allowable values are 'lambda, 'scaling' or None and
        it defaults to 'lambda'.
    base_tune_interval : int
        The frequency of tuning for the base proposal. Defaults to 100
        iterations.
    base_lamb : float
        Lambda parameter of the base level DE proposal mechanism. Only applicable when
        base_sampler is 'DEMetropolisZ'. Defaults to 2.38 / sqrt(2 * ndim)
    base_tune_drop_fraction: float
        Fraction of tuning steps that will be removed from the base level samplers
        history when the tuning ends. Only applicable when base_sampler is
        'DEMetropolisZ'. Defaults to 0.9 - keeping the last 10% of tuning steps
        for good mixing while removing 90% of potentially unconverged tuning positions.
    model : PyMC Model
        Optional model for sampling step. Defaults to None
        (taken from context). This model should be the finest of all
        multilevel models.
    mode :  string or `Mode` instance.
        Compilation mode passed to Theano functions
    subsampling_rates : integer or list of integers
        One interger for all levels or a list with one number for each level
        (excluding the finest level).
        This is the number of samples generated in level l-1 to propose a sample
        for level l for all l levels (excluding the finest level). The length of
        the list needs to be the same as the length of coarse_models.
    base_blocked : bool
        Flag to choose whether base sampler (level=0) is a
        Compound MetropolisMLDA step (base_blocked=False)
        or a blocked MetropolisMLDA step (base_blocked=True).
        Only applicable when base_sampler='Metropolis'.

    Examples
    ----------
    .. code:: ipython

        >>> import pymc3 as pm
        ... datum = 1
        ...
        ... with pm.Model() as coarse_model:
        ...     x = pm.Normal("x", mu=0, sigma=10)
        ...     y = pm.Normal("y", mu=x, sigma=1, observed=datum - 0.1)
        ...
        ... with pm.Model():
        ...     x = pm.Normal("x", mu=0, sigma=10)
        ...     y = pm.Normal("y", mu=x, sigma=1, observed=datum)
        ...     step_method = pm.MLDA(coarse_models=[coarse_model]
        ...                           subsampling_rates=5)
        ...     trace = pm.sample(draws=500, chains=2,
        ...                       tune=100, step=step_method,
        ...                       random_seed=123)
        ...
        ... pm.summary(trace)
            mean     sd	     hpd_3%	   hpd_97%
        x	0.982	1.026	 -0.994	   2.902

    References
    ----------
    .. [Dodwell2019] Dodwell, Tim & Ketelsen, Chris & Scheichl,
    Robert & Teckentrup, Aretha. (2019).
    Multilevel Markov Chain Monte Carlo.
    SIAM Review. 61. 509-545.
        `link <https://doi.org/10.1137/19M126966X>`__
    """

    name = "mlda"

    # All levels use block sampling,
    # except level 0 where the user can choose
    default_blocked = True
    generates_stats = True

    # These stats are extended within __init__
    stats_dtypes = [
        {
            "accept": np.float64,
            "accepted": np.bool,
            "tune": np.bool,
            "base_scaling": object,
        }
    ]

    def __init__(
        self,
        coarse_models: List[Model],
        vars: Optional[list] = None,
        base_sampler='DEMetropolisZ',
        base_S: Optional = None,
        base_proposal_dist: Optional[Type[Proposal]] = None,
        base_scaling: Union[float, int] = 1.0,
        tune: bool = True,
        base_tune_target='lambda',
        base_tune_interval: int = 100,
        base_lamb=None,
        base_tune_drop_fraction=0.9,
        model: Optional[Model] = None,
        mode: Optional = None,
        subsampling_rates: List[int] = 5,
        base_blocked: bool = False,
        **kwargs,
    ) -> None:

        warnings.warn(
            "The MLDA implementation in PyMC3 is very young. "
            "You should be extra critical about its results."
        )

        model = pm.modelcontext(model)

        # assign internal state
        self.coarse_models = coarse_models
        if not isinstance(coarse_models, list):
            raise ValueError("MLDA step method cannot use coarse_models if it is not a list")
        if len(self.coarse_models) == 0:
            raise ValueError(
                "MLDA step method was given an empty "
                "list of coarse models. Give at least "
                "one coarse model."
            )
        if isinstance(subsampling_rates, int):
            self.subsampling_rates = [subsampling_rates] * len(self.coarse_models)
        else:
            if len(subsampling_rates) != len(self.coarse_models):
                raise ValueError(
                    f"List of subsampling rates needs to have the same "
                    f"length as list of coarse models but the lengths "
                    f"were {len(subsampling_rates)}, {len(self.coarse_models)}"
                )
            self.subsampling_rates = subsampling_rates
        self.num_levels = len(self.coarse_models) + 1
        self.base_sampler = base_sampler
        self.base_S = base_S
        self.base_proposal_dist = base_proposal_dist

        if base_scaling is None:
            if self.base_sampler == 'Metropolis':
                self.base_scaling = 1.
            else:
                self.base_scaling = 0.001
        else:
            self.base_scaling = float(base_scaling)

        self.tune = tune
        if not self.tune and self.base_sampler == 'DEMetropolisZ':
            raise ValueError(f"The argument tune was set to False while using"
                             f" a 'DEMetropolisZ' base sampler. 'DEMetropolisZ' "
                             f" tune needs to be True.")

        self.base_tune_target = base_tune_target
        self.base_tune_interval = base_tune_interval
        self.base_lamb = base_lamb
        self.base_tune_drop_fraction = float(base_tune_drop_fraction)
        self.model = model
        self.next_model = self.coarse_models[-1]
        self.mode = mode
        self.base_blocked = base_blocked
        self.base_scaling_stats = None
        if self.base_sampler == 'DEMetropolisZ':
            self.base_lambda_stats = None

        # Process model variables
        if vars is None:
            vars = model.vars
        vars = pm.inputvars(vars)
        self.vars = vars
        self.var_names = [var.name for var in self.vars]

        self.accepted = 0

        # Construct theano function for current-level model likelihood
        # (for use in acceptance)
        shared = pm.make_shared_replacements(vars, model)
        self.delta_logp = delta_logp(model.logpt, vars, shared)

        # Construct theano function for next-level model likelihood
        # (for use in acceptance)
        next_model = pm.modelcontext(self.next_model)
        vars_next = [var for var in next_model.vars if var.name in self.var_names]
        vars_next = pm.inputvars(vars_next)
        shared_next = pm.make_shared_replacements(vars_next, next_model)
        self.delta_logp_next = delta_logp(next_model.logpt, vars_next, shared_next)

        super().__init__(vars, shared)

        # initialise complete step method hierarchy
        if self.num_levels == 2:
            with self.next_model:
                # make sure the correct variables are selected from next_model
                vars_next = [
                    var for var in self.next_model.vars if var.name in self.var_names
                ]
                if self.base_sampler == 'Metropolis':
                    # MetropolisMLDA sampler in base level (level=0), targeting self.next_model
                    self.next_step_method = pm.MetropolisMLDA(vars=vars_next,
                                                              proposal_dist=self.base_proposal_dist,
                                                              S=self.base_S,
                                                              scaling=self.base_scaling, tune=self.tune,
                                                              tune_interval=self.base_tune_interval,
                                                              model=None,
                                                              mode=self.mode,
                                                              blocked=self.base_blocked)
                else:
                    # DEMetropolisZMLDA sampler in base level (level=0), targeting self.next_model
                    self.next_step_method = pm.DEMetropolisZMLDA(vars=vars_next,
                                                                 S=self.base_S,
                                                                 proposal_dist=self.base_proposal_dist,
                                                                 lamb=self.base_lamb,
                                                                 scaling=self.base_scaling,
                                                                 tune=self.base_tune_target,
                                                                 tune_interval=self.base_tune_interval,
                                                                 tune_drop_fraction=self.base_tune_drop_fraction,
                                                                 model=None,
                                                                 mode=self.mode)

        else:
            # drop the last coarse model
            next_coarse_models = self.coarse_models[:-1]
            next_subsampling_rates = self.subsampling_rates[:-1]
            with self.next_model:
                # make sure the correct variables are selected from next_model
                vars_next = [var for var in self.next_model.vars if var.name in self.var_names]
                # MLDA sampler in some intermediate level, targeting self.next_model
                self.next_step_method = pm.MLDA(vars=vars_next, base_S=self.base_S,
                                                base_sampler=self.base_sampler,
                                                base_proposal_dist=self.base_proposal_dist,
                                                base_scaling=self.base_scaling,
                                                tune=self.tune,
                                                base_tune_target=self.base_tune_target,
                                                base_tune_interval=self.base_tune_interval,
                                                base_lamb=self.base_lamb,
                                                base_tune_drop_fraction=self.base_tune_drop_fraction,
                                                model=None, mode=self.mode,
                                                subsampling_rates=next_subsampling_rates,
                                                coarse_models=next_coarse_models,
                                                base_blocked=self.base_blocked,
                                                **kwargs)

        # instantiate the recursive DA proposal.
        # this is the main proposal used for
        # all levels (Recursive Delayed Acceptance)
        # (except for level 0 where the step method is MetropolisMLDA
        # or DEMetropolisZMLDA - not MLDA)
        self.proposal_dist = RecursiveDAProposal(
            self.next_step_method,
            self.next_model,
            self.tune,
            self.subsampling_rates[-1]
        )

        # add 'base_lambda' to stats if 'DEMetropolisZ' is used
        if self.base_sampler == 'DEMetropolisZ':
            self.stats_dtypes[0]['base_lambda'] = np.float64

    def astep(self, q0):
        """One MLDA step, given current sample q0"""
        # Check if the tuning flag has been changed and if yes,
        # change the proposal's tuning flag and reset self.accepted
        # This is triggered by _iter_sample while the highest-level MLDA step
        # method is running. It then propagates to all levels.
        if self.proposal_dist.tune != self.tune:
            self.proposal_dist.tune = self.tune
            # set tune in sub-methods of compound stepper explicitly because
            # it is not set within sample.py (only the CompoundStep's tune flag is)
            if isinstance(self.next_step_method, CompoundStep):
                for method in self.next_step_method.methods:
                    method.tune = self.tune
            self.accepted = 0

        # Convert current sample from numpy array ->
        # dict before feeding to proposal
        q0_dict = self.bij.rmap(q0)

        # Call the recursive DA proposal to get proposed sample
        # and convert dict -> numpy array
        q = self.bij.map(self.proposal_dist(q0_dict))

        # Evaluate MLDA acceptance log-ratio
        # If proposed sample from lower levels is the same as current one,
        # do not calculate likelihood, just set accept to 0.0
        if (q == q0).all():
            accept = np.float(0.0)
            skipped_logp = True
        else:
            accept = self.delta_logp(q, q0) + self.delta_logp_next(q0, q)
            skipped_logp = False

        # Accept/reject sample - next sample is stored in q_new
        q_new, accepted = metrop_select(accept, q, q0)
        if skipped_logp:
            accepted = False

        # Update acceptance counter
        self.accepted += accepted

        stats = {"tune": self.tune, "accept": np.exp(accept), "accepted": accepted}

        # Capture latest base chain scaling stats from next step method
        self.base_scaling_stats = {}
        if self.base_sampler == "DEMetropolisZ":
            self.base_lambda_stats = {}
        if isinstance(self.next_step_method, CompoundStep):
            # next method is Compound MetropolisMLDA
            scaling_list = []
            for method in self.next_step_method.methods:
                scaling_list.append(method.scaling)
            self.base_scaling_stats = {"base_scaling": np.array(scaling_list)}
        elif not isinstance(self.next_step_method, MLDA):
            # next method is any block sampler
            self.base_scaling_stats = {
                "base_scaling": np.array(self.next_step_method.scaling)
            }
            if self.base_sampler == "DEMetropolisZ":
                self.base_lambda_stats = {
                "base_lambda": self.next_step_method.lamb
            }
        else:
            # next method is MLDA - propagate dict from lower levels
            self.base_scaling_stats = self.next_step_method.base_scaling_stats
            if self.base_sampler == "DEMetropolisZ":
                self.base_lambda_stats = self.next_step_method.base_lambda_stats
        stats = {**stats, **self.base_scaling_stats}
        if self.base_sampler == "DEMetropolisZ":
            stats = {**stats, **self.base_lambda_stats}

        return q_new, [stats]

    @staticmethod
    def competence(var, has_grad):
        """Return MLDA competence for given var/has_grad. MLDA currently works
        only with continuous variables."""
        if var.dtype in pm.discrete_types:
            return Competence.INCOMPATIBLE
        return Competence.COMPATIBLE


# Available proposal distributions for MLDA


class RecursiveDAProposal(Proposal):
    """
    Recursive Delayed Acceptance proposal to be used with MLDA step sampler.
    Recursively calls an MLDA sampler if level > 0 and calls MetropolisMLDA or
    DEMetropolisZMLDA sampler if level = 0. The sampler generates
    subsampling_rate samples and the last one is used as a proposal.
    Results in a hierarchy of chains each of which is used to propose
    samples to the chain above.
    """

    def __init__(
        self,
        next_step_method: Union[MLDA, Metropolis, CompoundStep],
        next_model: Model,
        tune: bool,
        subsampling_rate: int,
    ) -> None:

        self.next_step_method = next_step_method
        self.next_model = next_model
        self.tune = tune
        self.subsampling_rate = subsampling_rate
        self.tuning_end_trigger = True
        self.trace = None

    def __call__(self, q0_dict: dict) -> dict:
        """Returns proposed sample given the current sample
        in dictionary form (q0_dict)."""

        # Logging is reduced to avoid extensive console output
        # during multiple recursive calls of subsample()
        _log = logging.getLogger("pymc3")
        _log.setLevel(logging.ERROR)

        with self.next_model:
            # Check if the tuning flag has been set to False
            # in which case tuning is stopped. The flag is set
            # to False (by MLDA's astep) when the burn-in
            # iterations of the highest-level MLDA sampler run out.
            # The change propagates to all levels.
            if self.tune:
                # Subsample in tuning mode
                self.trace = pm.subsample(draws=0, step=self.next_step_method,
                                          start=q0_dict, trace=self.trace,
                                          tune=self.subsampling_rate)
            else:
                # Subsample in normal mode without tuning
                # If DEMetropolisZMLDA is the base sampler a flag is raised to
                # make sure that history is edited after tuning ends
                if self.tuning_end_trigger and isinstance(self.next_step_method, DEMetropolisZMLDA):
                    self.next_step_method.tuning_end_trigger = True
                self.trace = pm.subsample(draws=self.subsampling_rate,
                                          step=self.next_step_method,
                                          start=q0_dict, trace=self.trace)
                self.tuning_end_trigger = False
                # If DEMetropolisZMLDA is the base sampler the flag is set to False
                # to avoid further deletion of samples history
                if isinstance(self.next_step_method, DEMetropolisZMLDA):
                    self.next_step_method.tuning_end_trigger = False

        # set logging back to normal
        _log.setLevel(logging.NOTSET)

        return self.trace.point(-1)
