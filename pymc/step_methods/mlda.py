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

import logging
import warnings

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import arviz as az
import numpy as np

from aesara.tensor.sharedvar import TensorSharedVariable

import pymc as pm

from pymc.blocking import DictToArrayBijection, RaveledVars
from pymc.model import Model, Point
from pymc.step_methods.arraystep import ArrayStepShared, Competence, metrop_select
from pymc.step_methods.compound import CompoundStep
from pymc.step_methods.metropolis import DEMetropolisZ, Metropolis, Proposal, delta_logp

__all__ = [
    "MetropolisMLDA",
    "DEMetropolisZMLDA",
    "RecursiveDAProposal",
    "MLDA",
    "extract_Q_estimate",
]


class MetropolisMLDA(Metropolis):
    """
    Metropolis-Hastings sampling step tailored for use as base sampler in MLDA.
    """

    name = "metropolis_mlda"

    def __init__(self, *args, **kwargs):
        """
        Initialise MetropolisMLDA. This is a mix of the parent's class' initialisation
        and some extra code specific for MLDA.
        """
        model = pm.modelcontext(kwargs.get("model", None))
        initial_values = model.compute_initial_point()

        # flag to that variance reduction is activated - forces MetropolisMLDA
        # to store quantities of interest in a register if True
        self.mlda_variance_reduction = kwargs.pop("mlda_variance_reduction", False)
        if self.mlda_variance_reduction:
            # Subsampling rate of MLDA sampler one level up
            self.mlda_subsampling_rate_above = kwargs.pop("mlda_subsampling_rate_above")
            self.sub_counter = 0
            self.Q_last = np.nan
            self.Q_reg = [np.nan] * self.mlda_subsampling_rate_above

        # call parent class __init__
        super().__init__(*args, **kwargs)

        # modify the delta function and point to model if VR is used
        if self.mlda_variance_reduction:
            self.model = model
            self.delta_logp_factory = self.delta_logp
            self.delta_logp = lambda q, q0: -self.delta_logp_factory(q0, q)

    def reset_tuning(self):
        """
        Does not reset sampler parameters. Allows continuation with
        the same settings when MetropolisMLDA steps are done in chunks
        under MLDA.
        """
        return

    def astep(self, q0):

        q_new, stats = super().astep(q0)

        # Add variance reduction functionality.
        if self.mlda_variance_reduction:
            if stats[0]["accepted"]:
                self.Q_last = self.model.Q.get_value()
            if self.sub_counter == self.mlda_subsampling_rate_above:
                self.sub_counter = 0
            self.Q_reg[self.sub_counter] = self.Q_last
            self.sub_counter += 1

        return q_new, stats


class DEMetropolisZMLDA(DEMetropolisZ):
    """
    DEMetropolisZ sampling step tailored for use as base sampler in MLDA
    """

    name = "DEMetropolisZ_mlda"

    def __init__(self, *args, **kwargs):
        """
        Initialise DEMetropolisZMLDA, uses parent class __init__
        and extra code specific for use within MLDA.
        """

        # flag used for signaling the end of tuning
        self.tuning_end_trigger = False

        model = pm.modelcontext(kwargs.get("model", None))
        initial_values = model.compute_initial_point()

        # flag to that variance reduction is activated - forces DEMetropolisZMLDA
        # to store quantities of interest in a register if True
        self.mlda_variance_reduction = kwargs.pop("mlda_variance_reduction", False)
        if self.mlda_variance_reduction:
            # Subsampling rate of MLDA sampler one level up
            self.mlda_subsampling_rate_above = kwargs.pop("mlda_subsampling_rate_above")
            self.sub_counter = 0
            self.Q_last = np.nan
            self.Q_reg = [np.nan] * self.mlda_subsampling_rate_above

        # call parent class __init__
        super().__init__(*args, **kwargs)

        # modify the delta function and point to model if VR is used
        if self.mlda_variance_reduction:
            self.model = model
            self.delta_logp_factory = self.delta_logp
            self.delta_logp = lambda q, q0: -self.delta_logp_factory(q0, q)

    def reset_tuning(self):
        """Skips resetting of tuned sampler parameters
        and history to their initial values. Allows
        continuation with the same settings when
        DEMetropolisZMLDA steps are done in chunks
        under MLDA."""
        return

    def astep(self, q0):

        q_new, stats = super().astep(q0)

        # Add variance reduction functionality.
        if self.mlda_variance_reduction:
            if stats[0]["accepted"]:
                self.Q_last = self.model.Q.get_value()
            if self.sub_counter == self.mlda_subsampling_rate_above:
                self.sub_counter = 0
            self.Q_reg[self.sub_counter] = self.Q_last
            self.sub_counter += 1

        return q_new, stats

    def stop_tuning(self):
        """At the end of the tuning phase, this method
        removes the first x% of the history so future
        proposals are not informed by unconverged tuning
        iterations. Runs only once after the end of tuning,
        when the self.tuning_end_trigger flag is set to True.
        """
        if self.tuning_end_trigger:
            self.tuning_end_trigger = False
            return super().stop_tuning()
        else:
            return


class MLDA(ArrayStepShared):
    """
    Multi-Level Delayed Acceptance (MLDA) sampling step that uses coarse
    approximations of a fine model to construct proposals in multiple levels.

    MLDA creates a hierarchy of MCMC chains. Chains sample from different
    posteriors that ideally should be approximations of the fine (top-level)
    posterior and require less computational effort to evaluate their likelihood.

    Each chain runs for a fixed number of iterations (up to subsampling_rate) and
    then the last sample generated is used as a proposal for the chain in the level
    above (excluding when variance reduction is used, where a random sample from
    the generated sequence is used). The bottom-level chain is a MetropolisMLDA
    or DEMetropolisZMLDA sampler.

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
        List of value variables for sampler
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
    base_tune_drop_fraction : float
        Fraction of tuning steps that will be removed from the base level samplers
        history when the tuning ends. Only applicable when base_sampler is
        'DEMetropolisZ'. Defaults to 0.9 - keeping the last 10% of tuning steps
        for good mixing while removing 90% of potentially unconverged tuning positions.
    model : PyMC Model
        Optional model for sampling step. Defaults to None
        (taken from context). This model should be the finest of all
        multilevel models.
    mode :  string or `Mode` instance.
        Compilation mode passed to Aesara functions
    subsampling_rates : integer or list of integers
        One interger for all levels or a list with one number for each level
        (excluding the finest level).
        This is the number of samples generated in level l-1 to
        propose a sample for level l - applies to all levels excluding the
        finest level). The length of the list needs to be the same as the
        length of coarse_models.
    base_blocked : bool
        Flag to choose whether base sampler (level=0) is a
        Compound MetropolisMLDA step (base_blocked=False)
        or a blocked MetropolisMLDA step (base_blocked=True).
        Only applicable when base_sampler='Metropolis'.
    variance_reduction: bool
        Calculate and store quantities of interest and quantity of interest
        differences between levels to enable computing a variance-reduced
        sum of the quantity of interest after sampling. In order to use
        variance reduction, the user needs to do the following when defining
        the PyMC model (also demonstrated in the example notebook):
            - Include a `pm.Data()` variable with the name `Q` in the
            model description of all levels.
            - Use an Aesara Op to calculate the forward model (or the
            combination of a forward model and a likelihood). This Op
            should have a `perform()` method which (in addition to all
            the other calculations), calculates the quantity of interest
            and stores it to the variable `Q` of the PyMC model,
            using the `set_value()` function.
        When variance_reduction=True, all subchains run for a fixed number
        of iterations (equal to subsampling_rates) and a random sample is
        selected from the generated sequence (instead of the last sample
        which is selected when variance_reduction=False).
    store_Q_fine: bool
        Store the values of the quantity of interest from the fine chain.
    adaptive_error_model : bool
        When True, MLDA will use the adaptive error model method
        proposed in [Cui2012]. The method requires the likelihood of
        the model to be adaptive and a forward model to be defined and
        fed to the sampler. Thus, it only works when the user does
        the following (also demonstrated in the example notebook):
            - Include in the model definition at all levels,
            the extra variable model_output, which will capture
            the forward model outputs. Also include in the model
            definition at all levels except the finest one, the
            extra variables mu_B and Sigma_B, which will capture
            the bias between different levels. All these variables
            should be instantiated using the pm.Data method.
            - Use an Aesara Op to define the forward model (and
            optionally the likelihood) for all levels. The Op needs
            to store the result of each forward model calculation
            to the variable model_output of the PyMC model,
            using the `set_value()` function.
            - Define a Multivariate Normal likelihood (either using
            the standard PyMC API or within an Op) which has mean
            equal to the forward model output plus mu_B and covariance
            equal to the model error plus Sigma_B.
        Given the above, MLDA will capture and iteratively update the
        bias terms internally for all level pairs and will correct
        each level so that all levels' forward models aim to estimate
        the finest level's forward model.

    Examples
    ----------
    .. code:: ipython
        >>> import pymc as pm
        ... datum = 1
        ...
        ... with pm.Model() as coarse_model:
        ...     x = pm.Normal("x", mu=0, sigma=10)
        ...     y = pm.Normal("y", mu=x, sigma=1, observed=datum - 0.1)
        ...
        ... with pm.Model():
        ...     x = pm.Normal("x", mu=0, sigma=10)
        ...     y = pm.Normal("y", mu=x, sigma=1, observed=datum)
        ...     step_method = pm.MLDA(coarse_models=[coarse_model],
        ...                           subsampling_rates=5)
        ...     idata = pm.sample(500, chains=2,
        ...                       tune=100, step=step_method,
        ...                       random_seed=123)
        ...
        ... az.summary(idata, kind="stats")
           mean     sd  hdi_3%  hdi_97%
        x  0.99  0.987  -0.734    2.992

    References
    ----------
    .. [Dodwell2019] Dodwell, Tim & Ketelsen, Chris & Scheichl,
    Robert & Teckentrup, Aretha. (2019).
    Multilevel Markov Chain Monte Carlo.
    SIAM Review. 61. 509-545.
        `link <https://doi.org/10.1137/19M126966X>`__
    .. [Cui2012] Cui, Tiangang & Fox, Colin & O'Sullivan, Michael.
    (2012). Adaptive Error Modelling in MCMC Sampling for Large
    Scale Inverse Problems.
    """

    name = "mlda"

    # All levels use block sampling,
    # except level 0 where the user can choose
    default_blocked = True
    generates_stats = True

    def __init__(
        self,
        coarse_models: List[Model],
        vars: Optional[list] = None,
        base_sampler="DEMetropolisZ",
        base_S: Optional = None,
        base_proposal_dist: Optional[Type[Proposal]] = None,
        base_scaling: Optional = None,
        tune: bool = True,
        base_tune_target: str = "lambda",
        base_tune_interval: int = 100,
        base_lamb: Optional = None,
        base_tune_drop_fraction: float = 0.9,
        model: Optional[Model] = None,
        mode: Optional = None,
        subsampling_rates: List[int] = 5,
        base_blocked: bool = False,
        variance_reduction: bool = False,
        store_Q_fine: bool = False,
        adaptive_error_model: bool = False,
        **kwargs,
    ) -> None:

        # this variable is used to identify MLDA objects which are
        # not in the finest level (i.e. child MLDA objects)
        self.is_child = kwargs.get("is_child", False)

        if not isinstance(coarse_models, list):
            raise ValueError("MLDA step method cannot use coarse_models if it is not a list")
        if len(coarse_models) == 0:
            raise ValueError(
                "MLDA step method was given an empty "
                "list of coarse models. Give at least "
                "one coarse model."
            )

        # assign internal state
        model = pm.modelcontext(model)
        initial_values = model.compute_initial_point()
        self.model = model
        self.coarse_models = coarse_models
        self.model_below = self.coarse_models[-1]
        self.num_levels = len(self.coarse_models) + 1

        # set up variance reduction.
        self.variance_reduction = variance_reduction
        self.store_Q_fine = store_Q_fine

        # check that certain requirements hold
        # for the variance reduction feature to work
        if self.variance_reduction or self.store_Q_fine:
            if not hasattr(self.model, "Q"):
                raise AttributeError(
                    "Model given to MLDA does not contain"
                    "variable 'Q'. You need to include"
                    "the variable in the model definition"
                    "for variance reduction to work or"
                    "for storing the fine Q."
                    "Use pm.Data() to define it."
                )
            if not isinstance(self.model.Q, TensorSharedVariable):
                raise TypeError(
                    "The variable 'Q' in the model definition is not of type "
                    "'TensorSharedVariable'. Use pm.Data() to define the"
                    "variable."
                )

        if self.is_child and self.variance_reduction:
            # this is the subsampling rate applied to the current level
            # it is stored in the level above and transferred here
            self.subsampling_rate_above = kwargs.pop("subsampling_rate_above", None)

        # set up adaptive error model
        self.adaptive_error_model = adaptive_error_model

        # check that certain requirements hold
        # for the adaptive error model feature to work
        if self.adaptive_error_model:
            if not hasattr(self.model_below, "mu_B"):
                raise AttributeError(
                    "Model below in hierarchy does not contain"
                    "variable 'mu_B'. You need to include"
                    "the variable in the model definition"
                    "for adaptive error model to work."
                    "Use pm.Data() to define it."
                )
            if not hasattr(self.model_below, "Sigma_B"):
                raise AttributeError(
                    "Model below in hierarchy does not contain"
                    "variable 'Sigma_B'. You need to include"
                    "the variable in the model definition"
                    "for adaptive error model to work."
                    "Use pm.Data() to define it."
                )
            if not (
                isinstance(self.model_below.mu_B, TensorSharedVariable)
                and isinstance(self.model_below.Sigma_B, TensorSharedVariable)
            ):
                raise TypeError(
                    "At least one of the variables 'mu_B' and 'Sigma_B' "
                    "in the definition of the below model is not of type "
                    "'TensorSharedVariable'. Use pm.Data() to define those "
                    "variables."
                )

            # this object is used to recursively update the mean and
            # variance of the bias correction given new differences
            # between levels
            self.bias = RecursiveSampleMoments(
                self.model_below.mu_B.get_value(), self.model_below.Sigma_B.get_value()
            )

            # this list holds the bias objects from all levels
            # it is gradually constructed when MLDA objects are
            # created and then shared between all levels
            self.bias_all = kwargs.pop("bias_all", None)
            if self.bias_all is None:
                self.bias_all = [self.bias]
            else:
                self.bias_all.append(self.bias)

            # variables used for adaptive error model
            self.last_synced_output_diff = None
            self.adaptation_started = False

        # set up subsampling rates.
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

        self.subsampling_rate = self.subsampling_rates[-1]
        self.subchain_selection = None

        # set up base sampling
        self.base_sampler = base_sampler

        # VR is not compatible with compound base samplers so an automatic conversion
        # to a block sampler happens here if
        if self.variance_reduction and self.base_sampler == "Metropolis" and not base_blocked:
            warnings.warn(
                "Variance reduction is not compatible with non-blocked (compound) samplers."
                "Automatically switching to a blocked Metropolis sampler."
            )
            self.base_blocked = True
        else:
            self.base_blocked = base_blocked

        self.base_S = base_S
        self.base_proposal_dist = base_proposal_dist

        if base_scaling is None:
            if self.base_sampler == "Metropolis":
                self.base_scaling = 1.0
            else:
                self.base_scaling = 0.001
        else:
            self.base_scaling = float(base_scaling)

        self.tune = tune
        if not self.tune and self.base_sampler == "DEMetropolisZ":
            raise ValueError(
                f"The argument tune was set to False while using"
                f" a 'DEMetropolisZ' base sampler. 'DEMetropolisZ' "
                f" tune needs to be True."
            )

        self.base_tune_target = base_tune_target
        self.base_tune_interval = base_tune_interval
        self.base_lamb = base_lamb
        self.base_tune_drop_fraction = float(base_tune_drop_fraction)
        self.base_tuning_stats = None

        self.mode = mode

        # Process model variables
        if vars is None:
            vars = model.value_vars
        else:
            vars = [model.rvs_to_values.get(var, var) for var in vars]
        vars = pm.inputvars(vars)
        self.vars = vars
        self.var_names = [var.name for var in self.vars]

        self.accepted = 0

        # Construct Aesara function for current-level model likelihood
        # (for use in acceptance)
        shared = pm.make_shared_replacements(initial_values, vars, model)
        self.delta_logp = delta_logp(initial_values, model.logpt(), vars, shared)

        # Construct Aesara function for below-level model likelihood
        # (for use in acceptance)
        model_below = pm.modelcontext(self.model_below)
        vars_below = [var for var in model_below.value_vars if var.name in self.var_names]
        vars_below = pm.inputvars(vars_below)
        shared_below = pm.make_shared_replacements(initial_values, vars_below, model_below)
        self.delta_logp_below = delta_logp(
            initial_values, model_below.logpt(), vars_below, shared_below
        )

        super().__init__(vars, shared)

        # initialise complete step method hierarchy
        if self.num_levels == 2:
            with self.model_below:
                # make sure the correct variables are selected from model_below
                vars_below = [
                    var for var in self.model_below.value_vars if var.name in self.var_names
                ]

                # create kwargs
                if self.variance_reduction:
                    base_kwargs = {
                        "mlda_subsampling_rate_above": self.subsampling_rate,
                        "mlda_variance_reduction": True,
                    }
                else:
                    base_kwargs = {}

                if self.base_sampler == "Metropolis":
                    # MetropolisMLDA sampler in base level (level=0), targeting self.model_below
                    self.step_method_below = pm.MetropolisMLDA(
                        vars=vars_below,
                        proposal_dist=self.base_proposal_dist,
                        S=self.base_S,
                        scaling=self.base_scaling,
                        tune=self.tune,
                        tune_interval=self.base_tune_interval,
                        model=None,
                        mode=self.mode,
                        blocked=self.base_blocked,
                        **base_kwargs,
                    )
                else:
                    # DEMetropolisZMLDA sampler in base level (level=0), targeting self.model_below
                    self.step_method_below = pm.DEMetropolisZMLDA(
                        vars=vars_below,
                        S=self.base_S,
                        proposal_dist=self.base_proposal_dist,
                        lamb=self.base_lamb,
                        scaling=self.base_scaling,
                        tune=self.base_tune_target,
                        tune_interval=self.base_tune_interval,
                        tune_drop_fraction=self.base_tune_drop_fraction,
                        model=None,
                        mode=self.mode,
                        **base_kwargs,
                    )
        else:
            # drop the last coarse model
            coarse_models_below = self.coarse_models[:-1]
            subsampling_rates_below = self.subsampling_rates[:-1]

            with self.model_below:
                # make sure the correct variables are selected from model_below
                vars_below = [
                    var for var in self.model_below.value_vars if var.name in self.var_names
                ]

                # create kwargs
                if self.variance_reduction:
                    mlda_kwargs = {
                        "is_child": True,
                        "subsampling_rate_above": self.subsampling_rate,
                    }
                else:
                    mlda_kwargs = {"is_child": True}
                if self.adaptive_error_model:
                    mlda_kwargs = {**mlda_kwargs, **{"bias_all": self.bias_all}}

                # MLDA sampler in some intermediate level, targeting self.model_below
                self.step_method_below = pm.MLDA(
                    vars=vars_below,
                    base_S=self.base_S,
                    base_sampler=self.base_sampler,
                    base_proposal_dist=self.base_proposal_dist,
                    base_scaling=self.base_scaling,
                    tune=self.tune,
                    base_tune_target=self.base_tune_target,
                    base_tune_interval=self.base_tune_interval,
                    base_lamb=self.base_lamb,
                    base_tune_drop_fraction=self.base_tune_drop_fraction,
                    model=None,
                    mode=self.mode,
                    subsampling_rates=subsampling_rates_below,
                    coarse_models=coarse_models_below,
                    base_blocked=self.base_blocked,
                    variance_reduction=self.variance_reduction,
                    store_Q_fine=False,
                    adaptive_error_model=self.adaptive_error_model,
                    **mlda_kwargs,
                )

        # instantiate the recursive DA proposal.
        # this is the main proposal used for
        # all levels (Recursive Delayed Acceptance)
        # (except for level 0 where the step method is MetropolisMLDA
        # or DEMetropolisZMLDA - not MLDA)
        self.proposal_dist = RecursiveDAProposal(
            self.step_method_below, self.model_below, self.tune, self.subsampling_rate
        )

        # set up data types of stats.
        if isinstance(self.step_method_below, MLDA):
            # get the stat types from the level below if that level is MLDA
            self.stats_dtypes = self.step_method_below.stats_dtypes

        else:
            # otherwise, set it up from scratch.
            self.stats_dtypes = [{"accept": np.float64, "accepted": bool, "tune": bool}]

            if isinstance(self.step_method_below, MetropolisMLDA):
                self.stats_dtypes.append({"base_scaling": np.float64})
            elif isinstance(self.step_method_below, DEMetropolisZMLDA):
                self.stats_dtypes.append({"base_scaling": np.float64, "base_lambda": np.float64})
            elif isinstance(self.step_method_below, CompoundStep):
                for method in self.step_method_below.methods:
                    if isinstance(method, MetropolisMLDA):
                        self.stats_dtypes.append({"base_scaling": np.float64})
                    elif isinstance(method, DEMetropolisZMLDA):
                        self.stats_dtypes.append(
                            {"base_scaling": np.float64, "base_lambda": np.float64}
                        )

        # initialise necessary variables for doing variance reduction
        if self.variance_reduction:
            self.sub_counter = 0
            self.Q_diff = []
            if self.is_child:
                self.Q_reg = [np.nan] * self.subsampling_rate_above
            if self.num_levels == 2:
                self.Q_base_full = []
            if not self.is_child:
                for level in range(self.num_levels - 1, 0, -1):
                    self.stats_dtypes[0][f"Q_{level}_{level - 1}"] = object
                self.stats_dtypes[0]["Q_0"] = object

        # initialise necessary variables for doing variance reduction or storing fine Q
        if self.variance_reduction or self.store_Q_fine:
            self.Q_last = np.nan
            self.Q_diff_last = np.nan
        if self.store_Q_fine and not self.is_child:
            self.stats_dtypes[0][f"Q_{self.num_levels - 1}"] = object

    def astep(self, q0: RaveledVars) -> Tuple[RaveledVars, List[Dict[str, Any]]]:
        """One MLDA step, given current sample q0"""
        # Check if the tuning flag has been changed and if yes,
        # change the proposal's tuning flag and reset self.accepted
        # This is triggered by _iter_sample while the highest-level MLDA step
        # method is running. It then propagates to all levels.
        if self.proposal_dist.tune != self.tune:
            self.proposal_dist.tune = self.tune
            # set tune in sub-methods of compound stepper explicitly because
            # it is not set within sample.py (only the CompoundStep's tune flag is)
            if isinstance(self.step_method_below, CompoundStep):
                for method in self.step_method_below.methods:
                    method.tune = self.tune
            self.accepted = 0

        # Set subchain_selection (which sample from the coarse chain
        # is passed as a proposal to the fine chain). If variance
        # reduction is used, a random sample is selected as proposal.
        # If variance reduction is not used, the last sample is
        # selected as proposal.
        if self.variance_reduction:
            self.subchain_selection = np.random.randint(0, self.subsampling_rate)
        else:
            self.subchain_selection = self.subsampling_rate - 1
        self.proposal_dist.subchain_selection = self.subchain_selection

        # Call the recursive DA proposal to get proposed sample
        # and convert dict -> numpy array
        q = self.proposal_dist(q0)

        # Evaluate MLDA acceptance log-ratio
        # If proposed sample from lower levels is the same as current one,
        # do not calculate likelihood, just set accept to 0.0
        if (q.data == q0.data).all():
            accept = np.float64(0.0)
            skipped_logp = True
        else:
            # NB! The order and sign of the first term are swapped compared
            # to the convention to make sure the proposal is evaluated last.
            accept = -self.delta_logp(q0.data, q.data) + self.delta_logp_below(q0.data, q.data)
            skipped_logp = False

        # Accept/reject sample - next sample is stored in q_new
        q_new, accepted = metrop_select(accept, q, q0)
        if skipped_logp:
            accepted = False

        # if sample is accepted, update self.Q_last with the sample's Q value
        # runs only for VR or when store_Q_fine is True
        if self.variance_reduction or self.store_Q_fine:
            if accepted and not skipped_logp:
                self.Q_last = self.model.Q.get_value()

        # Variance reduction
        if self.variance_reduction:
            self.update_vr_variables(accepted, skipped_logp)

        # Adaptive error model - runs only during tuning.
        if self.tune and self.adaptive_error_model:
            self.update_error_estimate(accepted, skipped_logp)

        # Update acceptance counter
        self.accepted += accepted

        stats = {"tune": self.tune, "accept": np.exp(accept), "accepted": accepted}

        # Save the VR statistics to the stats dictionary (only happens in the
        # top MLDA level)
        if (self.variance_reduction or self.store_Q_fine) and not self.is_child:
            q_stats = {}
            if self.variance_reduction:
                m = self
                for level in range(self.num_levels - 1, 0, -1):
                    # save the Q differences for this level and iteration
                    q_stats[f"Q_{level}_{level - 1}"] = np.array(m.Q_diff)
                    # this makes sure Q_diff is reset for
                    # the next iteration
                    m.Q_diff = []
                    if level == 1:
                        break
                    m = m.step_method_below
                q_stats["Q_0"] = np.array(m.Q_base_full)
                m.Q_base_full = []
            if self.store_Q_fine:
                q_stats["Q_" + str(self.num_levels - 1)] = np.array(self.Q_last)
            stats = {**stats, **q_stats}

        # Capture the base tuning stats from the level below.
        self.base_tuning_stats = []

        if isinstance(self.step_method_below, MLDA):
            self.base_tuning_stats = self.step_method_below.base_tuning_stats
        elif isinstance(self.step_method_below, MetropolisMLDA):
            self.base_tuning_stats.append({"base_scaling": self.step_method_below.scaling})
        elif isinstance(self.step_method_below, DEMetropolisZMLDA):
            self.base_tuning_stats.append(
                {
                    "base_scaling": self.step_method_below.scaling,
                    "base_lambda": self.step_method_below.lamb,
                }
            )
        elif isinstance(self.step_method_below, CompoundStep):
            # Below method is CompoundStep
            for method in self.step_method_below.methods:
                if isinstance(method, MetropolisMLDA):
                    self.base_tuning_stats.append({"base_scaling": method.scaling})
                elif isinstance(method, DEMetropolisZMLDA):
                    self.base_tuning_stats.append(
                        {"base_scaling": method.scaling, "base_lambda": method.lamb}
                    )

        return q_new, [stats] + self.base_tuning_stats

    def update_vr_variables(self, accepted, skipped_logp):
        """Updates all the variables necessary for VR to work.

        Each level has a Q_last and Q_diff_last register which store
        the Q of the last accepted MCMC sample and the difference
        between the Q of the last accepted sample in this level and
        the Q of the last sample in the level below.

        These registers are updated here so that they can be exported later."""

        # if this MLDA is not at the finest level, store Q_last in a
        # register Q_reg and increase sub_counter (until you reach
        # the subsampling rate, at which point you make it zero).
        # Q_reg will later be used by the level above to calculate differences
        if self.is_child:
            if self.sub_counter == self.subsampling_rate_above:
                self.sub_counter = 0
            self.Q_reg[self.sub_counter] = self.Q_last
            self.sub_counter += 1

        # if MLDA is in the level above the base level, extract the
        # latest set of Q values from Q_reg in the base level
        # and add them to Q_base_full (which stores all the history of
        # Q values from the base level)
        if self.num_levels == 2:
            self.Q_base_full.extend(self.step_method_below.Q_reg)

        # if the sample is accepted, update Q_diff_last with the latest
        # difference between the last Q of this level and the Q of the
        # proposed (selected) sample from the level below.
        # If sample is not accepted, just keep the latest accepted Q_diff
        if accepted and not skipped_logp:
            self.Q_diff_last = self.Q_last - self.step_method_below.Q_reg[self.subchain_selection]
        # Add the last accepted Q_diff to the list
        self.Q_diff.append(self.Q_diff_last)

    def update_error_estimate(self, accepted, skipped_logp):
        """Updates the adaptive error model estimate with
        the latest accepted forward model output difference. Also
        updates the model variables mu_B and Sigma_B.

        The current level estimates and stores the error
        model between the current level and the level below."""

        # only save errors when a sample is accepted (excluding skipped_logp)
        if accepted and not skipped_logp:
            # this is the error (i.e. forward model output difference)
            # between the current level's model and the model in the level below
            self.last_synced_output_diff = (
                self.model.model_output.get_value() - self.model_below.model_output.get_value()
            )
            self.adaptation_started = True

        if self.adaptation_started:
            # update the internal recursive bias estimator with the last saved error
            self.bias.update(self.last_synced_output_diff)
            # Update the model variables in the level below the current one.
            # Each level has its own bias correction (i.e. bias object) that
            # estimates the error between that level and the one below.
            # The model variables mu_B and Signa_B of a level are the
            # sum of the bias corrections of all levels below and including
            # that level. This sum is updated here.
            with self.model_below:
                pm.set_data(
                    {
                        "mu_B": sum(
                            bias.get_mu()
                            for bias in self.bias_all[: len(self.bias_all) - self.num_levels + 2]
                        )
                    }
                )
                pm.set_data(
                    {
                        "Sigma_B": sum(
                            bias.get_sigma()
                            for bias in self.bias_all[: len(self.bias_all) - self.num_levels + 2]
                        )
                    }
                )

    @staticmethod
    def competence(var, has_grad):
        """Return MLDA competence for given var/has_grad. MLDA currently works
        only with continuous variables."""
        if var.dtype in pm.discrete_types:
            return Competence.INCOMPATIBLE
        return Competence.COMPATIBLE


class RecursiveSampleMoments:
    """
    Iteratively constructs a sample mean
    and covariance, given input samples.

    Used to capture an estimate of the mean
    and covariance of the bias of an MLDA
    coarse model.
    """

    def __init__(self, mu_0, sigma_0, t=1):
        self.mu = mu_0
        self.sigma = sigma_0
        self.t = t

    def __call__(self):
        return self.mu, self.sigma

    def get_mu(self):
        """Returns the current mu value"""
        return self.mu

    def get_sigma(self):
        """Returns the current covariance value"""
        return self.sigma

    def update(self, x):
        """Updates the mean and covariance given a
        new sample x"""
        mu_previous = self.mu.copy()

        self.mu = (1 / (self.t + 1)) * (self.t * mu_previous + x)

        self.sigma = (self.t - 1) / self.t * self.sigma + 1 / self.t * (
            self.t * np.outer(mu_previous, mu_previous)
            - (self.t + 1) * np.outer(self.mu, self.mu)
            + np.outer(x, x)
        )

        self.t += 1


def extract_Q_estimate(trace, levels):
    """
    Returns expectation and standard error of quantity of interest,
    given a trace and the number of levels in the multilevel model.
    It makes use of the collapsing sum formula. Only applicable when
    MLDA with variance reduction has been used for sampling.
    """

    Q_0_raw = trace.get_sampler_stats("Q_0").squeeze()
    Q_0 = np.concatenate(Q_0_raw)[None, ::]
    ess_Q_0 = az.ess(np.array(Q_0, np.float64))
    Q_0_var = Q_0.var() / ess_Q_0

    Q_diff_means = []
    Q_diff_vars = []
    for l in range(1, levels):
        Q_diff_raw = trace.get_sampler_stats(f"Q_{l}_{l-1}").squeeze()
        Q_diff = np.hstack(Q_diff_raw)[None, ::]
        ess_diff = az.ess(np.array(Q_diff, np.float64))
        Q_diff_means.append(Q_diff.mean())
        Q_diff_vars.append(Q_diff.var() / ess_diff)

    Q_mean = Q_0.mean() + sum(Q_diff_means)
    Q_se = np.sqrt(Q_0_var + sum(Q_diff_vars))
    return Q_mean, Q_se


def subsample(
    draws=1,
    step=None,
    start=None,
    trace=None,
    tune=0,
    model=None,
):
    """
    A stripped down version of sample(), which is called only
    by the RecursiveDAProposal (which is the proposal used in the MLDA
    stepper). RecursiveDAProposal only requires a small set of the input
    parameters and checks normally performed by sample(), and this
    function thus skips some of the code in sampler(). It directly calls
    _iter_sample(), rather than sample_many(). The result is a reduced
    overhead when running multiple levels in MLDA.
    """

    model = pm.modelcontext(model)
    chain = 0
    random_seed = np.random.randint(2**30)
    callback = None

    draws += tune

    sampling = pm.sampling._iter_sample(
        draws, step, start, trace, chain, tune, model, random_seed, callback
    )

    try:
        for it, (trace, _) in enumerate(sampling):
            pass
    except KeyboardInterrupt:
        pass

    return trace


# Available proposal distributions for MLDA


class RecursiveDAProposal(Proposal):
    """
    Recursive Delayed Acceptance proposal to be used with MLDA step sampler.
    Recursively calls an MLDA sampler if level > 0 and calls MetropolisMLDA or
    DEMetropolisZMLDA sampler if level = 0. The sampler generates
    self.subsampling_rate samples and returns the sample with index
    self.subchain_selection to be used as a proposal.
    Results in a hierarchy of chains each of which is used to propose
    samples to the chain above.
    """

    def __init__(
        self,
        step_method_below: Union[MLDA, MetropolisMLDA, DEMetropolisZMLDA, CompoundStep],
        model_below: Model,
        tune: bool,
        subsampling_rate: int,
    ) -> None:

        self.step_method_below = step_method_below
        self.model_below = model_below
        self.tune = tune
        self.subsampling_rate = subsampling_rate
        self.subchain_selection = None
        self.tuning_end_trigger = True

    def __call__(self, q0: RaveledVars) -> RaveledVars:
        """Returns proposed sample given the current sample
        in dictionary form (q0_dict)."""

        # Logging is reduced to avoid extensive console output
        # during multiple recursive calls of subsample()
        _log = logging.getLogger("pymc")
        _log.setLevel(logging.ERROR)

        # Convert current sample from RaveledVars ->
        # dict before feeding to subsample.
        q0_dict = DictToArrayBijection.rmap(q0)

        with self.model_below:
            # Check if the tuning flag has been set to False
            # in which case tuning is stopped. The flag is set
            # to False (by MLDA's astep) when the burn-in
            # iterations of the highest-level MLDA sampler run out.
            # The change propagates to all levels.

            if self.tune:
                # Subsample in tuning mode
                trace = subsample(
                    draws=0,
                    step=self.step_method_below,
                    start=q0_dict,
                    tune=self.subsampling_rate,
                )
            else:
                # Subsample in normal mode without tuning
                # If DEMetropolisZMLDA is the base sampler a flag is raised to
                # make sure that history is edited after tuning ends
                if self.tuning_end_trigger:
                    if isinstance(self.step_method_below, DEMetropolisZMLDA):
                        self.step_method_below.tuning_end_trigger = True
                    self.tuning_end_trigger = False

                trace = subsample(
                    draws=self.subsampling_rate,
                    step=self.step_method_below,
                    start=q0_dict,
                    tune=0,
                )

        # set logging back to normal
        _log.setLevel(logging.NOTSET)

        # return sample with index self.subchain_selection from the generated
        # sequence of length self.subsampling_rate. The index is set within
        # MLDA's astep() function
        q_dict = trace.point(self.subchain_selection)

        # Make sure output dict is ordered the same way as the input dict.
        q_dict = Point(
            {key: q_dict[key] for key in q0_dict.keys()},
            model=self.model_below,
            filter_model_vars=True,
        )

        return DictToArrayBijection.map(q_dict)
