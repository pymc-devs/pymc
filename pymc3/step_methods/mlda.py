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

import arviz as az
import numpy as np
import warnings
import logging
from typing import Union, List, Optional, Type
import theano
import theano.tensor as tt

from .arraystep import ArrayStepShared, metrop_select, Competence
from .compound import CompoundStep
from .metropolis import Proposal, Metropolis, DEMetropolisZ, delta_logp, tune
from ..model import Model
import pymc3 as pm
from pymc3.theanof import floatX

__all__ = ["MetropolisMLDA", "DEMetropolisZMLDA", "RecursiveDAProposal", "MLDA", "extract_Q_estimate"]


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

        # flag to that variance reduction is activated - forces MetropolisMLDA
        # to store quantities of interest in a register if True
        self.mlda_variance_reduction = kwargs.pop("mlda_variance_reduction", False)
        if self.mlda_variance_reduction:
            # Subsampling rate of MLDA sampler one level up
            self.mlda_subsampling_rate_above = kwargs.pop("mlda_subsampling_rate_above")
            self.sub_counter = 0
            self.Q_last = np.nan
            self.Q_reg = [np.nan] * self.mlda_subsampling_rate_above

            # extract some necessary variables
            model = pm.modelcontext(kwargs.get("model", None))
            vars = kwargs.get("vars", None)
            if vars is None:
                vars = model.vars
            vars = pm.inputvars(vars)
            shared = pm.make_shared_replacements(vars, model)

        # call parent class __init__
        super().__init__(*args, **kwargs)

        # modify the delta function and point to model if VR is used
        if self.mlda_variance_reduction:
            self.delta_logp = delta_logp_inverse(model.logpt, vars, shared)
            self.model = model

    def reset_tuning(self):
        """
        Does not reset sampler parameters. Allows continuation with
        the same settings when MetropolisMLDA steps are done in chunks
        under MLDA.
        """
        return

    def astep(self, q0):
        """Modified astep that is compatible with MLDA features"""
        if not self.steps_until_tune and self.tune:
            # Tune scaling parameter
            self.scaling = tune(
                self.scaling, self.accepted / float(self.tune_interval))
            # Reset counter
            self.steps_until_tune = self.tune_interval
            self.accepted = 0

        delta = self.proposal_dist() * self.scaling

        if self.any_discrete:
            if self.all_discrete:
                delta = np.round(delta, 0).astype('int64')
                q0 = q0.astype('int64')
                q = (q0 + delta).astype('int64')
            else:
                delta[self.discrete] = np.round(
                    delta[self.discrete], 0)
                q = (q0 + delta)
        else:
            q = floatX(q0 + delta)

        accept = self.delta_logp(q, q0)
        q_new, accepted = metrop_select(accept, q, q0)
        self.accepted += accepted

        if self.mlda_variance_reduction:
            if accepted:
                self.Q_last = self.model.Q.get_value()
            if self.sub_counter == self.mlda_subsampling_rate_above:
                self.sub_counter = 0
            self.Q_reg[self.sub_counter] = self.Q_last
            self.sub_counter += 1

        self.steps_until_tune -= 1

        stats = {
            'tune': self.tune,
            'scaling': self.scaling,
            'accept': np.exp(accept),
            'accepted': accepted,
        }

        return q_new, [stats]


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

        # flag to that variance reduction is activated - forces DEMetropolisZMLDA
        # to store quantities of interest in a register if True
        self.mlda_variance_reduction = kwargs.pop("mlda_variance_reduction", False)
        if self.mlda_variance_reduction:
            # Subsampling rate of MLDA sampler one level up
            self.mlda_subsampling_rate_above = kwargs.pop("mlda_subsampling_rate_above")
            self.sub_counter = 0
            self.Q_last = np.nan
            self.Q_reg = [np.nan] * self.mlda_subsampling_rate_above

            # extract some necessary variables
            model = pm.modelcontext(kwargs.get("model", None))
            vars = kwargs.get("vars", None)
            if vars is None:
                vars = model.vars
            vars = pm.inputvars(vars)
            shared = pm.make_shared_replacements(vars, model)

        # call parent class __init__
        super().__init__(*args, **kwargs)

        # modify the delta function and point to model if VR is used
        if self.mlda_variance_reduction:
            self.delta_logp = delta_logp_inverse(model.logpt, vars, shared)
            self.model = model

    def reset_tuning(self):
        """Skips resetting of tuned sampler parameters
        and history to their initial values. Allows
        continuation with the same settings when
        DEMetropolisZMLDA steps are done in chunks
        under MLDA."""
        return

    def astep(self, q0):
        """Modified astep that is compatible with MLDA features"""

        # same tuning scheme as DEMetropolis
        if not self.steps_until_tune and self.tune:
            if self.tune_target == 'scaling':
                self.scaling = tune(self.scaling, self.accepted / float(self.tune_interval))
            elif self.tune_target == 'lambda':
                self.lamb = tune(self.lamb, self.accepted / float(self.tune_interval))
            # Reset counter
            self.steps_until_tune = self.tune_interval
            self.accepted = 0

        epsilon = self.proposal_dist() * self.scaling

        it = len(self._history)
        # use the DE-MCMC-Z proposal scheme as soon as the history has 2 entries
        if it > 1:
            # differential evolution proposal
            # select two other chains
            iz1 = np.random.randint(it)
            iz2 = np.random.randint(it)
            while iz2 == iz1:
                iz2 = np.random.randint(it)

            z1 = self._history[iz1]
            z2 = self._history[iz2]
            # propose a jump
            q = floatX(q0 + self.lamb * (z1 - z2) + epsilon)
        else:
            # propose just with noise in the first 2 iterations
            q = floatX(q0 + epsilon)

        accept = self.delta_logp(q, q0)
        q_new, accepted = metrop_select(accept, q, q0)
        self.accepted += accepted
        self._history.append(q_new)

        if self.mlda_variance_reduction:
            if accepted:
                self.Q_last = self.model.Q.get_value()
            if self.sub_counter == self.mlda_subsampling_rate_above:
                self.sub_counter = 0
            self.Q_reg[self.sub_counter] = self.Q_last
            self.sub_counter += 1

        self.steps_until_tune -= 1

        stats = {
            'tune': self.tune,
            'scaling': self.scaling,
            'lambda': self.lamb,
            'accept': np.exp(accept),
            'accepted': accepted
        }

        return q_new, [stats]

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
        Compilation mode passed to Theano functions
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
        the PyMC3 model (also demonstrated in the example notebook):
            - Include a `pm.Data()` variable with the name `Q` in the
            model description of all levels.
            - Use a Theano Op to calculate the forward model (or the
            combination of a forward model and a likelihood). This Op
            should have a `perform()` method which (in addition to all
            the other calculations), calculates the quantity of interest
            and stores it to the variable `Q` of the PyMC3 model,
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
            - Use a Theano Op to define the forward model (and
            optionally the likelihood) for all levels. The Op needs
            to store the result of each forward model calculation
            to the variable model_output of the PyMC3 model,
            using the `set_value()` function.
            - Define a Multivariate Normal likelihood (either using
            the standard PyMC3 API or within an Op) which has mean
            equal to the forward model output plus mu_B and covariance
            equal to the model error plus Sigma_B.
        Given the above, MLDA will capture and iteratively update the
        bias terms internally for all level pairs and will correct
        each level so that all levels' forward models aim to estimate
        the finest level's forward model.

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
    .. [Cui2012] Cui, Tiangang & Fox, Colin & O'Sullivan, Michael.
    (2012). Adaptive Error Modelling in MCMC Sampling for Large
    Scale Inverse Problems.
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
        base_scaling: Optional = None,
        tune: bool = True,
        base_tune_target='lambda',
        base_tune_interval: int = 100,
        base_lamb=None,
        base_tune_drop_fraction=0.9,
        model: Optional[Model] = None,
        mode: Optional = None,
        subsampling_rates: List[int] = 5,
        base_blocked: bool = False,
        variance_reduction=False,
        store_Q_fine=False,
        adaptive_error_model=False,
        **kwargs
    ) -> None:

        # this variable is used to identify MLDA objects which are
        # not in the finest level (i.e. child MLDA objects)
        self.is_child = kwargs.get("is_child", False)
        if not self.is_child:
            warnings.warn(
                'The MLDA implementation in PyMC3 is very young. '
                'You should be extra critical about its results.'
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
        self.model = model
        self.variance_reduction = variance_reduction
        self.store_Q_fine = store_Q_fine
        # check that certain requirements hold
        # for the variance reduction feature to work
        if self.variance_reduction or self.store_Q_fine:
            if not hasattr(self.model, 'Q'):
                raise AttributeError("Model given to MLDA does not contain"
                                     "variable 'Q'. You need to include"
                                     "the variable in the model definition"
                                     "for variance reduction to work or"
                                     "for storing the fine Q."
                                     "Use pm.Data() to define it.")
            if not isinstance(self.model.Q, tt.sharedvar.TensorSharedVariable):
                raise TypeError("The variable 'Q' in the model definition is not of type "
                                "'TensorSharedVariable'. Use pm.Data() to define the"
                                "variable.")

        self.next_model = self.coarse_models[-1]
        self.adaptive_error_model = adaptive_error_model
        # check that certain requirements hold
        # for the adaptive error model feature to work
        if self.adaptive_error_model:
            if not hasattr(self.next_model, 'mu_B'):
                raise AttributeError("Next model in hierarchy does not contain"
                                     "variable 'mu_B'. You need to include"
                                     "the variable in the model definition"
                                     "for adaptive error model to work."
                                     "Use pm.Data() to define it.")
            if not hasattr(self.next_model, 'Sigma_B'):
                raise AttributeError("Next model in hierarchy does not contain"
                                     "variable 'Sigma_B'. You need to include"
                                     "the variable in the model definition"
                                     "for adaptive error model to work."
                                     "Use pm.Data() to define it.")
            if not (isinstance(self.next_model.mu_B, tt.sharedvar.TensorSharedVariable) and
                    isinstance(self.next_model.Sigma_B, tt.sharedvar.TensorSharedVariable)):
                raise TypeError("At least one of the variables 'mu_B' and 'Sigma_B' "
                                "in the next model's definition is not of type "
                                "'TensorSharedVariable'. Use pm.Data() to define those "
                                "variables.")

            # this object is used to recursively update the mean and
            # variance of the bias correction given new differences
            # between levels
            self.bias = RecursiveSampleMoments(self.next_model.mu_B.get_value(),
                                               self.next_model.Sigma_B.get_value())

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
        if self.is_child and self.variance_reduction:
            # this is the subsampling rate applied to the current level
            # it is stored in the level above and transferred here
            self.subsampling_rate_above = kwargs.pop("subsampling_rate_above", None)

        self.num_levels = len(self.coarse_models) + 1
        self.base_sampler = base_sampler

        # VR is not compatible with compound base samplers so an automatic conversion
        # to a block sampler happens here if
        if self.variance_reduction and self.base_sampler == 'Metropolis' and not base_blocked:
            warnings.warn(
                'Variance reduction is not compatible with non-blocked (compound) samplers.'
                'Automatically switching to a blocked Metropolis sampler.'
            )
            self.base_blocked = True
        else:
            self.base_blocked = base_blocked
        self.next_model = self.coarse_models[-1]
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
        self.mode = mode
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
        self.delta_logp = delta_logp_inverse(model.logpt, vars, shared)

        # Construct theano function for next-level model likelihood
        # (for use in acceptance)
        next_model = pm.modelcontext(self.next_model)
        vars_next = [var for var in next_model.vars if var.name in self.var_names]
        vars_next = pm.inputvars(vars_next)
        shared_next = pm.make_shared_replacements(vars_next, next_model)
        self.delta_logp_next = delta_logp(next_model.logpt, vars_next, shared_next)
        self.model = model
        super().__init__(vars, shared)

        # initialise complete step method hierarchy
        if self.num_levels == 2:
            with self.next_model:
                # make sure the correct variables are selected from next_model
                vars_next = [
                    var for var in self.next_model.vars if var.name in self.var_names
                ]

                # create kwargs
                if self.variance_reduction:
                    base_kwargs = {"mlda_subsampling_rate_above": self.subsampling_rate,
                                   "mlda_variance_reduction": True}
                else:
                    base_kwargs = {}

                if self.base_sampler == 'Metropolis':
                    # MetropolisMLDA sampler in base level (level=0), targeting self.next_model
                    self.next_step_method = pm.MetropolisMLDA(vars=vars_next,
                                                              proposal_dist=self.base_proposal_dist,
                                                              S=self.base_S,
                                                              scaling=self.base_scaling, tune=self.tune,
                                                              tune_interval=self.base_tune_interval,
                                                              model=None,
                                                              mode=self.mode,
                                                              blocked=self.base_blocked,
                                                              ** base_kwargs)
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
                                                                 mode=self.mode,
                                                                 ** base_kwargs)
        else:
            # drop the last coarse model
            next_coarse_models = self.coarse_models[:-1]
            next_subsampling_rates = self.subsampling_rates[:-1]
            with self.next_model:
                # make sure the correct variables are selected from next_model
                vars_next = [
                    var for var in self.next_model.vars if var.name in self.var_names
                ]

                # create kwargs
                if self.variance_reduction:
                    mlda_kwargs = {"is_child": True,
                                   "subsampling_rate_above": self.subsampling_rate}
                else:
                    mlda_kwargs = {"is_child": True}
                if self.adaptive_error_model:
                    mlda_kwargs = {**mlda_kwargs, **{"bias_all": self.bias_all}}

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
                                                variance_reduction=self.variance_reduction,
                                                store_Q_fine=False,
                                                adaptive_error_model=self.adaptive_error_model,
                                                **mlda_kwargs)

        # instantiate the recursive DA proposal.
        # this is the main proposal used for
        # all levels (Recursive Delayed Acceptance)
        # (except for level 0 where the step method is MetropolisMLDA
        # or DEMetropolisZMLDA - not MLDA)
        self.proposal_dist = RecursiveDAProposal(
            self.next_step_method,
            self.next_model,
            self.tune,
            self.subsampling_rate
        )

        # add 'base_lambda' to stats if 'DEMetropolisZ' is used
        if self.base_sampler == 'DEMetropolisZ':
            self.stats_dtypes[0]['base_lambda'] = np.float64

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
                    self.stats_dtypes[0][f'Q_{level}_{level - 1}'] = object
                self.stats_dtypes[0]['Q_0'] = object

        # initialise necessary variables for doing variance reduction or storing fine Q
        if self.variance_reduction or self.store_Q_fine:
            self.Q_last = np.nan
            self.Q_diff_last = np.nan
        if self.store_Q_fine and not self.is_child:
            self.stats_dtypes[0][f'Q_{self.num_levels - 1}'] = object

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

        # Variance reduction
        self.update_vr_variables(accepted, skipped_logp)

        # Adaptive error model - runs only during tuning.
        if self.tune and self.adaptive_error_model:
            self.update_error_estimate(accepted, skipped_logp)

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

        # Save the VR statistics to the stats dictionary (only happens in the
        # top MLDA level)
        if (self.variance_reduction or self.store_Q_fine) and not self.is_child:
            q_stats = {}
            if self.variance_reduction:
                m = self
                for level in range(self.num_levels - 1, 0, -1):
                    # save the Q differences for this level and iteration
                    q_stats[f'Q_{level}_{level - 1}'] = np.array(m.Q_diff)
                    # this makes sure Q_diff is reset for
                    # the next iteration
                    m.Q_diff = []
                    if level == 1:
                        break
                    m = m.next_step_method
                q_stats['Q_0'] = np.array(m.Q_base_full)
                m.Q_base_full = []
            if self.store_Q_fine:
                q_stats['Q_' + str(self.num_levels - 1)] = np.array(self.Q_last)
            stats = {**stats, **q_stats}

        return q_new, [stats]

    def update_vr_variables(self, accepted, skipped_logp):
        """Updates all the variables necessary for VR to work.

        Each level has a Q_last and Q_diff_last register which store
        the Q of the last accepted MCMC sample and the difference
        between the Q of the last accepted sample in this level and
        the Q of the last sample in the level below.

        These registers are updated here so that they can be exported later."""

        # if sample is accepted, update self.Q_last with the sample's Q value
        # runs only for VR or when store_Q_fine is True
        if self.variance_reduction or self.store_Q_fine:
            if accepted and not skipped_logp:
                self.Q_last = self.model.Q.get_value()

        if self.variance_reduction:
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
                self.Q_base_full.extend(self.next_step_method.Q_reg)

            # if the sample is accepted, update Q_diff_last with the latest
            # difference between the last Q of this level and the Q of the
            # proposed (selected) sample from the level below.
            # If sample is not accepted, just keep the latest accepted Q_diff
            if accepted and not skipped_logp:
                self.Q_diff_last = self.Q_last -\
                                   self.next_step_method.Q_reg[self.subchain_selection]
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
            self.last_synced_output_diff = self.model.model_output.get_value() - \
                                           self.next_model.model_output.get_value()
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
            with self.next_model:
                pm.set_data({'mu_B': sum([bias.get_mu() for bias in
                                          self.bias_all[:len(self.bias_all) - self.num_levels + 2]])})
                pm.set_data({'Sigma_B': sum([bias.get_sigma() for bias in
                                             self.bias_all[:len(self.bias_all) - self.num_levels + 2]])})

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

        self.sigma = (self.t - 1) / self.t * self.sigma + 1 / self.t * \
                     (self.t * np.outer(mu_previous, mu_previous) -
                     (self.t + 1) * np.outer(self.mu, self.mu) +
                     np.outer(x, x))

        self.t += 1


def delta_logp_inverse(logp, vars, shared):
    [logp0], inarray0 = pm.join_nonshared_inputs([logp], vars, shared)

    tensor_type = inarray0.type
    inarray1 = tensor_type('inarray1')

    logp1 = pm.CallableTensor(logp0)(inarray1)

    f = theano.function([inarray1, inarray0], - logp0 + logp1)
    f.trust_input = True
    return f


def extract_Q_estimate(trace, levels):
    """
    Returns expectation and standard error of quantity of interest,
    given a trace and the number of levels in the multilevel model.
    It makes use of the collapsing sum formula. Only applicable when
    MLDA with variance reduction has been used for sampling.
    """

    Q_0_raw = trace.get_sampler_stats("Q_0")
    # total number of base level samples from all iterations
    total_base_level_samples = sum([it.shape[0] for it in Q_0_raw])
    Q_0 = np.concatenate(Q_0_raw).reshape((1, total_base_level_samples))
    ess_Q_0 = az.ess(np.array(Q_0, np.float64))
    Q_0_var = Q_0.var() / ess_Q_0

    Q_diff_means = []
    Q_diff_vars = []
    for l in range(1, levels):
        Q_diff_raw = trace.get_sampler_stats(f"Q_{l}_{l-1}")
        # total number of samples from all iterations
        total_level_samples = sum([it.shape[0] for it in Q_diff_raw])
        Q_diff = np.concatenate(Q_diff_raw).reshape((1, total_level_samples))
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
    random_seed=None,
    callback=None,
    **kwargs
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
    random_seed = np.random.randint(2 ** 30)

    if start is not None:
        pm.sampling._check_start_shape(model, start)
    else:
        start = {}

    draws += tune

    step = pm.sampling.assign_step_methods(model, step, step_kwargs=kwargs)

    if isinstance(step, list):
        step = CompoundStep(step)

    sampling = pm.sampling._iter_sample(draws, step, start, trace, chain, tune, model, random_seed, callback)

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
    def __init__(self,
                 next_step_method: Union[MLDA, MetropolisMLDA, DEMetropolisZMLDA, CompoundStep],
                 next_model: Model,
                 tune: bool,
                 subsampling_rate: int) -> None:

        self.next_step_method = next_step_method
        self.next_model = next_model
        self.tune = tune
        self.subsampling_rate = subsampling_rate
        self.subchain_selection = None
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
                self.trace = subsample(draws=0, step=self.next_step_method,
                                       start=q0_dict, trace=self.trace,
                                       tune=self.subsampling_rate)
            else:
                # Subsample in normal mode without tuning
                # If DEMetropolisZMLDA is the base sampler a flag is raised to
                # make sure that history is edited after tuning ends
                if self.tuning_end_trigger:
                    if isinstance(self.next_step_method, DEMetropolisZMLDA):
                        self.next_step_method.tuning_end_trigger = True
                    self.tuning_end_trigger = False
                
                self.trace = subsample(draws=self.subsampling_rate,
                                       step=self.next_step_method,
                                       start=q0_dict, trace=self.trace)

        # set logging back to normal
        _log.setLevel(logging.NOTSET)

        # return sample with index self.subchain_selection from the generated
        # sequence of length self.subsampling_rate. The index is set within
        # MLDA's astep() function
        return self.trace.point(- self.subsampling_rate + self.subchain_selection)
