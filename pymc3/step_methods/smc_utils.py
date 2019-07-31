"""
SMC and SMC-ABC common functions
"""
from collections import OrderedDict
import numpy as np
import pymc3 as pm
import theano
from .arraystep import metrop_select
from ..backends.ndarray import NDArray
from ..backends.base import MultiTrace
from ..theanof import floatX, join_nonshared_inputs
from ..util import get_untransformed_name, is_transformed_name


def _initial_population(draws, model, variables, start):
    """
    Create an initial population from the prior
    """

    population = []
    var_info = OrderedDict()
    if start is None:
        init_rnd = pm.sample_prior_predictive(
            draws, var_names=[v.name for v in model.unobserved_RVs], model=model
        )
    else:
        init_rnd = start

    init = model.test_point

    for v in variables:
        var_info[v.name] = (init[v.name].shape, init[v.name].size)

    for i in range(draws):

        point = pm.Point({v.name: init_rnd[v.name][i] for v in variables}, model=model)
        population.append(model.dict_to_array(point))

    return np.array(floatX(population)), var_info


def _calc_covariance(posterior, weights):
    """
    Calculate trace covariance matrix based on importance weights.
    """
    cov = np.cov(posterior, aweights=weights.ravel(), bias=False, rowvar=0)
    cov = np.atleast_2d(cov)
    cov += 1e-6 * np.eye(cov.shape[0])
    if np.isnan(cov).any() or np.isinf(cov).any():
        raise ValueError('Sample covariances not valid! Likely "draws" is too small!')
    return cov


def _tune(acc_rate, proposed, step):
    """
    Tune scaling and/or n_steps based on the acceptance rate.

    Parameters
    ----------
    acc_rate: float
        Acceptance rate of the previous stage
    proposed: int
        Total number of proposed steps (draws * n_steps)
    step: SMC step method
    """
    if step.tune_scaling:
        # a and b after Muto & Beck 2008.
        a = 1 / 9
        b = 8 / 9
        step.scaling = (a + b * acc_rate) ** 2
    if step.tune_steps:
        acc_rate = max(1.0 / proposed, acc_rate)
        step.n_steps = min(step.max_steps, 1 + int(np.log(step.p_acc_rate) / np.log(1 - acc_rate)))


def _posterior_to_trace(posterior, variables, model, var_info):
    """
    Save results into a PyMC3 trace
    """
    lenght_pos = len(posterior)
    varnames = [v.name for v in variables]

    with model:
        strace = NDArray(model)
        strace.setup(lenght_pos, 0)
    for i in range(lenght_pos):
        value = []
        size = 0
        for var in varnames:
            shape, new_size = var_info[var]
            value.append(posterior[i][size : size + new_size].reshape(shape))
            size += new_size
        strace.record({k: v for k, v in zip(varnames, value)})
    return MultiTrace([strace])


def metrop_kernel(
    q_old,
    old_tempered_logp,
    proposal,
    scaling,
    accepted,
    any_discrete,
    all_discrete,
    discrete,
    n_steps,
    prior_logp,
    likelihood_logp,
    beta,
    ABC,
):
    """
    Metropolis kernel
    """
    deltas = np.squeeze(proposal(n_steps) * scaling)
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

        new_tempered_logp = prior_logp(q_new) + ll * beta

        q_old, accept = metrop_select(new_tempered_logp - old_tempered_logp, q_new, q_old)
        if accept:
            accepted += 1
            old_tempered_logp = new_tempered_logp

    return q_old, accepted


def calc_beta(beta, likelihoods, threshold=0.5):
    """
    Calculate next inverse temperature (beta) and importance weights based on current beta
    and tempered likelihood.

    Parameters
    ----------
    beta : float
        tempering parameter of current stage
    likelihoods : numpy array
        likelihoods computed from the model
    threshold : float
        Determines the change of beta from stage to stage, i.e.indirectly the number of stages,
        the higher the value of threshold the higher the number of stage. Defaults to 0.5.
        It should be between 0 and 1.

    Returns
    -------
    new_beta : float
        tempering parameter of the next stage
    old_beta : float
        tempering parameter of the current stage
    weights : numpy array
        Importance weights (floats)
    sj : float
        Partial marginal likelihood
    """
    low_beta = old_beta = beta
    up_beta = 2.0
    rN = int(len(likelihoods) * threshold)

    while up_beta - low_beta > 1e-6:
        new_beta = (low_beta + up_beta) / 2.0
        weights_un = np.exp((new_beta - old_beta) * (likelihoods - likelihoods.max()))
        weights = weights_un / np.sum(weights_un)
        ESS = int(1 / np.sum(weights ** 2))
        if ESS == rN:
            break
        elif ESS < rN:
            up_beta = new_beta
        else:
            low_beta = new_beta
    if new_beta >= 1:
        new_beta = 1
    sj = np.exp((new_beta - old_beta) * likelihoods)
    weights_un = np.exp((new_beta - old_beta) * (likelihoods - likelihoods.max()))
    weights = weights_un / np.sum(weights_un)
    return new_beta, old_beta, weights, np.mean(sj)


def logp_forw(out_vars, vars, shared):
    """Compile Theano function of the model and the input and output variables.

    Parameters
    ----------
    out_vars : List
        containing :class:`pymc3.Distribution` for the output variables
    vars : List
        containing :class:`pymc3.Distribution` for the input variables
    shared : List
        containing :class:`theano.tensor.Tensor` for depended shared data
    """
    out_list, inarray0 = join_nonshared_inputs(out_vars, vars, shared)
    f = theano.function([inarray0], out_list[0])
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
        model,
        var_info,
        distance="absolute_error",
        sum_stat=False,
    ):
        """
        kernel : function
            a valid scipy.stats distribution. Defaults to `stats.norm`

        """
        self.epsilon = epsilon
        self.observations = observations
        self.function = function
        self.model = model
        self.var_info = var_info
        self.kernel = self.gauss_kernel
        self.dist_func = distance
        self.sum_stat = sum_stat

        if distance == "absolute_error":
            self.dist_func = self.absolute_error
        elif distance == "sum_of_squared_distance":
            self.dist_func = self.sum_of_squared_distance
        else:
            raise ValueError("Distance metric not understood")

    def posterior_to_function(self, posterior):
        model = self.model
        var_info = self.var_info
        parameters = {}
        size = 0
        for var, values in var_info.items():
            shape, new_size = values
            value = posterior[size : size + new_size].reshape(shape)
            if is_transformed_name(var):
                var = get_untransformed_name(var)
                value = model[var].transformation.backward_val(value)
            parameters[var] = value
            size += new_size
        return parameters

    def gauss_kernel(self, value):
        epsilon = self.epsilon
        return (-(value ** 2) / epsilon ** 2 + np.log(1 / (2 * np.pi * epsilon ** 2))) / 2.0

    def absolute_error(self, a, b):
        if self.sum_stat:
            return np.atleast_2d(np.abs(a.mean() - b.mean()))
        else:
            return np.mean(np.atleast_2d(np.abs(a - b)))

    def sum_of_squared_distance(self, a, b):
        if self.sum_stat:
            return np.sum(np.atleast_2d((a.mean() - b.mean()) ** 2))
        else:
            return np.mean(np.sum(np.atleast_2d((a - b) ** 2)))

    def __call__(self, posterior):
        """
        a : array
            vector of (simulated) data or summary statistics
        b : array
            vector of (observed) data or sumary statistics
        epsilon :
        """
        func_parameters = self.posterior_to_function(posterior)
        sim_data = self.function(**func_parameters)
        value = self.dist_func(self.observations, sim_data)
        return self.kernel(value)
