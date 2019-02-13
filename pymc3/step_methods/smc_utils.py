"""
SMC and SMC-ABC common functions
"""
import numpy as np
import pymc3 as pm
from ..backends.ndarray import NDArray
from ..backends.base import MultiTrace
from ..theanof import floatX


def _initial_population(draws, model, variables):
    """
    Create an initial population from the prior
    """

    population = []
    var_info = {}
    start = model.test_point
    init_rnd = pm.sample_prior_predictive(draws, model=model)
    for v in variables:
        var_info[v.name] = (start[v.name].shape, start[v.name].size)

    for i in range(draws):
        point = pm.Point({v.name: init_rnd[v.name][i] for v in variables}, model=model)
        population.append(model.dict_to_array(point))

    return np.array(floatX(population)), var_info


def _calc_covariance(posterior, weights):
    """
    Calculate trace covariance matrix based on importance weights.
    """
    cov = np.cov(posterior, aweights=weights.ravel(), bias=False, rowvar=0)
    if np.isnan(cov).any() or np.isinf(cov).any():
        raise ValueError('Sample covariances not valid! Likely "draws" is too small!')
    return np.atleast_2d(cov)


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
