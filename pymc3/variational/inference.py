import numpy as np
import theano
import pymc3 as pm
from .approximations import Advi, BaseApproximation
from .advi import adagrad_optimizer
from tqdm import trange


APPROXIMATIONS = {
    'advi': Advi,
}


def approximate(n=10000, population=None, local_vars=None,
                optimizer=None, method='advi', samples=1, pi=1,
                callbacks=None, learning_rate=.001, epsilon=.1,
                model=None, more_params=None, more_updates=None,
                *args, **kwargs):
    """Interface for efficient variational inference with gradient
    variance reduction as described in [4]_


    PyMC3 supports the following variational inference methods:
        1) Automatic differentiation variational inference (ADVI) [2]_

    Parameters
    ----------
    n : int
        number of training iterations
    population : dict[Variable->int]
        maps observed_RV to its population size
        if not provided defaults to full population
        Note: population size is `shape[0]` of the whole data
    local_vars : dict[Variable->(mu, rho)]
        maps random variable to mu and rho
        for posterior parametrization it is used for Autoencoding Variational Bayes
        (AEVB; Kingma and Welling, 2014)[1]_
    optimizer : callable
        optional custom optimizer to be called in the following way:
            :code:`updates = optimizer(-elbo, list_of_params)`
    method : str|Approximation
        string description of approximation to be used or
        Approximation instance used to calculate elbo and provide for shared params
    samples : int|Tensor
        number of Monte Carlo samples used for approximation,
        defaults to 1
    pi : float|Tensor
        pi in [0;1] reweighting constant for KL divergence
        this trick was described in [3]_ for fine-tuning minibatch ADVI
    callbacks : list[callable]
        callables that will be called in the following way
            :code:`callback(Approximation, elbo_history, i)`
    learning_rate : float
        learning rate for adagrad optimizer,
        it will be ignored if optimizer is passed
    epsilon : float - epsilon for adagrad optimizer,
        it will be ignored if optimizer is passed
    model : Model
        Probabilistic model if function is called out of context
    more_params : list
        optional more parameters for computing gradients
    more_updates : dict
        more updates are included in step function
    args : additional args that will be passed to Approximation
    kwargs : additional kwargs that will be passed to Approximation

    Returns
    -------
    (Approximation, elbo_history)

    Notes
    -----
    Remember that you can manipulate pi and number of samples with callbacks

    References
    ----------
    .. [1] Kingma, D. P., & Welling, M. (2014).
      Auto-Encoding Variational Bayes. stat, 1050, 1.

    .. [2] Kucukelbir, A., Ranganath, R., Gelman, A., & Blei, D. (2015).
      Automatic variational inference in Stan. In Advances in neural
      information processing systems (pp. 568-576).

    .. [3] Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015).
      Weight Uncertainty in Neural Network. In Proceedings of the 32nd
      International Conference on Machine Learning (ICML-15) (pp. 1613-1622).

    .. [4] Geoffrey, R., Yuhuai, W., David, D. (2016)
        Sticking the Landing: A Simple Reduced-Variance Gradient for ADVI.
        NIPS Workshop
    """
    if isinstance(method, BaseApproximation):
        approx = method
    else:
        approx = APPROXIMATIONS[method](known=local_vars, population=population,
                                        model=model, *args, **kwargs)
    if callbacks is None:
        callbacks = []
    if optimizer is None:
        optimizer = adagrad_optimizer(learning_rate, epsilon)
    elbo = approx.elbo(samples=samples, pi=pi).mean()
    params = approx.params
    if more_params is not None:
        params += more_params
    updates = optimizer(-elbo, params)
    if more_updates is not None:
        more_updates.update(updates)
        updates = more_updates
    step = theano.function([], elbo, updates=updates)
    i = 0
    elbos = np.empty(n)
    try:
        progress = trange(n)
        for i in progress:
            e = step()
            elbos[i] = e
            for callback in callbacks:
                callback(approx, elbos[:i+1], i)
            if n < 10:
                progress.set_description('ELBO = {:,.5g}'.format(elbos[i]))
            elif i % (n // 10) == 0 and i > 0:
                avg_elbo = elbos[i - n // 10:i].mean()
                progress.set_description('Average ELBO = {:,.5g}'.format(avg_elbo))
    except KeyboardInterrupt:
        elbos = elbos[:i]
        if n < 10:
            pm._log.info('Interrupted at {:,d} [{:.0f}%]: ELBO = {:,.5g}'.format(
                i, 100 * i // n, elbos[i]))
        else:
            avg_elbo = elbos[i - n // 10:].mean()
            pm._log.info('Interrupted at {:,d} [{:.0f}%]: Average ELBO = {:,.5g}'.format(
                i, 100 * i // n, avg_elbo))
    else:
        if n < 10:
            pm._log.info('Finished [100%]: ELBO = {:,.5g}'.format(elbos[-1]))
        else:
            avg_elbo = elbos[-n // 10:].mean()
            pm._log.info('Finished [100%]: Average ELBO = {:,.5g}'.format(avg_elbo))
    return approx, elbos