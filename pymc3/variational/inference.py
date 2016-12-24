import numpy as np
import theano
import pymc3 as pm
from .replacements import MeanField, BaseReplacement
from .advi import adagrad_optimizer
from tqdm import trange


APPROXIMATIONS = {
    'advi': MeanField,
    'meanfield': MeanField
}


def approximate(n=10000, population=None, local_vars=None,
                optimizer=None, method='advi', samples=1, callbacks=None,
                learning_rate=.001, epsilon=.1,
                *args, **kwargs):
    if isinstance(method, BaseReplacement):
        approx = method
    else:
        approx = APPROXIMATIONS[method](known=local_vars, population=population,
                                        *args, **kwargs)
    if callbacks is None:
        callbacks = []
    if optimizer is None:
        optimizer = adagrad_optimizer(learning_rate, epsilon)
    elbo = approx.elbo(samples).mean()
    updates = optimizer(-elbo, approx.params)
    step = theano.function([], elbo, updates=updates)
    i = 0
    elbos = np.empty(n)
    try:
        progress = trange(n)
        for i in progress:
            e = step()
            elbos[i] = e
            for callback in callbacks:
                callback(approx, e, i)
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