import numpy as np
import theano
from theano import tensor as tt
import tqdm

import pymc3 as pm
from pymc3.distributions.dist_math import log_normal, rho2sd, log_normal_mv
from pymc3.variational.opvi import Operator, Approximation, TestFunction


__all__ = [
    'TestFunction',
    'KL',
    'MeanField',
    'FullRank',
    'ADVI',
    'FullRankADVI',
    'Inference'
]
# OPERATORS


class KL(Operator):
    def apply(self, f):
        """
        KL divergence between posterior and approximation for input `z`
            :math:`z ~ Approximation`
        """
        z = self.input
        return self.logq(z) - self.logp(z)

# APPROXIMATIONS


class MeanField(Approximation):
    def create_shared_params(self):
        return {'mu': theano.shared(
                    self.input.tag.test_value[self.global_slc]),
                'rho': theano.shared(
                    np.zeros((self.global_size,), dtype=theano.config.floatX))
                }

    def log_q_W_global(self, z):
        """
        log_q_W samples over q for global vars
        Gradient wrt mu, rho in density parametrization
        is set to zero to lower variance of ELBO
        """
        mu = self.shared_params['mu']
        rho = self.shared_params['rho']
        mu = self.scale_grad(mu)
        rho = self.scale_grad(rho)
        logq = tt.sum(log_normal(z[self.global_slc], mu, rho=rho))
        return logq

    def random_global(self, samples=None, no_rand=False):
        initial = self.initial(samples, no_rand, l=self.global_size)
        sd = rho2sd(self.shared_params['rho'])
        mu = self.shared_params['mu']
        return sd * initial + mu


class FullRank(Approximation):
    def __init__(self, local_rv=None, model=None, cost_part_grad_scale=1, gpu_compat=False):
        super(FullRank, self).__init__(
            local_rv=local_rv, model=model,
            cost_part_grad_scale=cost_part_grad_scale
        )
        self.gpu_compat = gpu_compat

    def create_shared_params(self):
        return {'mu': theano.shared(
                    self.input.tag.test_value[self.global_slc]),
                'L': theano.shared(
                    np.eye(self.global_size, dtype=theano.config.floatX))
                }

    def log_q_W_global(self, z):
        """
        log_q_W samples over q for global vars
        Gradient wrt mu, rho in density parametrization
        is set to zero to lower variance of ELBO
        """
        mu = self.shared_params['mu']
        L = self.shared_params['L']
        mu = self.scale_grad(mu)
        L = self.scale_grad(L)
        return log_normal_mv(z, mu, chol=L, gpu_compat=self.gpu_compat)

    def random_global(self, samples=None, no_rand=False):
        initial = self.initial(samples, no_rand, l=self.global_size)
        L = self.shared_params['L']
        mu = self.shared_params['mu']
        return initial.dot(L) + mu


class Inference(object):
    def __init__(self, op, approx, tf, local_rv=None, model=None, cost_part_grad_scale=1, **kwargs):
        self.hist = np.asarray(())
        self.objective = op(approx(
            local_rv=local_rv,
            model=model,
            cost_part_grad_scale=cost_part_grad_scale, **kwargs)
        )(tf)

    approx = property(lambda self: self.objective.approx)

    def fit(self, n=10000, callbacks=None, score_every=1, callback_every=1, **kwargs):
        if callbacks is None:
            callbacks = []
        sc_n_mc = kwargs.get('sc_n_mc')
        kwargs['score'] = False
        step_func = self.objective.step_function(**kwargs)
        if score_every is not None:
            score_func = self.objective.score_function(sc_n_mc)
        else:
            score_func = None
        i = 0
        j = 0
        if score_every is not None:
            scores = np.empty(n // score_every)
            scores[:] = np.nan
        else:
            scores = np.asarray(())
        logger = pm._log  # noqa
        progress = tqdm.trange(n)
        if score_every is not None:
            try:
                scores = np.empty(n // score_every)
                for i in progress:
                    step_func()
                    if i % score_every == 0:
                        e = score_func()
                        if np.isnan(e):
                            scores = scores[:j]
                            self.hist = np.concatenate([self.hist, scores])
                            raise FloatingPointError('NaN occurred in optimization.')
                        scores[j] = e
                        j += 1
                        if i % 10 == 0:
                            avg_elbo = scores[max(0, j - 1000):j].mean()
                            progress.set_description('Average Loss = {:,.5g}'.format(avg_elbo))
                    if i % callback_every == 0:
                        for callback in callbacks:
                            callback(self.approx, scores[:j], i)
            except KeyboardInterrupt:
                scores = scores[:j]
                if n < 10:
                    logger.info('Interrupted at {:,d} [{:.0f}%]: Loss = {:,.5g}'.format(
                        i, 100 * i // n, scores[i]))
                else:
                    avg_elbo = scores[min(0, j - 1000):j].mean()
                    logger.info('Interrupted at {:,d} [{:.0f}%]: Average Loss = {:,.5g}'.format(
                        i, 100 * i // n, avg_elbo))
            else:
                if n < 10:
                    logger.info('Finished [100%]: Loss = {:,.5g}'.format(scores[-1]))
                else:
                    avg_elbo = scores[max(0, j - 1000):j].mean()
                    logger.info('Finished [100%]: Average Loss = {:,.5g}'.format(avg_elbo))
        else:
            try:
                for _ in progress:
                    step_func()
            except KeyboardInterrupt:
                pass
        self.hist = np.concatenate([self.hist, scores])
        return self.approx


class ADVI(Inference):
    def __init__(self, local_rv=None, model=None, cost_part_grad_scale=1):
        super(ADVI, self).__init__(
            KL, MeanField, None,
            local_rv=local_rv, model=model, cost_part_grad_scale=cost_part_grad_scale)


class FullRankADVI(Inference):
    def __init__(self, local_rv=None, model=None, cost_part_grad_scale=1, gpu_compat=False):
        super(FullRankADVI, self).__init__(
            KL, FullRank, None,
            local_rv=local_rv, model=model, cost_part_grad_scale=cost_part_grad_scale, gpu_compat=gpu_compat)
