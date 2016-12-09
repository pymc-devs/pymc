import collections
import theano.tensor as tt
import theano
from ..theanof import tt_rng
from ..distributions.dist_math import rho2sd, log_normal3
import numpy as np


def flatten(tensors):
    joined = tt.concatenate([var.ravel() for var in tensors])
    return joined


class Replacement(object):
    def __init__(self, model, local=None, population=None):
        if local is None:
            local = dict()
        if population is None:
            population = dict()
        self.population = population
        self.model = model
        self._prepare_local_dict()
        self.stochastic_replacements, \
            self.global_dict = \
            self.create_mapping(model, local)

    def _prepare_local_dict(self):
        self.local_x = []
        self.local_dict = collections.OrderedDict()
        self.local_dict['means'] = collections.OrderedDict()
        self.local_dict['rhos'] = collections.OrderedDict()

    def known_node(self, _for_, mu, rho):
        e = tt_rng().normal(rho.shape)
        v = mu + rho2sd(rho) * e
        self.local_dict['means'][_for_.name] = mu
        self.local_dict['rhos'][_for_.name] = rho
        self.local_x.append(v)
        return v

    def names2nodes(self, names):
        return [self.model[n] for n in names]

    def create_mapping(self, model, local):
        # returns tuple(mapping, dict with shared trainable params)
        raise NotImplementedError

    @property
    def deterministic_replacements(self):
        """Method specific deterministic replacements
        """
        raise NotImplementedError

    @property
    def params(self):
        """Method specific parametrization
        """
        raise NotImplementedError

    @property
    def log_q_W_local(self):
        x = flatten(self.local_x)
        mu = flatten(self.local_dict['means'].values())
        rho = flatten(self.local_dict['rhos'].values())
        _log_q_W_local_ = tt.sum(log_normal3(x, mu, rho))
        return self.apply_replacements(_log_q_W_local_)

    @property
    def log_q_W_global(self):
        """Method specific log_q_W_global
        """
        raise NotImplementedError

    @property
    def log_q_W(self):
        return self.log_q_W_global + self.log_q_W_local

    @property
    def log_p_D(self):
        _log_p_D_ = tt.add(
            *map(self.weighted_likelihood, self.model.observed_RVs)
        )
        return self.apply_replacements(_log_p_D_)

    @property
    def log_p_W(self):
        _log_p_W_ = self.model.varlogpt + tt.sum(self.model.potentials)
        return self.apply_replacements(_log_p_W_)

    def apply_replacements(self, node, deterministic=False):
        if deterministic:
            return theano.clone(
                node, self.deterministic_replacements, strict=False)
        else:
            return theano.clone(
                node, self.stochastic_replacements, strict=False)

    def weighted_likelihood(self, var):
        tot = self.population.get(
            var, self.population.get(var.name))
        logpt = tt.sum(var.logpt)
        if tot is not None:
            tot = tt.as_tensor(tot)
            logpt *= tot
            if var.ndim >= 1:
                logpt /= var.shape[0]
        return logpt

    def sample_elbo(self, samples=1, pi=1):
        samples = tt.as_tensor(samples)
        pi = tt.as_tensor(pi)
        _elbo_ = self.log_p_D + pi * (self.log_p_W - self.log_q_W)
        elbos, updates = theano.scan(fn=lambda: _elbo_,
                                     outputs_info=None,
                                     n_steps=samples)
        return elbos, updates


class MeanField(Replacement):
    @staticmethod
    def random_node(old):
        """Creates random node with shared params

        Parameters
        ----------
        old : pm.FreeRV

        Returns
        -------
        tuple : (new node, shared mu, shared rho)
        """
        if len(old.broadcastable) > 0:
            rho = theano.shared(
                np.ones(old.tag.test_value.shape),
                name='{}_rho_shared'.format(old.name),
                broadcastable=old.broadcastable)
            mu = theano.shared(
                old.tag.test_value,
                name='{}_mu_shared'.format(old.name),
                broadcastable=old.broadcastable)
            e = tt.patternbroadcast(
                tt_rng().normal(rho.shape), old.broadcastable)
        else:
            rho = theano.shared(
                np.ones(old.tag.test_value.shape),
                name='{}_rho_shared'.format(old.name))
            mu = theano.shared(
                old.tag.test_value,
                name='{}_mu_shared'.format(old.name))
            e = tt_rng().normal(rho.shape)
        return mu + rho2sd(rho) * e, mu, rho

    def create_mapping(self, model, local):
        replacements = collections.OrderedDict()
        global_means = collections.OrderedDict()
        global_rhos = collections.OrderedDict()
        for var in model.vars:
            if var in local:
                v = self.known_node(var, *local[var])
            else:
                v, mu, rho = self.random_node(var)
                global_means[var.name] = mu
                global_rhos[var.name] = rho
            replacements[var] = v
        return (replacements,
                dict(means=global_means, rhos=global_rhos))

    @property
    def log_q_W_global(self):
        x = flatten(self.names2nodes(self.global_dict['means'].keys()))
        mu = flatten(self.global_dict['means'].values())
        rho = flatten(self.global_dict['rhos'].values())
        _log_q_W_global_ = tt.sum(log_normal3(x, mu, rho))
        return self.apply_replacements(_log_q_W_global_)

    @property
    def params(self):
        return (list(self.global_dict['means'].values()) +
                list(self.global_dict['rhos'].values()))

    @property
    def deterministic_replacements(self):
        return collections.OrderedDict(
            [(self.model[k], v for k, v in self.global_dict.items())]
        )
