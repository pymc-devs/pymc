import collections
import theano.tensor as tt
import theano
from ..theanof import tt_rng
from ..distributions.dist_math import rho2sd, log_normal3
from ..math import flatten
import numpy as np


class Replacement(object):
    def __init__(self, model, population=None, local=None):
        if local is None:
            local = dict()
        if population is None:
            population = dict()
        self.population = population
        self.model = model
        s, g, l = self.create_mapping(model, local)
        self.stochastic_replacements = s
        self.global_dict = g
        self.local_dict = l

    @staticmethod
    def new_local_dict():
        local_dict = dict(
            means=collections.OrderedDict(),
            rhos=collections.OrderedDict(),
            x=list()
        )
        return local_dict

    @staticmethod
    def new_global_dict():
        raise NotImplementedError

    @staticmethod
    def known_node(local_dict, node, mu, rho):
        e = tt_rng().normal(rho.shape)
        v = mu + rho2sd(rho) * e
        local_dict['means'][node.name] = mu
        local_dict['rhos'][node.name] = rho
        local_dict['x'].append(v)
        return v

    def names2nodes(self, names):
        return [self.model[n] for n in names]

    def create_mapping(self, model, local):
        """

        Parameters
        ----------
        model - pm.Model
        local - local_RV

        Returns
        -------
        replacements, global_dict, local_dict
        """
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
        x = flatten(self.local_dict['x'])
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
            logpt /= var.size
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
    def random_node(global_dict, old):
        """Creates random node with shared params and
        places shared parameters to global dict

        Parameters
        ----------
        global_dict : dict - placeholder for parameters
        old : pm.FreeRV

        Returns
        -------
        tt.Variable : new node
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
        v = mu + rho2sd(rho) * e
        global_dict['means'][old.name] = mu
        global_dict['rhos'][old.name] = rho
        global_dict['x'].append(v)
        return v

    @staticmethod
    def new_global_dict():
        global_dict = dict(
            x=list(),
            means=collections.OrderedDict(),
            rhos=collections.OrderedDict()
        )
        return global_dict

    def create_mapping(self, model, local):
        local_dict = self.new_local_dict()
        global_dict = self.new_global_dict()
        replacements = collections.OrderedDict()
        for var in model.vars:
            if var in local:
                v = self.known_node(local_dict, var, *local[var])
            else:
                v = self.random_node(global_dict, var)
            replacements[var] = v
        return replacements, global_dict, local_dict

    @property
    def log_q_W_global(self):
        x = flatten(self.global_dict['x'])
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
