import collections
import functools
import theano.tensor as tt
import theano
from ..theanof import tt_rng
from ..distributions.dist_math import rho2sd, log_normal3
from ..math import flatten_list
import numpy as np


def replace_out(func):
    @functools.wraps(func)
    def wrapped(self, *args, **kwargs):
        out = func(self, *args, **kwargs)
        return self.apply_replacements(out)
    return wrapped


class BaseReplacement(object):
    def __init__(self, model, population=None, known=None):
        if known is None:
            known = dict()
        if population is None:
            population = dict()
        self.population = population
        self.model = model
        self.known = known
        s, g, l = self.create_mapping()
        self.stochastic_replacements = s
        self.global_dict = g
        self.local_dict = l

    def new_global_dict(self):
        """
        :return: dict
        """
        raise NotImplementedError

    def new_local_dict(self):
        """
        :return: dict
        """
        local_dict = dict(
            means=collections.OrderedDict(),
            rhos=collections.OrderedDict(),
            x=list(),
            replacement=self.__class__.__name__
        )
        return local_dict

    def new_rep_glob_loc(self):
        """
        :return: empty_replacements, new_global_dict, new_local_dict
        """
        return collections.OrderedDict(), self.new_global_dict(), self.new_local_dict()

    @staticmethod
    def known_node(local_dict, node, *args):
        """
        :param local_dict: placeholder for local params
        :param node: node to be replaced
        :param args: mu, rho
        :return: replacement for given node
        """
        mu, rho = args
        e = tt_rng().normal(rho.shape)
        v = mu + rho2sd(rho) * e
        local_dict['means'][node.name] = mu
        local_dict['rhos'][node.name] = rho
        local_dict['x'].append(v)
        return v

    def create_mapping(self):
        """Implements creation of new replacements and parameters
        Returns
        -------
        replacements, global_dict, local_dict
        """
        raise NotImplementedError

    @property
    def deterministic_replacements(self):
        """
        :return: dict
        """
        replacements = collections.OrderedDict()
        replacements.update(self.deterministic_replacements_global)
        replacements.update(self.deterministic_replacements_local)
        return replacements

    @property
    def deterministic_replacements_local(self):
        """
        :return: dict
        """
        return collections.OrderedDict(
            [(self.model[k], v) for k, v in self.local_dict['means'].items()]
        )

    @property
    def deterministic_replacements_global(self):
        """
        :return: dict
        """
        raise NotImplementedError

    @property
    def params(self):
        """
        :return: list - shared params to fit
        """
        return self.params_local + self.params_global

    @property
    def params_local(self):
        """
        :return: list - shared params for local replacements
        """
        return []

    @property
    def params_global(self):
        """
        :return: list - shared params for global replacements
        """
        raise NotImplementedError

    @property
    @replace_out
    def log_q_W_local(self):
        x = flatten_list(self.local_dict['x'])
        mu = flatten_list(self.local_dict['means'].values())
        rho = flatten_list(self.local_dict['rhos'].values())
        _log_q_W_local_ = tt.sum(log_normal3(x, mu, rho))
        return _log_q_W_local_

    @property
    def log_q_W_global(self):
        raise NotImplementedError

    @property
    def log_q_W(self):
        return self.log_q_W_global + self.log_q_W_local

    @property
    @replace_out
    def log_p_D(self):
        _log_p_D_ = tt.add(
            *map(self.weighted_likelihood, self.model.observed_RVs)
        )
        return _log_p_D_

    @property
    @replace_out
    def log_p_W(self):
        _log_p_W_ = self.model.varlogpt + tt.sum(self.model.potentials)
        return _log_p_W_

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


class MeanField(BaseReplacement):
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

    def new_global_dict(self):
        global_dict = dict(
            x=list(),
            means=collections.OrderedDict(),
            rhos=collections.OrderedDict(),
            replacement=self.__class__.__name__
        )
        return global_dict

    def create_mapping(self):
        replacements, global_dict, local_dict = self.new_rep_glob_loc()
        for var in self.model.vars:
            if var in self.known:
                v = self.known_node(local_dict, var, *self.known[var])
            else:
                v = self.random_node(global_dict, var)
            replacements[var] = v
        return replacements, global_dict, local_dict

    @property
    @replace_out
    def log_q_W_global(self):
        x = flatten_list(self.global_dict['x'])
        mu = flatten_list(self.global_dict['means'].values())
        rho = flatten_list(self.global_dict['rhos'].values())
        _log_q_W_global_ = tt.sum(log_normal3(x, mu, rho))
        return _log_q_W_global_

    @property
    def params_global(self):
        return (list(self.global_dict['means'].values()) +
                list(self.global_dict['rhos'].values()))

    @property
    def deterministic_replacements_global(self):
        return collections.OrderedDict(
            [(self.model[k], v) for k, v in self.global_dict['means'].items()]
        )
