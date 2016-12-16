import collections
import theano.tensor as tt
import theano
from ..theanof import tt_rng
from ..distributions.dist_math import rho2sd, log_normal3
from ..math import flatten_list
import numpy as np


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

    @staticmethod
    def new_global_dict():
        raise NotImplementedError

    @staticmethod
    def new_local_dict():
        local_dict = dict(
            means=collections.OrderedDict(),
            rhos=collections.OrderedDict(),
            x=list(),
        )
        return local_dict

    def new_rep_glob_loc(self):
        """
        Returns
        -------
        empty_replacements, new_global_dict, new_local_dict
        """
        g, l = self.new_global_dict(), self.new_local_dict()
        # specify type for some probable generic purposes
        g['__type__'] = self.__class__.__name__
        l['__type__'] = self.__class__.__name__
        return collections.OrderedDict(), g, l

    @staticmethod
    def known_node(local_dict, node, *args):
        """
        Parameters
        ----------
        local_dict: placeholder for local params
        node : node to be replaced
        args : mu, rho

        Returns
        -------
        replacement for given node
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
        starting poing is always the following:
        >>> replacements, global_dict, local_dict = self.new_rep_glob_loc()
        Just for not having any side effect

        Returns
        -------
        replacements, global_dict, local_dict
        """
        raise NotImplementedError

    @property
    def deterministic_replacements(self):
        replacements = collections.OrderedDict()
        replacements.update(self.deterministic_replacements_global)
        replacements.update(self.deterministic_replacements_local)
        return replacements

    @property
    def deterministic_replacements_local(self):
        return collections.OrderedDict(
            [(self.model[k], v) for k, v in self.local_dict['means'].items()]
        )

    @property
    def deterministic_replacements_global(self):
        raise NotImplementedError

    @property
    def params(self):
        """
        Returns
        -------
        list - shared params to fit
        """
        return self.params_local + self.params_global

    @property
    def params_local(self):
        """
        Returns
        -------
        list - shared params for local replacements
        """
        return []

    @property
    def params_global(self):
        """
        Returns
        -------
        list - shared params for global replacements
        """
        raise NotImplementedError

    @property
    def log_q_W_local(self):
        if self.local_dict['x']:
            x = flatten_list(self.local_dict['x'])
            mu = flatten_list(self.local_dict['means'].values())
            rho = flatten_list(self.local_dict['rhos'].values())
            _log_q_W_local_ = tt.sum(log_normal3(x, mu, rho))
            return _log_q_W_local_
        else:
            return 0

    @property
    def log_q_W_global(self):
        raise NotImplementedError

    @property
    def log_q_W(self):
        return self.log_q_W_global + self.log_q_W_local

    @property
    def log_p_D(self):
        _log_p_D_ = tt.add(
            *map(self.weighted_likelihood, self.model.observed_RVs)
        )
        return _log_p_D_

    @property
    def log_p_W(self):
        _log_p_W_ = self.model.varlogpt + tt.sum(self.model.potentials)
        return _log_p_W_

    @property
    def elbo(self):
        return self.log_p_D + self.log_p_W - self.log_q_W

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
        """Output of this function should be used for fitting a model

        Parameters
        ----------
        samples : Tensor - number of samples
        pi : Tensor - weight of variational part


        Notes
        -----
        In case of samples == 1 updates are empty and can be ignored

        Returns
        -------
        elbos, updates
        """
        samples = tt.as_tensor(samples)
        pi = tt.as_tensor(pi)
        _elbo_ = self.log_p_D + pi * (self.log_p_W - self.log_q_W)
        _elbo_ = self.apply_replacements(_elbo_)
        elbos, updates = theano.scan(fn=lambda: _elbo_,
                                     outputs_info=None,
                                     n_steps=samples)
        return elbos, updates


class MeanField(BaseReplacement):
    @staticmethod
    def random_node(global_dict, node):
        """Creates random node with shared params and
        places shared parameters to global dict

        Parameters
        ----------
        global_dict : dict - placeholder for parameters
        node : pm.FreeRV

        Returns
        -------
        tt.Variable : new node
        """
        if len(node.broadcastable) > 0:
            rho = theano.shared(
                np.ones(node.tag.test_value.shape),
                name='{}_rho_shared'.format(node.name),
                broadcastable=node.broadcastable)
            mu = theano.shared(
                node.tag.test_value,
                name='{}_mu_shared'.format(node.name),
                broadcastable=node.broadcastable)
            e = tt.patternbroadcast(
                tt_rng().normal(rho.shape), node.broadcastable)
        else:
            rho = theano.shared(
                np.ones(node.tag.test_value.shape),
                name='{}_rho_shared'.format(node.name))
            mu = theano.shared(
                node.tag.test_value,
                name='{}_mu_shared'.format(node.name))
            e = tt_rng().normal(rho.shape)
        v = mu + rho2sd(rho) * e
        global_dict['means'][node.name] = mu
        global_dict['rhos'][node.name] = rho
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
    def log_q_W_global(self):
        if self.global_dict['x']:
            x = flatten_list(self.global_dict['x'])
            mu = flatten_list(self.global_dict['means'].values())
            rho = flatten_list(self.global_dict['rhos'].values())
            _log_q_W_global_ = tt.sum(log_normal3(x, mu, rho))
            return _log_q_W_global_
        else:
            return 0

    @property
    def params_global(self):
        return (list(self.global_dict['means'].values()) +
                list(self.global_dict['rhos'].values()))

    @property
    def deterministic_replacements_global(self):
        return collections.OrderedDict(
            [(self.model[k], v) for k, v in self.global_dict['means'].items()]
        )
