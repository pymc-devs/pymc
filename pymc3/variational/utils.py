from collections import namedtuple, OrderedDict

import numpy as np
import theano
import theano.tensor as tt
from ..distributions.dist_math import rho2sd
from ..theanof import tt_rng

SharedNodes = namedtuple('SharedNodes', 'means, rhos')


class VariationalParams(namedtuple('VariationalParamsBase', 'mapping,shared')):
    @property
    def params(self):
        return list(self.shared.means.values())+list(self.shared.rhos.values())


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


def variational_replacements(model):
    """Util for getting variational replacements

    Parameters
    ----------
    model : pymc3.Model

    Returns
    -------
    VariationalParams : (mapping, SharedNodes)

    Notes
    -----
    Mappings and shared vars dicts are all OrderedDicts
    """
    replacements = OrderedDict()
    means = OrderedDict()
    rhos = OrderedDict()
    for var in model.vars:
        v, mu, rho = random_node(var)
        replacements[var] = v
        means[var.name] = mu
        rhos[var.name] = rho
    return VariationalParams(replacements, SharedNodes(means, rhos))


def apply_replacements(out, variational_params, deterministic=False):
    """

    Parameters
    ----------
    out : target node of the graph for which we apply replacements
    variational_params : VariationalParams
    deterministic : bool - whether do deterministic or stochastic replacements

    Returns
    -------
    tt.Variable
        cloned node with applied replacements
    """
    if deterministic:
        replacements = OrderedDict(zip(
            variational_params.mapping.keys(),
            variational_params.shared.means.values()
        ))
    else:
        replacements = variational_params.mapping
    return theano.clone(out, replacements, strict=False)


def flatten(tensors):
    joined = tt.concatenate([var.ravel() for var in tensors])
    return joined
