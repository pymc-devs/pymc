import theano.tensor as tt
import theano
from .utils import (variational_replacements,
                    apply_replacements,
                    flatten)
from ..distributions.dist_math import log_normal3

__all__ = [
    'sample_elbo'
]


def sample_elbo(model, population=None, samples=1, pi=1, vp=None):
    """ pi*KL[q(w|mu,rho)||p(w)] + E_q[log p(D|w)]
    approximated by Monte Carlo sampling

    Parameters
    ----------
    model : pymc3.Model
    population : dict - maps observed_RV to its population size
        if not provided defaults to full population
    samples : number of Monte Carlo samples used for approximation,
        defaults to 1
    pi : additional coefficient for KL[q(w|mu,rho)||p(w)] as proposed in [1]_
    vp : gelato.variational.utils.VariatioanalParams
        tuple, holding nodes mappings with shared params, if None - new
        will be created

    Returns
    -------
    (elbos, updates, VariationalParams)
        sampled elbos, updates for random streams, shared dicts

    Notes
    -----
    You can pass tensors for `pi`  and `samples` to control them while
        training

    References
    ----------
    .. [1] Charles Blundell et al: "Weight Uncertainty in Neural Networks"
        arXiv preprint arXiv:1505.05424
    """
    if population is None:
        population = dict()
    if vp is None:
        vp = variational_replacements(model.root)
    x = flatten(vp.mapping.values())
    mu = flatten(vp.shared.means.values())
    rho = flatten(vp.shared.rhos.values())

    def likelihood(var):
        tot = population.get(var, population.get(var.name))
        logpt = tt.sum(var.logpt)
        if tot is not None:
            tot = tt.as_tensor(tot)
            logpt *= tot / var.shape[0]
        return logpt

    log_p_D = tt.add(*map(likelihood, model.root.observed_RVs))
    log_p_W = model.root.varlogpt + tt.sum(model.root.potentials)
    log_q_W = tt.sum(log_normal3(x, mu, rho))
    _elbo_ = log_p_D + pi * (log_p_W - log_q_W)
    _elbo_ = apply_replacements(_elbo_, vp)

    samples = tt.as_tensor(samples)
    elbos, updates = theano.scan(fn=lambda: _elbo_,
                                 outputs_info=None,
                                 n_steps=samples)
    return elbos, updates, vp
