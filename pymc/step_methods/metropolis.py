from numpy.linalg import cholesky

from ..core import *
from quadpotential import quad_potential

from arraystep import *
from numpy.random import normal, standard_cauchy, standard_exponential

__all__ = ['Metropolis']

# Available proposal distributions for Metropolis
def normal_proposal(s, n):
    return lambda: normal(scale=s, size=n)

def cauchy_proposal(s, n):
    return lambda: standard_cauchy(size=n) * s

def laplace_proposal(s, n):
    return lambda: (standard_exponential(size=n) - standard_exponential(size=n)) * s


class Metropolis(ArrayStep):
    """
    Metropolis-Hastings sampling step

    Parameters
    ----------
    vars : list
        List of variables for sampler
    S : standard deviation or covariance matrix
        Some measure of variance to parameterize proposal distribution
    proposal_dist : function
        Function that returns zero-mean deviates when parameterized with
        S (and n). Defaults to quad_potential.
    scaling : scalar or array
        Initial scale factor for proposal. Defaults to 1.
    tune : bool
        Flag for tuning. Defaults to True.
    model : PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).

    """
    def __init__(self, vars, S, proposal_dist=quad_potential, scaling=1.,
            tune=True, model=None):

        model = modelcontext(model)

        try:
            self.proposal_dist = proposal_dist(S, n=len(vars))
        except TypeError:
            # quadpotential does not require n
            self.proposal_dist = proposal_dist(S)
        self.scaling = scaling
        self.tune = tune
        super(Metropolis,self).__init__(vars, [model.logpc])

    def astep(self, q0, logp):

        delta = self.proposal_dist() * self.scaling

        q = q0 + delta

        return metrop_select(logp(q) - logp(q0), q, q0)
