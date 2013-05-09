from numpy.linalg import cholesky

from ..core import *
from quadpotential import quad_potential

from arraystep import *
from numpy.random import normal, standard_cauchy, standard_exponential, poisson, random
from numpy import round, exp, copy, where


__all__ = ['Metropolis', 'BinaryMetropolis', 'NormalProposal', 'CauchyProposal', 'LaplaceProposal', 'PoissonProposal', 'MultivariateNormalProposal']

# Available proposal distributions for Metropolis


class Proposal(object):
    def __init__(self, s):
        self.s =s

class NormalProposal(Proposal):
    def __call__(self):
        return normal(scale=self.s)


class CauchyProposal(Proposal):
    def __call__(self):
        return standard_cauchy(size=np.size(self.s)) * self.s

class LaplaceProposal(Proposal):
    def __call__(self):
        size = np.size(self.s)
        return (standard_exponential(size=size) - standard_exponential(size=size)) * self.s

class PoissonProposal(Proposal):
    def __call__(self):
        return poisson(lam=self.s, size=np.size(self.s)) - self.s


class MultivariateNormalProposal(Proposal):
    def __call__(self):
        return np.random.multivariate_normal(mean = np.zeros(self.s.shape[0]),  cov=self.s)


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
    def __init__(self, vars, S=None, proposal_dist=NormalProposal, scaling=1.,
                 tune=True, tune_interval=100, model=None):

        model = modelcontext(model)

        if S is None:
            S = np.ones(sum(v.dsize for v in vars))
        self.proposal_dist = proposal_dist(S)
        self.scaling = scaling
        self.tune = tune
        self.tune_interval = tune_interval
        self.steps_until_tune = tune_interval
        self.accepted = 0

        # Determine type of variables
        if all([v.dtype in discrete_types for v in vars]):
            self.discrete = True
        elif all([v.dtype in continuous_types for v in vars]):
            self.discrete = False
        else:
            raise ValueError('All variables in vars must be of the same dtype for Metropolis')

        super(Metropolis,self).__init__(vars, [model.logpc])

    def astep(self, q0, logp):

        if self.tune and not self.steps_until_tune:
            # Tune scaling parameter
            self.scaling = tune(
                self.scaling, self.accepted / float(self.tune_interval))
            # Reset counter
            self.steps_until_tune = self.tune_interval
            self.accepted = 0

        delta = self.proposal_dist() * self.scaling

        q = q0 + delta
        if self.discrete:
            q = round(q, 0).astype(int)

        q_new = metrop_select(logp(q) - logp(q0), q, q0)

        if (any(q_new != q0) or all(q0 == q)):
            self.accepted += 1

        self.steps_until_tune -= 1

        return q_new


def tune(scale, acc_rate):
    """
    Tunes the scaling parameter for the proposal distribution
    according to the acceptance rate over the last tune_interval:

    Rate    Variance adaptation
    ----    -------------------
    <0.001        x 0.1
    <0.05         x 0.5
    <0.2          x 0.9
    >0.5          x 1.1
    >0.75         x 2
    >0.95         x 10

    """

    # Switch statement
    if acc_rate < 0.001:
        # reduce by 90 percent
        scale *= 0.1
    elif acc_rate < 0.05:
        # reduce by 50 percent
        scale *= 0.5
    elif acc_rate < 0.2:
        # reduce by ten percent
        scale *= 0.9
    elif acc_rate > 0.95:
        # increase by factor of ten
        scale *= 10.0
    elif acc_rate > 0.75:
        # increase by double
        scale *= 2.0
    elif acc_rate > 0.5:
        # increase by ten percent
        scale *= 1.1

    return scale


class BinaryMetropolis(ArrayStep):
    """Metropolis-Hastings optimized for binary variables"""
    def __init__(self, vars, scaling=1., tune=True, tune_interval=100, model=None):

        model = modelcontext(model)

        self.scaling = scaling
        self.tune = tune
        self.tune_interval = tune_interval
        self.steps_until_tune = tune_interval
        self.accepted = 0

        if not all([v.dtype in discrete_types for v in vars]):
            raise ValueError('All variables must be Bernoulli for BinaryMetropolis')

        super(BinaryMetropolis,self).__init__(vars, [model.logpc])

    def astep(self, q0, logp):

        # Convert adaptive_scale_factor to a jump probability
        p_jump = 1.-.5**self.scaling

        rand_array = random(q0.shape)
        q = copy(q0)
        # Locations where switches occur, according to p_jump
        switch_locs = where(rand_array<p_jump)
        q[switch_locs] = True - q[switch_locs]

        q_new = metrop_select(logp(q) - logp(q0), q, q0)

        return q_new
