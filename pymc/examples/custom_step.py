import pymc as pm
from pymc import six
import numpy as np


class TruncatedMetropolis(pm.Metropolis):

    def __init__(self, stochastic, low_bound, up_bound, *args, **kwargs):
        self.low_bound = low_bound
        self.up_bound = up_bound
        pm.Metropolis.__init__(self, stochastic, *args, **kwargs)

    # Propose method written by hacking Metropolis.propose()
    def propose(self):
        tau = 1. / (self.adaptive_scale_factor * self.proposal_sd) ** 2
        self.stochastic.value = pm.rtruncnorm(
            self.stochastic.value,
            tau,
            self.low_bound,
            self.up_bound)

    # Hastings factor method accounts for asymmetric proposal distribution
    def hastings_factor(self):
        tau = 1. / (self.adaptive_scale_factor * self.proposal_sd) ** 2
        cur_val = self.stochastic.value
        last_val = self.stochastic.last_value

        lp_for = pm.truncnorm_like(
            cur_val,
            last_val,
            tau,
            self.low_bound,
            self.up_bound)
        lp_bak = pm.truncnorm_like(
            last_val,
            cur_val,
            tau,
            self.low_bound,
            self.up_bound)

        if self.verbose > 1:
            six.print_(self._id + ': Hastings factor %f' % (lp_bak - lp_for))
        return lp_bak - lp_for


# Maximum data value is around 1 (plot the histogram).
data = np.array(
    [-1.6464815, -0.86463278, 0.80656378, 0.67664181, -0.34312965,
     -0.29654303, 0.76170081, 0.30418603, -0.45473639, -0.24894277,
     -0.07173209, -1.64602289, 0.0804062, -0.82159472, 0.98224623,
     -1.92538425, -1.95388748, -0.41145515, 0.23972844, -0.78645389,
     -0.21687104, -0.2939634, 0.51229013, 0.04626286, 0.18329919,
     -1.12775839, -1.64187249, 0.33440094, -0.95224695, 0.15650266,
     -0.54056102, 0.12240128, -0.95397459, 0.44806432, -1.02955556,
        0.31740861, -0.8762523, 0.47377688, 0.76516415, 0.27890419,
     -0.07819642, -0.13399348, 0.82877293, 0.22308624, 0.7485783,
     -0.14700254, -1.03145657, 0.85641097, 0.43396285, 0.47901653,
        0.80137086, 0.33566812, 0.71443253, -1.57590815, -0.24090179,
     -2.0128344, 0.34503324, 0.12944091, -1.5327008, 0.06363034,
        0.21042021, -0.81425636, 0.20209279, -1.48130423, -1.04983523,
        0.16001774, -0.75239072, 0.33427956, -0.10224921, 0.26463561,
     -1.09374674, -0.72749811, -0.54892116, -1.89631844, -0.94393545,
     -0.2521341, 0.26840341, 0.23563219, 0.35333094])

# Model: the data are truncated-normally distributed with unknown upper bound.
mu = pm.Normal('mu', 0, .01, value=0)
tau = pm.Exponential('tau', .01, value=1)
cutoff = pm.Exponential('cutoff', 1, value=1.3)
D = pm.TruncatedNormal(
    'D',
    mu,
    tau,
    -np.inf,
    cutoff,
    value=data,
    observed=True)

M = pm.MCMC([mu, tau, cutoff, D])

# Use a TruncatedMetropolis step method that will never propose jumps
# below D's maximum value.
M.use_step_method(TruncatedMetropolis, cutoff, D.value.max(), np.inf)
# Get a handle to the step method handling cutoff to investigate its behavior.
S = M.step_method_dict[cutoff][0]

M.isample(10000, 0, 10)
