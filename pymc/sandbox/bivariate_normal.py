"""Example of trivial Gibbs sampling from
Gelman et al., Bayesian Data Analysis.

Consider a single observation $(y_1, y_2)$ from a bivariate normal
distribution populatioan with unknown mean $\theta=(\theta_1, \theta_2)$
and known covariance matrix ${1,\rho \choose \rho, 1}$. With a uniform
prior distribution on $\theta$, the posterior distribution is
$$p(\theta_1, \theta_2|y) = \mathrm{N}\left( {y_1\choose y_2}
, {1,\rho \choose \rho, 1} \right)\]$$

p(\theta_1 \mid \theta_2, y) = \mathrm{N}(y_1 + \rho(\theta_2 -y_2), 1-\rho^2)
p(\theta_2 \mid \theta_1, y) = \mathrm{N}(y_2 + \rho(\theta_1 -y_1), 1-\rho^2)
"""
from pymc import stoch, data, rnormal, normal_like, Uniform, GibbsSampler

@stoch
def theta1(value, theta2, y, rho):
    """Conditional probability p(theta1|theta2, y, rho)"""
    def logp(value, theta2, y, rho):
        mean = y[0]+rho*(theta2-y[1])
        var = 1.-rho^2
        return normal_like(value, mean, 1./var)
    def random(theta2, y, rho):
        mean = y[0]+rho*(theta2-y[1])
        var = 1.-rho^2
        return rnormal(value, mean, 1./var)

@stoch
def theta2(value, theta1, y, rho):
    """Conditional probability p(theta2|theta1, y, rho)"""
    def logp(value, theta2, y, rho):
        mean = y[1]+rho*(theta1-y[0])
        var = 1.-rho^2
        return normal_like(value, mean, 1./var)

    def random(theta2, y, rho):
        mean = y[0]+rho*(theta2-y[1])
        var = 1.-rho^2
        return rnormal(value, mean, 1./var)


rho = Uniform('rho', rseed=True, lower=0, upper=1)

@data
def y(value=(3,6)):
    return 0

G = GibbsSampler([theta1, theta2, rho])
