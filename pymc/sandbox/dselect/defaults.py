"""Default parameters for pymc distributions."""
import pymc
import inspect

pymc_default_parameters = {
    'arlognormal': dict(a=1., sigma=1., rho=0., beta=1.),
    'bernoulli': dict(p=.5),
    'beta': dict(alpha=1., beta=1.), 
    'binomial': None, 
    'categorical': None,
    'cauchy': dict(alpha=0., beta=1.), 
    'chi2': dict(nu=1),
    'dirichlet': None, 
    'discrete_uniform': dict(lower=-10000, upper=10000),
    'exponential': dict(beta=1.), 
    'exponweib': dict(alpha=1., k=1., loc=0., scale=0.),
    'gamma': dict(alpha=.5, beta=.5), 
    'geometric': dict(p=.5), 
    'gev': dict(xi=1., mu=0., sigma=1.), 
    'half_normal': dict(tau=1.), 
    'hypergeometric': dict(n=2, m=1, N=5),
    'inverse_gamma': dict(alpha=1., beta=1.), 
    'lognormal': dict(mu=0., tau=1.), 
    'multinomial': None, 
    'multivariate_hypergeometric': None, 
    'mv_normal_chol': None, 
    'mv_normal_cov': None, 
    'mv_normal_like': None,
    'negative_binomial': dict(mu=1., alpha=1.),
    'normal': dict(mu=0., tau=1.),
    'one_over_x': None,
    'poisson': dict(mu=2.), 
    'skew_normal': dict(mu=0., tau=1., alpha=1.),
    'trunc_norm': dict(mu=0., sigma=1., a=-10000, b=10000),
    'uniform': dict(lower=-10000, upper=10000),
    'uninformative': None,
    'weibull': dict(alpha=1., beta=1.),
    'wishart': None, 
    'wishart_cov': None}


def pymc_default_list(dist):
    """Return the list of default values for distribution."""
    f = getattr(pymc, '%s_like'%dist)
    vars = inspect.getargs(f.func_code)[0][1:] 
    return [pymc_default_parameters[dist][v] for v in vars]
    

scipy_default_parameters={}
