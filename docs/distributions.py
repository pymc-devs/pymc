__docformat__='reStructuredText'
import pymc
"""
import re
objs = dir(pymc)
likelihood_pat = re.compile(r'\w+_like')
likelihoods = []
for f in objs:
    try:
    	match = likelihood_pat.match(f)
        if match:
            statement = 'from pymc import %s'%f
            print statement
            likelihoods.append(f)
            exec(statement)

    except:
        pass
print likelihoods
"""
__all__=['arlognormal_like', 'bernoulli_like', 'beta_like', 'binomial_like', 'categorical_like', 'cauchy_like', 'chi2_like', 'dirichlet_like', 'discrete_uniform_like', 'exponential_like', 'exponweib_like', 'gamma_like', 'geometric_like', 'gev_like', 'half_normal_like', 'hypergeometric_like', 'inverse_gamma_like', 'lognormal_like', 'mod_categor_like', 'mod_multinom_like', 'multinomial_like', 'multivariate_hypergeometric_like', 'mv_normal_chol_like', 'mv_normal_cov_like', 'mv_normal_like', 'negative_binomial_like', 'normal_like', 'one_over_x_like', 'poisson_like', 'skew_normal_like', 'truncnorm_like', 'uniform_like', 'uninformative_like', 'weibull_like', 'wishart_cov_like', 'wishart_like']


from pymc import arlognormal_like
from pymc import bernoulli_like
from pymc import beta_like
from pymc import binomial_like
from pymc import categorical_like
from pymc import cauchy_like
from pymc import chi2_like
from pymc import dirichlet_like
from pymc import discrete_uniform_like
from pymc import exponential_like
from pymc import exponweib_like
from pymc import gamma_like
from pymc import geometric_like
from pymc import gev_like
from pymc import half_normal_like
from pymc import hypergeometric_like
from pymc import inverse_gamma_like
from pymc import lognormal_like
from pymc import mod_categor_like
from pymc import mod_multinom_like
from pymc import multinomial_like
from pymc import multivariate_hypergeometric_like
from pymc import mv_normal_chol_like
from pymc import mv_normal_cov_like
from pymc import mv_normal_like
from pymc import negative_binomial_like
from pymc import normal_like
from pymc import one_over_x_like
from pymc import poisson_like
from pymc import skew_normal_like
from pymc import truncnorm_like
from pymc import uniform_like
from pymc import uninformative_like
from pymc import weibull_like
from pymc import wishart_cov_like
from pymc import wishart_like
