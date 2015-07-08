from __future__ import division
import unittest
from pymc3.tests.checks import *
from pymc3 import Model, Point
from pymc3.distributions.continuous import *
import numpy as np
import scipy.special as sp
import scipy.stats as st

from pymc3.tests.knownfailure import *
from pymc3.tests.test_distributions import (build_model, product,
                                 R, Rplus, RNZplus, Rplusbig, Rplusdunif)

def pymc3_pdf_matches_scipy(dist, values, paradomains, reference_pdf):
    model = build_model(dist, values, paradomains)
    for domain in product(paradomains):
        params = {k:v for k, v in domain}
        vals = np.array(values.vals)
        pymc3_pdf = model.named_vars['value'].pdf(vals, point=Point(params, model=model))
        scipy_pdf = reference_pdf(vals, **params)
        assert_almost_equal(pymc3_pdf, scipy_pdf, 6, err_msg=str(params))
        
def test_uniform_pdf():  
    pymc3_pdf_matches_scipy(Uniform, R, {'lower': -RNZplus, 'upper': RNZplus},
                      lambda value, lower=None, upper=None: st.uniform.pdf(value, loc=lower, scale=upper - lower))
    
def test_normal_pdf():
    pymc3_pdf_matches_scipy(Normal, R, {'mu': R, 'tau': RNZplus},
                      lambda value, mu=0., tau=1.: st.norm.pdf(value, loc=mu, scale=np.sqrt(1. / tau)))
  
def test_halfnormal_pdf():
    pymc3_pdf_matches_scipy(HalfNormal, R, {'tau': RNZplus},
                    lambda value, tau=None: st.halfnorm.pdf(value, scale=np.sqrt(1. / tau)))
  
def test_beta_pdf():
    pymc3_pdf_matches_scipy(Beta, R, {'alpha': RNZplus, 'beta':RNZplus},
                    lambda value, alpha=None, beta=None: st.beta.pdf(value, a=alpha, b=beta))
    
def test_exponential_pdf():
    pymc3_pdf_matches_scipy(Exponential, R, {'lam': RNZplus},
                    lambda value, lam=None: st.expon.pdf(value, scale=1./lam))
    
def test_laplace_pdf():
    pymc3_pdf_matches_scipy(Laplace, R, {'mu':R, 'b': RNZplus},
                    lambda value, mu=None, b=None: st.laplace.pdf(value, loc=mu, scale=b))
    
def test_lognormal_pdf():
    pymc3_pdf_matches_scipy(Lognormal, R, {'mu':R, 'tau': RNZplus},
                    lambda value, mu=None, tau=None: st.lognorm.pdf(value, s=tau**-.5, loc=0, scale=np.exp(mu)))
    

def test_t_pdf():
    pymc3_pdf_matches_scipy(
            T, R, {'nu': Rplus, 'mu': R, 'lam': RNZplus},
            lambda value, nu=None, mu=None, lam=None: st.t.pdf(value, nu, mu, lam**-.5)
            )

def test_cauchy_pdf():
    pymc3_pdf_matches_scipy(
            Cauchy, R, {'alpha': R, 'beta': Rplusbig},
            lambda value, alpha=None, beta=None: st.cauchy.pdf(value, loc=alpha, scale=beta)
            )

def test_half_cauchy_pdf():
    pymc3_pdf_matches_scipy(
            HalfCauchy, Rplus, {'beta': Rplusbig},
            lambda value, beta=None: st.halfcauchy.pdf(value, scale=beta)
            )

def test_gamma_pdf():
    pymc3_pdf_matches_scipy(
            Gamma, Rplus, {'alpha': Rplusbig, 'beta': Rplusbig},
            lambda value, alpha=None, beta=None: st.gamma.pdf(value, a=alpha, scale=1./beta)
            )
    pymc3_pdf_matches_scipy(
            Gamma, Rplus, {'mu': Rplusbig, 'sd': Rplusbig},
            lambda value, mu=None, sd=None: st.gamma.pdf(value, a=(mu / sd) **2 , scale=(sd ** 2) / mu)
            )


def test_pareto_pdf():
    pymc3_pdf_matches_scipy(
            Pareto, Rplus, {'alpha': Rplusbig, 'm': Rplusbig},
            lambda value, alpha=None, m=None: st.pareto.pdf(value, alpha, scale=m)
            )

def test_inverse_gamma_pdf():
    pymc3_pdf_matches_scipy(
            InverseGamma, Rplus, {'alpha': Rplus, 'beta': Rplus},
            lambda value, alpha=None, beta=None: st.invgamma.pdf(value, alpha, scale=beta)
            )
        
def test_chi_squared_pdf():
    pymc3_pdf_matches_scipy(
            ChiSquared, Rplus, {'nu': Rplusdunif},
            lambda value, nu=None: st.chi2.pdf(value, df=nu)
            )
    

def scipy_weibull_pdf_sucks(x, alpha, beta):
    # scipy.stats.weibull doesn't have a shape parameter.
    return alpha * (x ** (alpha - 1)) * (beta ** -alpha) * \
                       np.exp(-(x / beta) ** alpha)
                       
def test_weibull_pdf():
    pymc3_pdf_matches_scipy(
            Weibull, Rplus, {'alpha': Rplusbig, 'beta': Rplusbig},
            lambda value, alpha=None, beta=None: scipy_weibull_pdf_sucks(value, alpha, beta)
            )

def test_ex_gaussian_pdf():
    # Probabilities calculated using the dexGAUS function from the R package gamlss.
    # See e.g., doi: 10.1111/j.1467-9876.2005.00510.x, or http://www.gamlss.org/.
    test_cases = [
        (-50.0, -50.000, 0.500, 0.500, 0.5231566),
        (1.0, -1.000, 0.001, 0.001, 0.0000000),
        (2.0, 0.001, 1.000, 1.000, 0.1878631),
        (5.0, 0.500, 2.500, 2.500, 0.0859178),
        (7.5, 2.000, 5.000, 5.000, 0.0592528),
        (15.0, 5.000, 7.500, 7.500, 0.0365386),
        (50.0, 50.000, 10.000, 10.000, 0.0261578),
        (600.0, 500.000, 10.000, 20.000, 0.0003818)
        ]
    for value, mu, sigma, nu, p in test_cases:
        yield check_ex_gaussian_pdf, value, mu, sigma, nu, p
             
def check_ex_gaussian_pdf(value, mu, sigma, nu, p):
    with Model():
        ig = ExGaussian('eg', mu=mu, sigma=sigma, nu=nu)
    pt = {'eg': value}
    assert_almost_equal(ig.pdf(value),
                p, decimal=6, err_msg=str(pt))


def test_wald_pdf():
    # Probabilities calculated using the dIG function from the R package gamlss.
    # See e.g., doi: 10.1111/j.1467-9876.2005.00510.x, or http://www.gamlss.org/.
    test_cases = [# 
        (0.5, 0.001, 0.500, None, 0.000, 0.0000000),
        (1.0, 0.500, 0.001, None, 0.500, 0.0356825),
        (2.0, 1.000, 1.000, None, 1.000, 0.3989423),
        (5.0, 2.000, 2.500, None, 2.000, 0.1093854),
        (7.5, 5.000, 5.000, None, 2.500, 0.0797885),
        (15.0, 10.000, 7.500, None, 5.000, 0.0345494),
        (50.0, 15.000, 10.000, None, 10.000, 0.0035239),
        (0.5, 0.001, None, 500, 0.000, 0.0000000),
        (1.0, 0.500, None, 0.002, 0.500, 0.0356825),
        (2.0, 1.000, None, 1.000, 1.000, 0.3989423),
        (5.0, 2.000, None, 1.250, 2.000, 0.1093854),
        (7.5, 5.000, None, 1.000, 2.500, 0.0797885),
        (15.0, 10.000, None, 0.750, 5.000, 0.0345494),
        (50.0, 15.000, None, 0.666, 10.000, 0.0035239),
        ]
    for value, mu, lam, phi, alpha, p in test_cases:
        yield check_wald_pdf, value, mu, lam, phi, alpha, p
             
def check_wald_pdf(value, mu, lam, phi, alpha, p):   
    with Model():
        wald = Wald('wald', mu=mu, lam=lam, phi=phi, alpha=alpha, transform=None)
    pt = {'wald': value}
    assert_almost_equal(wald.pdf(value),
                    p, decimal=6, err_msg=str(pt)) 

def test_gev_pdf():
    # Probabilities calculated using the dgev function from the R package 
    # "fExtremes: Rmetrics - Extreme Financial Market Data".
    # See http://cran.r-project.org/web/packages/fExtremes/index.html
    test_cases = [
    ([-5,-2.5,-1,0,1,2.5,5], -5.000, 0.010, -1.500, 
        [36.7879441,0.0000000,0.0000000,0.0000000,0.0000000,0.0000000,0.0000000]),
    ([-5,-2.5,-1,0,1,2.5,5], -2.500, 0.100, -1.000, 
        [0.0000000,3.6787944,0.0000000,0.0000000,0.0000000,0.0000000,0.0000000]),
    ([-5,-2.5,-1,0,1,2.5,5], -1.000, 1.000, -0.500, 
        [0.0003702,0.0818486,0.3678794,0.3894004,0.0000000,0.0000000,0.0000000]),
    ([-5,-2.5,-1,0,1,2.5,5], 0.000, 2.000, 0.000, 
        [0.0000312,0.0532110,0.1585210,0.1839397,0.1653521,0.1075659,0.0378081]),
    ([-5,-2.5,-1,0,1,2.5,5], 2.500, 2.500, 0.000, 
        [0.0000000,0.0018265,0.0281139,0.0717496,0.1178421,0.1471518,0.1018586]),
    ([-5,-2.5,-1,0,1,2.5,5], 1.000, 5.000, 0.500, 
        [0.0060327,0.0682927,0.0818794,0.0798245,0.0735759,0.0617373,0.0437590]),
    ([-5,-2.5,-1,0,1,2.5,5], 5.000, 7.500, 1.000, 
        [0.0000000,0.0000000,0.0224598,0.0597445,0.0718281,0.0669390,0.0490506]),
    ([-5,-2.5,-1,0,1,2.5,5], 50.000, 10.000, 1.500, 
        [0.0000000,0.0000000,0.0000000,0.0000000,0.0000000,0.0000000,0.0000000])
                ]
    with Model():
        gev = GeneralizedExtremeValue('gev', 
                                      mu=Flat('mu', testval=0.), 
                                      sigma=Flat('sigma', testval=1.),
                                      xi=Flat('xi', testval=0.), transform=None)

        for value, mu, sigma, xi, p in test_cases:
            yield check_gev_pdf, gev, value, mu, sigma, xi, p         

def check_gev_pdf(gev, value, mu, sigma, xi, p):   
    pt = {'mu':mu, 'sigma':sigma, 'xi':xi}
    assert_almost_equal(gev.pdf(np.array(value), point=pt),
                    p, decimal=6, err_msg=str(pt)) 
