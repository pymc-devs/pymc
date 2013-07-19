import matplotlib.pyplot as plt
import numpy as np
from pymc import  *
import patsy
import theano
import pandas as pd
from collections import defaultdict
from statsmodels.formula.api import glm as glm_sm
import statsmodels.api as sm
from pandas.tools.plotting import scatter_matrix

import links
import families

def linear_component(formula, data, priors=None,
                     intercept_prior=None,
                     regressor_prior=None,
                     init=True, init_vals=None, family=None,
                     model=None):
    """Create linear model according to patsy specification.

    Parameters
    ----------
    formula : str
        Patsy linear model descriptor.
    data : array
        Labeled array (e.g. pandas DataFrame, recarray).
    priors : dict
        Mapping prior name to prior distribution.
        E.g. {'Intercept': Normal.dist(mu=0, sd=1)}
    intercept_prior : pymc distribution
        Prior to use for the intercept.
    	Default: Normal.dist(mu=0, tau=1.0E-12)
    regressor_prior : pymc distribution
        Prior to use for all regressor(s).
	    Default: Normal.dist(mu=0, tau=1.0E-12)
    init : bool
        Whether to set the starting values via statsmodels
        Default: True
    init_vals : dict
        Set starting values externally: parameter -> value
        Default: None
    family : statsmodels.family
        Link function to pass to statsmodels (init has to be True).
	See `statsmodels.api.families`
        Default: identity

    Output
    ------
    (y_est, coeffs) : Estimate for y, list of coefficients

    Example
    -------
    # Logistic regression
    y_est, coeffs = glm('male ~ height + weight',
                        htwt_data,
                        family=glm.families.Binomial(links=glm.link.Logit))
    y_data = Bernoulli('y', y_est, observed=data.male)
    """
    if intercept_prior is None:
        intercept_prior = Normal.dist(mu=0, tau=1.0E-12)
    if regressor_prior is None:
        regressor_prior = Normal.dist(mu=0, tau=1.0E-12)

    if priors is None:
        priors = defaultdict(None)

    # Build patsy design matrix and get regressor names.
    _, dmatrix = patsy.dmatrices(formula, data)
    reg_names = dmatrix.design_info.column_names

    if init_vals is None and init:
        try:
            from statsmodels.formula.api import glm as glm_sm
            init_vals = glm_sm(formula, data, family=family).fit().params
        except ImportError:
            raise ImportError("Can't initialize -- statsmodels not found. Set init=False and run find_MAP().")
    else:
        init_vals = defaultdict(lambda: None)

    # Create individual coefficients
    model = modelcontext(model)
    coeffs = []

    if reg_names[0] == 'Intercept':
        prior = priors.get('Intercept', intercept_prior)
        coeff = model.Var(reg_names.pop(0), prior)
        coeff.tag.test_value = init_vals['Intercept']
        coeffs.append(coeff)

    for reg_name in reg_names:
        prior = priors.get(reg_name, regressor_prior)
        coeff = model.Var(reg_name, prior)
    	coeff.tag.test_value = init_vals[reg_name]
        coeffs.append(coeff)

    y_est = theano.dot(np.asarray(dmatrix), theano.tensor.stack(*coeffs)).reshape((1, -1))

    return y_est, coeffs


def glm(*args, **kwargs):
    """Create GLM after Patsy model specification string.

    Parameters
    ----------
    formula : str
        Patsy linear model descriptor.
    data : array
        Labeled array (e.g. pandas DataFrame, recarray).
    priors : dict
        Mapping prior name to prior distribution.
        E.g. {'Intercept': Normal.dist(mu=0, sd=1)}
    intercept_prior : pymc distribution
        Prior to use for the intercept.
    	Default: Normal.dist(mu=0, tau=1.0E-12)
    regressor_prior : pymc distribution
        Prior to use for all regressor(s).
	    Default: Normal.dist(mu=0, tau=1.0E-12)
    init : bool
        Whether initialize test values via statsmodels
        Default: True
    init_vals : dict
        Set starting values externally: parameter -> value
        Default: None
    family : statsmodels.family
        Distribution of likelihood, see pymc.glm.families
        (init has to be True).

    Output
    ------
    vars : List of created random variables (y_est, coefficients etc)

    Example
    -------
    # Logistic regression
    vars = glm('male ~ height + weight',
               data,
               family=glm.families.Binomial(link=glm.links.Logit))
    """

    model = modelcontext(kwargs.get('model'))

    family = kwargs.pop('family', families.Normal())

    formula = args[0]
    data = args[1]
    y_data = np.asarray(patsy.dmatrices(formula, data)[0]).T

    # Create GLM
    kwargs['family'] = family.create_statsmodel_family()

    y_est, coeffs = linear_component(*args, **kwargs)
    family.create_likelihood(y_est, y_data)

    # Find vars we have not initialized yet
    non_init_vars = set(model.vars).difference(set(coeffs))
    start = find_MAP(vars=non_init_vars)
    for var in non_init_vars:
        var.tag.test_value = start[var.name]

    return [y_est] + coeffs + list(non_init_vars)


def plot_posterior_predictive(trace):
    for rand_loc in np.random.randint(0, len(trace), 30):
        rand_sample = trace[rand_loc]
        plt.plot([0, 1], [rand_sample['Intercept'],
                          rand_sample['Intercept'] + rand_sample['x']], lw=.2, c='k')
    plt.title('Posterior predictive')
