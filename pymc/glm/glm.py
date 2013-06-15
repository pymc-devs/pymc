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

def linear_component(formula, data,
                     intercept_prior=None,
                     regressor_prior=None,
                     init=True, init_vals=None, family=None,
                     model=None):
    """Create GLM coefficients.

    Parameters
    ----------
    formula : str
        Patsy linear model descriptor.
    data : array
        Labeled array (e.g. pandas DataFrame, recarray).
    intercept_prior : pymc distribution
        Prior to use for the intercept.
    	Default: Normal.dist(mu=0, tau=1.0E-12)
    regressor_prior : pymc distribution
        Prior to use for the regressor(s).
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
                        htwt_data, family=sm.families.Binomial(),
                        link_func=theano.tensor.nnet.sigmoid)
    y_data = Bernoulli('y', y_est, observed=data.male)
    """
    if intercept_prior is None:
        intercept_prior = Normal.dist(mu=0, tau=1.0E-12)
    if regressor_prior is None:
        regressor_prior = Normal.dist(mu=0, tau=1.0E-12)

    # Build patsy design matrix and get regressor names.
    _, dmatrix = patsy.dmatrices(formula, data)
    reg_names = dmatrix.design_info.column_names

    if init_vals is None and init:
        try:
            from statsmodels.formula.api import glm as glm_sm
	except ImportError:
	    raise ImportError("Can't initialize -- statsmodels not found. Set init=False and run find_MAP().")
        init_vals = glm_sm(formula, data, family=family).fit().params
    else:
        init_vals = defaultdict(lambda: None)

    # Create individual coefficients
    model = modelcontext(model)
    coeffs = []

    if reg_names[0] == 'Intercept':
        intercept_prior.testval = init_vals['Intercept']
        coeff = model.Var(reg_names.pop(0), intercept_prior)
        coeffs.append(coeff)

    for reg_name in reg_names:
    	regressor_prior.testval = init_vals[reg_name]
        coeff = model.Var(reg_name, regressor_prior)
        coeffs.append(coeff)

    y_est = theano.dot(np.asarray(dmatrix), theano.tensor.stack(*coeffs)).reshape((1, -1))

    return y_est, coeffs


def glm(*args, **kwargs):
    family = kwargs.pop('family', families.Normal())

    formula = args[0]
    data = args[1]
    y_data = np.asarray(patsy.dmatrices(formula, data)[0]).T
    model = modelcontext(kwargs.get('model'))

    kwargs['family'] = family.sm_family()
    y_est, coeffs = linear_component(*args, **kwargs)
    family.likelihood(y_est, y_data)

    non_init_vars = set(model.vars).difference(set(coeffs))
    start = find_MAP(vars=non_init_vars)

    return start
