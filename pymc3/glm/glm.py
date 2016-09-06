import numpy as np
from ..distributions import Normal
from ..tuning.starting import find_MAP
from ..model import modelcontext
import patsy
import theano
import pandas as pd
from collections import defaultdict
from pandas.tools.plotting import scatter_matrix

from . import families

__all__ = ['glm', 'linear_component', 'plot_posterior_predictive']


def linear_component(formula, data, priors=None,
                     intercept_prior=None,
                     regressor_prior=None,
                     init_vals=None, family=None,
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
    intercept_prior : pymc3 distribution
        Prior to use for the intercept.
        Default: Normal.dist(mu=0, tau=1.0E-12)
    regressor_prior : pymc3 distribution
        Prior to use for all regressor(s).
        Default: Normal.dist(mu=0, tau=1.0E-12)
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
                        family=glm.families.Binomial(link=glm.family.logit))
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

    if init_vals is None:
        init_vals = {}

    # Create individual coefficients
    model = modelcontext(model)
    coeffs = []

    if reg_names[0] == 'Intercept':
        prior = priors.get('Intercept', intercept_prior)
        coeff = model.Var(reg_names.pop(0), prior)
        if 'Intercept' in init_vals:
            coeff.tag.test_value = init_vals['Intercept']
        coeffs.append(coeff)

    for reg_name in reg_names:
        prior = priors.get(reg_name, regressor_prior)
        coeff = model.Var(reg_name, prior)
        if reg_name in init_vals:
            coeff.tag.test_value = init_vals[reg_name]
        coeffs.append(coeff)

    y_est = theano.dot(np.asarray(dmatrix),
                       theano.tensor.stack(*coeffs)).reshape((1, -1))

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
    intercept_prior : pymc3 distribution
        Prior to use for the intercept.
        Default: Normal.dist(mu=0, tau=1.0E-12)
    regressor_prior : pymc3 distribution
        Prior to use for all regressor(s).
        Default: Normal.dist(mu=0, tau=1.0E-12)
    init_vals : dict
        Set starting values externally: parameter -> value
        Default: None
    family : Family object
        Distribution of likelihood, see pymc3.glm.families
        (init has to be True).

    Output
    ------
    vars : List of created random variables (y_est, coefficients etc)

    Example
    -------
    # Logistic regression
    vars = glm('male ~ height + weight',
               data,
               family=glm.families.Binomial(link=glm.families.logit))
    """

    model = modelcontext(kwargs.get('model'))

    family = kwargs.pop('family', families.Normal())

    call_find_map = kwargs.pop('find_MAP', True)
    formula = args[0]
    data = args[1]
    y_data = np.asarray(patsy.dmatrices(formula, data)[0]).T

    y_est, coeffs = linear_component(*args, **kwargs)
    family.create_likelihood(y_est, y_data)

    return [y_est] + coeffs


def plot_posterior_predictive(trace, eval=None, lm=None, samples=30, **kwargs):
    """Plot posterior predictive of a linear model.

    :Arguments:
        trace : <array>
            Array of posterior samples with columns
        eval : <array>
            Array over which to evaluate lm
        lm : function <default: linear function>
            Function mapping parameters at different points
            to their respective outputs.
            input: point, sample
            output: estimated value
        samples : int <default=30>
            How many posterior samples to draw.

    Additional keyword arguments are passed to pylab.plot().

    """
    import matplotlib.pyplot as plt

    if lm is None:
        lm = lambda x, sample: sample['Intercept'] + sample['x'] * x

    if eval is None:
        eval = np.linspace(0, 1, 100)

    # Set default plotting arguments
    if 'lw' not in kwargs and 'linewidth' not in kwargs:
        kwargs['lw'] = .2
    if 'c' not in kwargs and 'color' not in kwargs:
        kwargs['c'] = 'k'

    for rand_loc in np.random.randint(0, len(trace), samples):
        rand_sample = trace[rand_loc]
        plt.plot(eval, lm(eval, rand_sample), **kwargs)
        # Make sure to not plot label multiple times
        kwargs.pop('label', None)

    plt.title('Posterior predictive')
