import numpy as np
from ..distributions import Normal
from ..model import modelcontext
import patsy
import pandas as pd
import theano
from collections import defaultdict, namedtuple

from . import families

__all__ = ['glm', 'linear_component', 'plot_posterior_predictive']


def _xy_to_data_and_formula(X, y):
    if not isinstance(y, pd.Series):
        y = pd.Series(y, name='y')
    else:
        if not y.name:
            y.name = 'y'
    if not isinstance(X, (pd.DataFrame, pd.Series)):
        if len(X.shape) > 1:
            cols = ['x%d' % i for i in range(X.shape[1])]
        else:
            cols = ['x']
        X = pd.DataFrame(X, columns=cols)
    elif isinstance(X, pd.Series):
        if not X.name:
            X.name = 'x'
    # else -> pd.DataFrame -> ok
    data = pd.concat([y, X], 1)
    formula = patsy.ModelDesc(
        [patsy.Term([patsy.LookupFactor(y.name)])],
        [patsy.Term([patsy.LookupFactor(p)]) for p in X.columns]
    )
    return data, formula


class linear_component(namedtuple('Estimate', 'y_est,coeffs')):
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

    Output
    ------
    (y_est, coeffs) : Estimate for y, list of coefficients

    Example
    -------
    # Logistic regression
    y_est, coeffs = linear_component('male ~ height + weight',
                        htwt_data)
    probability = glm.families.logit(y_est)
    y_data = Bernoulli('y', probability, observed=data.male)
    """
    __slots__ = ()

    def __new__(cls, formula, data, priors=None,
                     intercept_prior=None,
                     regressor_prior=None,
                     init_vals=None,
                     model=None,
                     name=''):
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
        if name:
            name = '{}_'.format(name)
        if reg_names[0] == 'Intercept':
            prior = priors.get('Intercept', intercept_prior)
            coeff = model.Var('{}{}'.format(name, reg_names.pop(0)), prior)
            if 'Intercept' in init_vals:
                coeff.tag.test_value = init_vals['Intercept']
            coeffs.append(coeff)

        for reg_name in reg_names:
            prior = priors.get(reg_name, regressor_prior)
            coeff = model.Var('{}{}'.format(name, reg_name), prior)
            if reg_name in init_vals:
                coeff.tag.test_value = init_vals[reg_name]
            coeffs.append(coeff)

        y_est = theano.dot(np.asarray(dmatrix),
                           theano.tensor.stack(*coeffs)).reshape((1, -1))

        return super(linear_component, cls).__new__(cls, y_est, coeffs)

    @classmethod
    def from_xy(cls, X, y,
                priors=None,
                intercept_prior=None,
                regressor_prior=None,
                init_vals=None,
                model=None,
                name=''):
        data, formula = _xy_to_data_and_formula(X, y)
        return cls(formula, data,
                   priors=priors,
                   intercept_prior=intercept_prior,
                   regressor_prior=regressor_prior,
                   init_vals=init_vals,
                   model=model,
                   name=name
                   )


class glm(namedtuple('Estimate', 'y_est,coeffs')):
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
    (y_est, coeffs) : Estimate for y, list of coefficients

    Example
    -------
    # Logistic regression
    vars = glm('male ~ height + weight',
               data,
               family=glm.families.Binomial())
    """
    __slots__ = ()

    def __new__(cls, formula, data, priors=None,
            intercept_prior=None,
            regressor_prior=None,
            init_vals=None,
            family='normal',
            model=None,
            name=''):
        _families = dict(
            normal=families.Normal,
            student=families.StudentT,
            binomial=families.Binomial,
            poisson=families.Poisson
        )
        if isinstance(family, str):
            family = _families[family]()

        y_data = np.asarray(patsy.dmatrices(formula, data)[0]).T

        y_est, coeffs = linear_component(
            formula, data, priors=priors,
            intercept_prior=intercept_prior,
            regressor_prior=regressor_prior,
            init_vals=init_vals,
            model=model,
            name=name
            )
        family.create_likelihood(name, y_est, y_data, model=model)

        return super(glm, cls).__new__(cls, y_est, coeffs)

    @classmethod
    def from_xy(cls, X, y,
                priors=None,
                intercept_prior=None,
                regressor_prior=None,
                init_vals=None,
                family='normal',
                model=None,
                name=''):
        data, formula = _xy_to_data_and_formula(X, y)
        return cls(formula, data,
                   priors=priors,
                   intercept_prior=intercept_prior,
                   regressor_prior=regressor_prior,
                   init_vals=init_vals,
                   model=model,
                   family=family,
                   name=name
                   )


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
