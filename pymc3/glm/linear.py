import theano.tensor as tt
import numpy as np
from ..distributions import Normal, Flat
from . import families
from ..model import Model, Deterministic
from .utils import any_to_tensor_and_labels


__all__ = [
    'LinearComponent',
    'GLM'
]


class LinearComponent(Model):
    """Creates linear component, y_est is accessible via attribute

    Parameters
    ----------
    name : str - name, associated with the linear component
    x : pd.DataFrame or np.ndarray
    y : pd.Series or np.array
    intercept : bool - fit with intercept or not?
    labels : list - replace variable names with these labels
    priors : dict - priors for coefficients
        use `Intercept` key for defining Intercept prior
            defaults to Flat.dist()
        use `Regressor` key for defining default prior for all regressors
            defaults to Normal.dist(mu=0, tau=1.0E-6)
    vars : dict - random variables instead of creating new ones
    offset : scalar, or numpy/theano array with the same shape as y
        this can be used to specify an a priori known component to be
        included in the linear predictor during fitting.
    """
    default_regressor_prior = Normal.dist(mu=0, tau=1.0E-6)
    default_intercept_prior = Flat.dist()

    def __init__(self, x, y, intercept=True, labels=None,
                 priors=None, vars=None, name='', model=None, offset=0.):
        super().__init__(name, model)
        if priors is None:
            priors = {}
        if vars is None:
            vars = {}
        x, labels = any_to_tensor_and_labels(x, labels)
        # now we have x, shape and labels
        if intercept:
            x = tt.concatenate(
                [tt.ones((x.shape[0], 1), x.dtype), x],
                axis=1
            )
            labels = ['Intercept'] + labels
        coeffs = list()
        for name in labels:
            if name == 'Intercept':
                if name in vars:
                    v = Deterministic(name, vars[name])
                else:
                    v = self.Var(
                        name=name,
                        dist=priors.get(
                            name,
                            self.default_intercept_prior
                        )
                    )
                coeffs.append(v)
            else:
                if name in vars:
                    v = Deterministic(name, vars[name])
                else:
                    v = self.Var(
                        name=name,
                        dist=priors.get(
                            name,
                            priors.get(
                                'Regressor',
                                self.default_regressor_prior
                            )
                        )
                    )
                coeffs.append(v)
        self.coeffs = tt.stack(coeffs, axis=0)
        self.y_est = x.dot(self.coeffs) + offset

    @classmethod
    def from_formula(cls, formula, data, priors=None, vars=None,
                     name='', model=None, offset=0.):
        import patsy
        y, x = patsy.dmatrices(formula, data)
        labels = x.design_info.column_names
        return cls(np.asarray(x), np.asarray(y)[:, -1], intercept=False,
                   labels=labels, priors=priors, vars=vars, name=name,
                   model=model, offset=offset)


class GLM(LinearComponent):
    """Creates glm model, y_est is accessible via attribute

    Parameters
    ----------
    name : str - name, associated with the linear component
    x : pd.DataFrame or np.ndarray
    y : pd.Series or np.array
    intercept : bool - fit with intercept or not?
    labels : list - replace variable names with these labels
    priors : dict - priors for coefficients
        use `Intercept` key for defining Intercept prior
            defaults to Flat.dist()
        use `Regressor` key for defining default prior for all regressors
            defaults to Normal.dist(mu=0, tau=1.0E-6)
    init : dict - test_vals for coefficients
    vars : dict - random variables instead of creating new ones
    family : pymc3..families object
    offset : scalar, or numpy/theano array with the same shape as y
        this can be used to specify an a priori known component to be
        included in the linear predictor during fitting.
    """
    def __init__(self, x, y, intercept=True, labels=None,
                 priors=None, vars=None, family='normal', name='',
                 model=None, offset=0.):
        super().__init__(
            x, y, intercept=intercept, labels=labels,
            priors=priors, vars=vars, name=name,
            model=model, offset=offset
        )

        _families = dict(
            normal=families.Normal,
            student=families.StudentT,
            binomial=families.Binomial,
            poisson=families.Poisson,
            negative_binomial=families.NegativeBinomial,
        )
        if isinstance(family, str):
            family = _families[family]()
        self.y_est = family.create_likelihood(
            name='', y_est=self.y_est,
            y_data=y, model=self)

    @classmethod
    def from_formula(cls, formula, data, priors=None,
                     vars=None, family='normal', name='',
                     model=None, offset=0.):
        import patsy
        y, x = patsy.dmatrices(formula, data)
        labels = x.design_info.column_names
        return cls(np.asarray(x), np.asarray(y)[:, -1], intercept=False,
                   labels=labels, priors=priors, vars=vars, family=family,
                   name=name, model=model, offset=offset)


glm = GLM
