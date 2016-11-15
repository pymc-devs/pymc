import theano.tensor as tt
import pandas as pd
import numpy as np
from ..distributions import Normal, Flat
from .base import UserModel


class LinearComponent(UserModel):
    """Creates linear component, y_est is accessible via attribute
    Parameters
    ----------
    name : name, associated with the component
    x : pd.DataFrame or np.ndarray
    y : pd.Series or np.array
    intercept : bool - fit with intercept or not?
    labels : replace variable names with these labels
    priors : priors for coefficients
    init : test_vals for coefficients
    """
    def __init__(self, name, x, y, intercept=True, labels=None, priors=None, init=None):
        super(LinearComponent, self).__init__(name)
        if priors is None:
            priors = {}
        if init is None:
            init = {}
        if isinstance(x, pd.DataFrame):
            self._init_pandas_(x, y, intercept, labels, priors, init)
        if isinstance(x, np.ndarray):
            self._init_numpy_(x, y, intercept, labels, priors, init)

    def _init_pandas_(self, x, y, intercept=True, labels=None, priors=None, init=None):
        if labels is None:
            labels = list(x.columns)
        else:
            assert len(labels) == x.shape[1], \
                'Cannot deal with not full list of labels, need {}'.format(x.shape[1])
        if intercept:
            x = pd.concat([pd.Series(np.ones(x.shape[0])), x], 1)
            self.intercept = self.new_var(
                name='Intercept',
                dist=priors.get('Intercept', Flat.dist()),
                test_val=init.get('Intercept')
            )
        else:
            self.intercept = 0
        for name in labels:
            self.new_var(
                name=name,
                dist=priors.get(name, Normal.dist(mu=0, tau=1.0E-6)),
                test_val=init.get(name)
            )
            if name == 'Intercept':
                self.intercept = self['Intercept']
        self.coeffs = tt.stack(self.vars.values(), axis=0)
        self.y_est = tt.dot(np.asarray(x), self.coeffs).reshape((1, -1))

    def _init_numpy_(self, x, y, intercept=True, labels=None, priors=None, init=None):
        x = pd.DataFrame(x).rename(columns=lambda i: "x%d" % i)
        self._init_pandas_(x, y, intercept, labels, priors, init)

    @classmethod
    def from_formula(cls, name, formula, data, priors=None, init=None):
        import patsy
        y, x = patsy.dmatrices(formula, data)
        labels = x.design_info.column_names
        return cls(name, x, y, intercept=False, labels=labels, priors=priors, init=init)
