import theano.tensor as tt
import pandas as pd
import numpy as np
from ..distributions import Normal, Flat
from ..glm import families
from .base import UserModel
from .utils import any_to_tensor_and_labels


class LinearComponent(UserModel):
    """Creates linear component, y_est is accessible via attribute
    Parameters
    ----------
    name : str - name, associated with the linear component
    x : pd.DataFrame or np.ndarray
    y : pd.Series or np.array
    intercept : bool - fit with intercept or not?
    labels : list - replace variable names with these labels
    priors : dict - priors for coefficients
    init : dict - test_vals for coefficients
    rvars : dict - random variables instead of creating new ones
    """
    def __init__(self, x, y, intercept=True, labels=None,
                 priors=None, init=None, rvars=None, name=''):
        super(LinearComponent, self).__init__(name)
        if priors is None:
            priors = {}
        if init is None:
            init = {}
        if rvars is None:
            rvars = {}
        x, labels = any_to_tensor_and_labels(x, labels)
        # now we have x, shape and labels
        if intercept:
            x = tt.concatenate(
                [tt.ones((x.shape[0], 1), x.dtype), x],
                axis=1
            )
            labels = ['Intercept'] + labels
        for name in labels:
            if name == 'Intercept':
                if name in rvars:
                    self.add_var(name, rvars[name])
                else:
                    self.new_var(
                        name=name,
                        dist=priors.get(name, Flat.dist()),
                        test_val=init.get(name)
                    )
            else:
                if name in rvars:
                    self.add_var(name, rvars[name])
                else:
                    self.new_var(
                        name=name,
                        dist=priors.get(name, Normal.dist(mu=0, tau=1.0E-6)),
                        test_val=init.get(name)
                    )
        self.coeffs = tt.stack(self.vars.values(), axis=0)
        self.y_est = x.dot(self.coeffs)

    @classmethod
    def from_formula(cls, formula, data, priors=None, init=None, rvars=None, name=''):
        import patsy
        y, x = patsy.dmatrices(formula, data)
        labels = x.design_info.column_names
        return cls(np.asarray(x), np.asarray(y)[:, 0], intercept=False, labels=labels,
                   priors=priors, init=init, rvars=rvars, name=name)


class Glm(LinearComponent):
    """Creates glm model, y_est is accessible via attribute
        Parameters
        ----------
        name : str - name, associated with the linear component
        x : pd.DataFrame or np.ndarray
        y : pd.Series or np.array
        intercept : bool - fit with intercept or not?
        labels : list - replace variable names with these labels
        priors : dict - priors for coefficients
        init : dict - test_vals for coefficients
        rvars : dict - random variables instead of creating new ones
        family : pymc3.glm.families object
        """
    def __init__(self, x, y, intercept=True, labels=None,
                 priors=None, init=None, rvars=None, family='normal', name=''):
        super(Glm, self).__init__(x, y, intercept, labels, priors, init, rvars, name)

        _families = dict(
            normal=families.Normal,
            student=families.StudentT,
            binomial=families.Binomial,
            poisson=families.Poisson
        )
        if isinstance(family, str):
            family = _families[family]()
        family.create_likelihood(name, self.y_est, y, model=self.model)
        if name:
            name = '{}_y'.format(name)
        else:
            name = 'y'
        self.add_var('y', self.model.named_vars[name])
        self.y_est = self['y']

    @classmethod
    def from_formula(cls, formula, data, priors=None, init=None, rvars=None, family='normal', name=''):
        import patsy
        y, x = patsy.dmatrices(formula, data)
        labels = x.design_info.column_names
        return cls(np.asarray(x), np.asarray(y)[:, 0], intercept=False, labels=labels,
                   priors=priors, init=init, rvars=rvars, family=family, name=name)
