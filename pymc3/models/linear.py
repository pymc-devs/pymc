import theano
import theano.tensor as tt
import pandas as pd
import numpy as np
from ..distributions import Normal, Flat
from ..glm import families
from .base import UserModel


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
        if isinstance(x, pd.DataFrame):
            labels = list(x.columns)
            x = x.as_matrix()
        elif isinstance(x, pd.Series):
            labels = [x.name]
            x = x.as_matrix()[:, None]
        elif isinstance(x, dict):
            x = pd.DataFrame(x)
            labels = x.columns
            x = x.as_matrix()
        else:
            tx = type(x)
            x = tt.as_tensor_variable(x)
            try:
                shape = x.shape.eval()
            except theano.gof.fg.MissingInputError:
                shape = None
            if not labels and shape is None:
                raise TypeError(
                    'Cannot infer shape of %r, '
                    'please provide list of labels '
                    'or x without missing inputs' % tx
                )
            if len(shape) == 0:
                raise ValueError('scalars are not available as input')
            elif len(shape) == 1:
                x = x[:, None]
                shape = x.shape.eval()
            if not labels:
                labels = ['x%d' % i for i in range(shape[1])]
            else:
                assert len(labels) == shape[1], (
                    'Please provide all labels for coefficients'
                )
        if isinstance(labels, pd.RangeIndex):
            labels = ['x%d' % i for i in labels]
        if not isinstance(labels, list):
            labels = list(labels)
        x = tt.as_tensor_variable(x)
        shape = x.shape.eval()
        # now we have x, shape and labels
        if intercept:
            x = tt.concatenate(
                [tt.ones((shape[0], 1), x.dtype), x],
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
