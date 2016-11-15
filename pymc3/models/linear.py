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
    def __init__(self, name, x, y, intercept=True, labels=None,
                 priors=None, init=None, rvars=None):
        super(LinearComponent, self).__init__(name)
        if priors is None:
            priors = {}
        if init is None:
            init = {}
        if rvars is None:
            rvars = {}
        if isinstance(x, pd.DataFrame):
            self._init_pandas_(x, y, intercept, labels, priors, init, rvars)
        elif isinstance(x, np.ndarray):
            self._init_numpy_(x, y, intercept, labels, priors, init, rvars)
        else:
            raise NotImplementedError('Not implemented for type %s' % type(x))

    def _init_pandas_(self, x, y, intercept=True, labels=None, priors=None, init=None, rvars=None):
        if labels is None:
            labels = list(x.columns)
        else:
            assert len(labels) == x.shape[1], \
                'Cannot deal with not full list of labels, need {}'.format(x.shape[1])
        if intercept:
            x = pd.concat([pd.Series(np.ones(x.shape[0])), x], 1)
            _dist, _var, _init = (priors.get('Intercept'),
                                  rvars.get('Intercept'),
                                  init.get('Intercept'))
            self._add_intercept(_dist, _var, _init)
        else:
            self.intercept = 0
        for name in labels:
            if name == 'Intercept':     # comes from patsy
                _dist, _var, _init = (priors.get('Intercept'),
                                      rvars.get('Intercept'),
                                      init.get('Intercept'))
                self._add_intercept(_dist, _var, _init)
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
        self.y_est = tt.dot(np.asarray(x), self.coeffs).reshape((1, -1))

    def _init_numpy_(self, x, y, intercept=True, labels=None, priors=None, init=None, rvars=None):
        x = pd.DataFrame(x).rename(columns=lambda i: "x%d" % i)
        self._init_pandas_(x, y, intercept, labels, priors, init, rvars)

    @classmethod
    def from_formula(cls, name, formula, data, priors=None, init=None, rvars=None):
        import patsy
        y, x = patsy.dmatrices(formula, data)
        labels = x.design_info.column_names
        return cls(name, x, y, intercept=False, labels=labels, priors=priors, init=init, rvars=rvars)

    def _add_intercept(self, dist, var, init):
        if var:
            self.add_var('Intercept', var)
        else:
            self.new_var(
                name='Intercept',
                dist=dist or Flat.dist(),
                test_val=init
            )
        self.intercept = self['Intercept']


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
    def __init__(self, name, x, y, intercept=True, labels=None,
                 priors=None, init=None, rvars=None, family='normal'):
        super(Glm, self).__init__(name, x, y, intercept, labels, priors, init, rvars)

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
