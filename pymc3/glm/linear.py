#   Copyright 2020 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy as np
import theano.tensor as tt

from pymc3.distributions import Flat, Normal
from pymc3.glm import families
from pymc3.glm.utils import any_to_tensor_and_labels
from pymc3.model import Deterministic, Model

__all__ = ["LinearComponent", "GLM"]


class LinearComponent(Model):
    """Creates linear component, y_est is accessible via attribute

    Parameters
    ----------
    name: str - name, associated with the linear component
    x: pd.DataFrame or np.ndarray
    y: pd.Series or np.array
    intercept: bool - fit with intercept or not?
    labels: list - replace variable names with these labels
    priors: dict - priors for coefficients
        use `Intercept` key for defining Intercept prior
            defaults to Flat.dist()
        use `Regressor` key for defining default prior for all regressors
            defaults to Normal.dist(mu=0, tau=1.0E-6)
    vars: dict - random variables instead of creating new ones
    offset: scalar, or numpy/theano array with the same shape as y
        this can be used to specify an a priori known component to be
        included in the linear predictor during fitting.
    """

    default_regressor_prior = Normal.dist(mu=0, tau=1.0e-6)
    default_intercept_prior = Flat.dist()

    def __init__(
        self,
        x,
        y,
        intercept=True,
        labels=None,
        priors=None,
        vars=None,
        name="",
        model=None,
        offset=0.0,
    ):
        super().__init__(name, model)
        if len(y.shape) > 1:
            err_msg = (
                "Only one-dimensional observed variable objects (i.e."
                " of shape `(n, )`) are supported"
            )
            raise TypeError(err_msg)
        if priors is None:
            priors = {}
        if vars is None:
            vars = {}
        x, labels = any_to_tensor_and_labels(x, labels)
        # now we have x, shape and labels
        if intercept:
            x = tt.concatenate([tt.ones((x.shape[0], 1), x.dtype), x], axis=1)
            labels = ["Intercept"] + labels
        coeffs = list()
        for name in labels:
            if name == "Intercept":
                if name in vars:
                    v = Deterministic(name, vars[name])
                else:
                    v = self.Var(name=name, dist=priors.get(name, self.default_intercept_prior))
                coeffs.append(v)
            else:
                if name in vars:
                    v = Deterministic(name, vars[name])
                else:
                    v = self.Var(
                        name=name,
                        dist=priors.get(
                            name, priors.get("Regressor", self.default_regressor_prior)
                        ),
                    )
                coeffs.append(v)
        self.coeffs = tt.stack(coeffs, axis=0)
        self.y_est = x.dot(self.coeffs) + offset

    @classmethod
    def from_formula(
        cls, formula, data, priors=None, vars=None, name="", model=None, offset=0.0, eval_env=0
    ):
        """Creates linear component from `patsy` formula.

        Parameters
        ----------
        formula: str - a patsy formula
        data: a dict-like object that can be used to look up variables referenced
            in `formula`
        eval_env: either a `patsy.EvalEnvironment` or else a depth represented as
            an integer which will be passed to `patsy.EvalEnvironment.capture()`.
            See `patsy.dmatrix` and `patsy.EvalEnvironment` for details.
        Other arguments are documented in the constructor.
        """
        import patsy

        eval_env = patsy.EvalEnvironment.capture(eval_env, reference=1)
        y, x = patsy.dmatrices(formula, data, eval_env=eval_env)
        labels = x.design_info.column_names
        return cls(
            np.asarray(x),
            np.asarray(y)[:, -1],
            intercept=False,
            labels=labels,
            priors=priors,
            vars=vars,
            name=name,
            model=model,
            offset=offset,
        )


class GLM(LinearComponent):
    """Creates glm model, y_est is accessible via attribute

    Parameters
    ----------
    name: str - name, associated with the linear component
    x: pd.DataFrame or np.ndarray
    y: pd.Series or np.array
    intercept: bool - fit with intercept or not?
    labels: list - replace variable names with these labels
    priors: dict - priors for coefficients
        use `Intercept` key for defining Intercept prior
            defaults to Flat.dist()
        use `Regressor` key for defining default prior for all regressors
            defaults to Normal.dist(mu=0, tau=1.0E-6)
    init: dict - test_vals for coefficients
    vars: dict - random variables instead of creating new ones
    family: pymc3..families object
    offset: scalar, or numpy/theano array with the same shape as y
        this can be used to specify an a priori known component to be
        included in the linear predictor during fitting.
    """

    def __init__(
        self,
        x,
        y,
        intercept=True,
        labels=None,
        priors=None,
        vars=None,
        family="normal",
        name="",
        model=None,
        offset=0.0,
    ):
        super().__init__(
            x,
            y,
            intercept=intercept,
            labels=labels,
            priors=priors,
            vars=vars,
            name=name,
            model=model,
            offset=offset,
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
        self.y_est = family.create_likelihood(name="", y_est=self.y_est, y_data=y, model=self)

    @classmethod
    def from_formula(
        cls,
        formula,
        data,
        priors=None,
        vars=None,
        family="normal",
        name="",
        model=None,
        offset=0.0,
        eval_env=0,
    ):
        """
        Creates GLM from formula.

        Parameters
        ----------
        formula: str - a `patsy` formula
        data: a dict-like object that can be used to look up variables referenced
            in `formula`
        eval_env: either a `patsy.EvalEnvironment` or else a depth represented as
            an integer which will be passed to `patsy.EvalEnvironment.capture()`.
            See `patsy.dmatrix` and `patsy.EvalEnvironment` for details.
        Other arguments are documented in the constructor.
        """
        import patsy

        eval_env = patsy.EvalEnvironment.capture(eval_env, reference=1)
        y, x = patsy.dmatrices(formula, data, eval_env=eval_env)
        labels = x.design_info.column_names
        return cls(
            np.asarray(x),
            np.asarray(y)[:, -1],
            intercept=False,
            labels=labels,
            priors=priors,
            vars=vars,
            family=family,
            name=name,
            model=model,
            offset=offset,
        )


glm = GLM
