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

"""
=================
Bounded Variables
=================

PyMC includes the construct :class:`~pymc.distributions.bound.Bound` for
placing constraints on existing probability distributions.  It modifies a given
distribution to take values only within a specified interval.

Some types of variables require constraints.  For instance, it doesn't make
sense for a standard deviation to have a negative value, so something like a
Normal prior on a parameter that represents a standard deviation would be
inappropriate.  PyMC includes distributions that have positive support, such
as :class:`~pymc.distributions.continuous.Gamma` or
:class:`~pymc.distributions.continuous.Exponential`.  PyMC also includes
several bounded distributions, such as
:class:`~pymc.distributions.continuous.Uniform`,
:class:`~pymc.distributions.continuous.HalfNormal`, and
:class:`~pymc.distributions.continuous.HalfCauchy`, that are restricted to a
specific domain.

All univariate distributions in PyMC can be given bounds.  The distribution of
a continuous variable that has been bounded is automatically transformed into
an unnormalized distribution whose domain is unconstrained.  The transformation
improves the efficiency of sampling and variational inference algorithms.

Usage
#####

For example, one may have prior information that suggests that the value of a
parameter representing a standard deviation is near one.  One could use a
Normal distribution while constraining the support to be positive.  The
specification of a bounded distribution should go within the model block::

    import pymc as pm

    with pm.Model() as model:
        norm = Normal.dist(mu=1.0, sigma=3.0)
        x = Bound('x', norm, lower=0.0)

Caveats
#######

* Bounds cannot be given to variables that are ``observed``.  To model
  truncated data, use a :func:`~pymc.model.Potential` in combination with a cumulative
  probability function.  See `this example notebook <https://docs.pymc.io/pymc-examples/examples/survival_analysis/weibull_aft.html>`_.

* The automatic transformation applied to continuous distributions results in
  an unnormalized probability distribution.  This doesn't effect inference
  algorithms but may complicate some model comparison procedures.
"""

import numpy as np

from aesara.tensor import as_tensor_variable
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.var import TensorVariable

from pymc.aesaraf import floatX, intX
from pymc.distributions import _logp
from pymc.distributions.continuous import BoundedContinuous
from pymc.distributions.dist_math import bound
from pymc.distributions.distribution import Continuous, Discrete
from pymc.distributions.shape_utils import to_tuple
from pymc.model import modelcontext

__all__ = ["Bound"]


class BoundRV(RandomVariable):
    name = "bound"
    ndim_supp = 0
    ndims_params = [0, 0, 0]
    dtype = "floatX"
    _print_name = ("Bound", "\\operatorname{Bound}")

    @classmethod
    def rng_fn(cls, rng, distribution, lower, upper, size):
        raise NotImplementedError("Cannot sample from a bounded variable")


boundrv = BoundRV()


class _ContinuousBounded(BoundedContinuous):
    rv_op = boundrv
    bound_args_indices = [1, 2]

    def logp(value, distribution, lower, upper):
        """
        Calculate log-probability of Bounded distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value for which log-probability is calculated.
        distribution: TensorVariable
            Distribution which is being bounded
        lower: numeric
            Lower bound for the distribution being bounded.
        upper: numeric
            Upper bound for the distribution being bounded.

        Returns
        -------
        TensorVariable
        """
        logp = _logp(distribution.owner.op, value, {}, *distribution.owner.inputs[3:])
        return bound(logp, (value >= lower), (value <= upper))


class DiscreteBoundRV(BoundRV):
    name = "discrete_bound"
    dtype = "int64"


discrete_boundrv = DiscreteBoundRV()


class _DiscreteBounded(Discrete):
    rv_op = discrete_boundrv

    def __new__(cls, *args, **kwargs):
        transform = kwargs.get("transform", None)
        if transform is not None:
            raise ValueError("Cannot transform discrete variable.")
        return super().__new__(cls, *args, **kwargs)

    def logp(value, distribution, lower, upper):
        """
        Calculate log-probability of Bounded distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value for which log-probability is calculated.
        distribution: TensorVariable
            Distribution which is being bounded
        lower: numeric
            Lower bound for the distribution being bounded.
        upper: numeric
            Upper bound for the distribution being bounded.

        Returns
        -------
        TensorVariable
        """
        logp = _logp(distribution.owner.op, value, {}, *distribution.owner.inputs[3:])
        return bound(logp, (value >= lower), (value <= upper))


class Bound:
    r"""
    Create a Bound variable object that can be applied to create
    a new upper, lower, or upper and lower bounded distribution.

    The resulting distribution is not normalized anymore. This
    is usually fine if the bounds are constants. If you need
    truncated distributions, use `Bound` in combination with
    a :class:`~pymc.model.Potential` with the cumulative probability function.

    The bounds are inclusive for discrete distributions.

    Parameters
    ----------
    distribution: pymc distribution
        Distribution to be transformed into a bounded distribution.
    lower: float or array like, optional
        Lower bound of the distribution.
    upper: float or array like, optional
        Upper bound of the distribution.

    Examples
    --------
    .. code-block:: python

        with pm.Model():
            normal_dist = Normal.dist(mu=0.0, sigma=1.0, initval=-0.5)
            negative_normal = pm.Bound(normal_dist, upper=0.0)

    """

    def __new__(
        cls,
        name,
        distribution,
        lower=None,
        upper=None,
        size=None,
        shape=None,
        initval=None,
        dims=None,
        **kwargs,
    ):

        cls._argument_checks(distribution, **kwargs)

        if dims is not None:
            model = modelcontext(None)
            if dims in model.coords:
                dim_obj = np.asarray(model.coords[dims])
                size = dim_obj.shape
            else:
                raise ValueError("Given dims do not exist in model coordinates.")

        lower, upper, initval = cls._set_values(lower, upper, size, shape, initval)

        if isinstance(distribution.owner.op, Continuous):
            res = _ContinuousBounded(
                name,
                [distribution, lower, upper],
                initval=floatX(initval),
                size=size,
                shape=shape,
                **kwargs,
            )
        else:
            res = _DiscreteBounded(
                name,
                [distribution, lower, upper],
                initval=intX(initval),
                size=size,
                shape=shape,
                **kwargs,
            )
        return res

    @classmethod
    def dist(
        cls,
        distribution,
        lower=None,
        upper=None,
        size=None,
        shape=None,
        **kwargs,
    ):

        cls._argument_checks(distribution, **kwargs)
        lower, upper, initval = cls._set_values(lower, upper, size, shape, initval=None)

        if isinstance(distribution.owner.op, Continuous):
            res = _ContinuousBounded.dist(
                [distribution, lower, upper],
                size=size,
                shape=shape,
                **kwargs,
            )
            res.tag.test_value = floatX(initval)
        else:
            res = _DiscreteBounded.dist(
                [distribution, lower, upper],
                size=size,
                shape=shape,
                **kwargs,
            )
            res.tag.test_value = intX(initval)
        return res

    @classmethod
    def _argument_checks(cls, distribution, **kwargs):
        if "observed" in kwargs:
            raise ValueError(
                "Observed Bound distributions are not supported. "
                "If you want to model truncated data "
                "you can use a pm.Potential in combination "
                "with the cumulative probability function."
            )

        if not isinstance(distribution, TensorVariable):
            raise ValueError(
                "Passing a distribution class to `Bound` is no longer supported.\n"
                "Please pass the output of a distribution instantiated via the "
                "`.dist()` API such as:\n"
                '`pm.Bound("bound", pm.Normal.dist(0, 1), lower=0)`'
            )

        try:
            model = modelcontext(None)
        except TypeError:
            pass
        else:
            if distribution in model.basic_RVs:
                raise ValueError(
                    f"The distribution passed into `Bound` was already registered "
                    f"in the current model.\nYou should pass an unregistered "
                    f"(unnamed) distribution created via the `.dist()` API, such as:\n"
                    f'`pm.Bound("bound", pm.Normal.dist(0, 1), lower=0)`'
                )

        if distribution.owner.op.ndim_supp != 0:
            raise NotImplementedError("Bounding of MultiVariate RVs is not yet supported.")

        if not isinstance(distribution.owner.op, (Discrete, Continuous)):
            raise ValueError(
                f"`distribution` {distribution} must be a Discrete or Continuous"
                " distribution subclass"
            )

    @classmethod
    def _set_values(cls, lower, upper, size, shape, initval):
        if size is None:
            size = shape

        lower = np.asarray(lower)
        lower = floatX(np.where(lower == None, -np.inf, lower))
        upper = np.asarray(upper)
        upper = floatX(np.where(upper == None, np.inf, upper))

        if initval is None:
            _size = np.broadcast_shapes(to_tuple(size), np.shape(lower), np.shape(upper))
            _lower = np.broadcast_to(lower, _size)
            _upper = np.broadcast_to(upper, _size)
            initval = np.where(
                (_lower == -np.inf) & (_upper == np.inf),
                0,
                np.where(
                    _lower == -np.inf,
                    _upper - 1,
                    np.where(_upper == np.inf, _lower + 1, (_lower + _upper) / 2),
                ),
            )

        lower = as_tensor_variable(floatX(lower))
        upper = as_tensor_variable(floatX(upper))
        return lower, upper, initval
