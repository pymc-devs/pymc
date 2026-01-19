#   Copyright 2024 - present The PyMC Developers
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
from functools import singledispatch

import numpy as np
import pytensor
import pytensor.tensor as pt

from pytensor import config, graph_replace, scan
from pytensor.graph import Op
from pytensor.graph.basic import Apply
from pytensor.raise_op import CheckAndRaise
from pytensor.scan import until
from pytensor.tensor import TensorConstant, TensorVariable
from pytensor.tensor.random.basic import NormalRV
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.random.type import RandomType

from pymc.distributions.continuous import TruncatedNormal, bounded_cont_transform
from pymc.distributions.dist_math import check_parameters
from pymc.distributions.distribution import (
    Distribution,
    SymbolicRandomVariable,
    _support_point,
    support_point,
)
from pymc.distributions.shape_utils import (
    _change_dist_size,
    change_dist_size,
    rv_size_is_none,
    to_tuple,
)
from pymc.distributions.transforms import _default_transform
from pymc.exceptions import TruncationError
from pymc.logprob.abstract import _logcdf, _logprob
from pymc.logprob.basic import icdf, logccdf, logcdf, logp
from pymc.math import logdiffexp
from pymc.pytensorf import collect_default_updates
from pymc.util import check_dist_not_registered


class TruncatedRV(SymbolicRandomVariable):
    """An `Op` constructed from a PyTensor graph that represents a truncated univariate random variable."""

    default_output: int = 0
    base_rv_op: Op
    max_n_steps: int

    def __init__(
        self,
        *args,
        base_rv_op: Op,
        max_n_steps: int,
        **kwargs,
    ):
        self.base_rv_op = base_rv_op
        self.max_n_steps = max_n_steps
        self._print_name = (
            f"Truncated{self.base_rv_op._print_name[0]}",
            f"\\operatorname{{{self.base_rv_op._print_name[1]}}}",
        )
        super().__init__(*args, **kwargs)

    @classmethod
    def rv_op(cls, dist, lower, upper, max_n_steps, *, size=None):
        # We don't accept rng because we don't have control over it when using a specialized Op
        # and there may be a need for multiple RNGs in dist.

        # Try to use specialized Op
        try:
            return _truncated(dist.owner.op, lower, upper, size, *dist.owner.inputs)
        except NotImplementedError:
            pass

        lower = pt.as_tensor_variable(lower) if lower is not None else pt.constant(-np.inf)
        upper = pt.as_tensor_variable(upper) if upper is not None else pt.constant(np.inf)

        if size is not None:
            size = pt.as_tensor(size, dtype="int64", ndim=1)

        if rv_size_is_none(size):
            size = pt.broadcast_shape(dist, lower, upper)

        dist = change_dist_size(dist, new_size=size)

        rv_inputs = [
            inp
            if not isinstance(inp.type, RandomType)
            else pytensor.shared(np.random.default_rng())
            for inp in dist.owner.inputs
        ]
        graph_inputs = [*rv_inputs, lower, upper]

        # Variables with `_` suffix identify dummy inputs for the OpFromGraph
        graph_inputs_ = [
            inp.type() if not isinstance(inp.type, RandomType) else inp for inp in graph_inputs
        ]
        *rv_inputs_, lower_, upper_ = graph_inputs_

        rv_ = dist.owner.op.make_node(*rv_inputs_).default_output()

        # Try to use inverted cdf sampling
        # truncated_rv = icdf(rv, draw(uniform(cdf(lower), cdf(upper))))
        try:
            logcdf_lower_, logcdf_upper_ = TruncatedRV._create_logcdf_exprs(
                rv_, rv_, lower_, upper_
            )
            # We use the first RNG from the base RV, so we don't have to introduce a new one
            # This is not problematic because the RNG won't be used in the RV logcdf graph
            uniform_rng_ = next(inp_ for inp_ in rv_inputs_ if isinstance(inp_.type, RandomType))
            uniform_next_rng_, uniform_ = pt.random.uniform(
                pt.exp(logcdf_lower_),
                pt.exp(logcdf_upper_),
                rng=uniform_rng_,
                size=rv_.shape,
            ).owner.outputs
            truncated_rv_ = icdf(rv_, uniform_, warn_rvs=False)
            return TruncatedRV(
                base_rv_op=dist.owner.op,
                inputs=graph_inputs_,
                outputs=[truncated_rv_, uniform_next_rng_],
                ndim_supp=0,
                max_n_steps=max_n_steps,
            )(*graph_inputs)
        except NotImplementedError:
            pass

        # Fallback to rejection sampling
        # truncated_rv = zeros(rv.shape)
        # reject_draws = ones(rv.shape, dtype=bool)
        # while any(reject_draws):
        #    truncated_rv[reject_draws] = draw(rv)[reject_draws]
        #    reject_draws = (truncated_rv < lower) | (truncated_rv > upper)
        def loop_fn(truncated_rv, reject_draws, lower, upper, *rv_inputs):
            new_truncated_rv = dist.owner.op.make_node(*rv_inputs).default_output()
            # Avoid scalar boolean indexing
            if truncated_rv.type.ndim == 0:
                truncated_rv = new_truncated_rv
            else:
                truncated_rv = pt.set_subtensor(
                    truncated_rv[reject_draws],
                    new_truncated_rv[reject_draws],
                )
            reject_draws = pt.or_((truncated_rv < lower), (truncated_rv > upper))

            return (
                (truncated_rv, reject_draws),
                collect_default_updates(new_truncated_rv),
                until(~pt.any(reject_draws)),
            )

        (truncated_rv_, reject_draws_), updates = scan(
            loop_fn,
            outputs_info=[
                pt.zeros_like(rv_),
                pt.ones_like(rv_, dtype=bool),
            ],
            non_sequences=[lower_, upper_, *rv_inputs_],
            n_steps=max_n_steps,
            strict=True,
        )

        truncated_rv_ = truncated_rv_[-1]
        convergence_ = ~pt.any(reject_draws_[-1])
        truncated_rv_ = TruncationCheck(f"Truncation did not converge in {max_n_steps} steps")(
            truncated_rv_, convergence_
        )

        # Sort updates of each RNG so that they show in the same order as the input RNGs
        def sort_updates(update):
            rng, next_rng = update
            return graph_inputs.index(rng)

        next_rngs = [next_rng for rng, next_rng in sorted(updates.items(), key=sort_updates)]

        return TruncatedRV(
            base_rv_op=dist.owner.op,
            inputs=graph_inputs_,
            outputs=[truncated_rv_, *next_rngs],
            ndim_supp=0,
            max_n_steps=max_n_steps,
        )(*graph_inputs)

    @staticmethod
    def _create_logcdf_exprs(
        base_rv: TensorVariable,
        value: TensorVariable,
        lower: TensorVariable,
        upper: TensorVariable,
    ) -> tuple[TensorVariable, TensorVariable]:
        """Create lower and upper logcdf expressions for base_rv.

        Uses `value` as a template for broadcasting.
        """
        # For left truncated discrete RVs, we need to include the whole lower bound.
        lower_value = lower - 1 if base_rv.type.dtype.startswith("int") else lower
        lower_value = pt.full_like(value, lower_value, dtype=config.floatX)
        upper_value = pt.full_like(value, upper, dtype=config.floatX)
        lower_logcdf = logcdf(base_rv, lower_value, warn_rvs=False)
        upper_logcdf = graph_replace(lower_logcdf, {lower_value: upper_value})
        return lower_logcdf, upper_logcdf

    @staticmethod
    def _create_lower_logccdf_expr(
        base_rv: TensorVariable,
        value: TensorVariable,
        lower: TensorVariable,
    ) -> TensorVariable:
        """Create logccdf expression at lower bound for base_rv.

        Uses `value` as a template for broadcasting. This is numerically more
        stable than computing log(1 - exp(logcdf)) for distributions that have
        a registered logccdf method.
        """
        # For left truncated discrete RVs, we need to include the whole lower bound.
        lower_value = lower - 1 if base_rv.type.dtype.startswith("int") else lower
        lower_value = pt.full_like(value, lower_value, dtype=config.floatX)
        return logccdf(base_rv, lower_value, warn_rvs=False)

    def update(self, node: Apply):
        """Return the update mapping for the internal RNGs.

        TruncatedRVs are created in a way that the rng updates follow the same order as the input RNGs.
        """
        rngs = [inp for inp in node.inputs if isinstance(inp.type, RandomType)]
        next_rngs = [out for out in node.outputs if isinstance(out.type, RandomType)]
        return dict(zip(rngs, next_rngs))


@singledispatch
def _truncated(op: Op, lower, upper, size, *params):
    """Return the truncated equivalent of another `RandomVariable`."""
    raise NotImplementedError(f"{op} does not have an equivalent truncated version implemented")


class TruncationCheck(CheckAndRaise):
    """Implements a check in truncated graphs.

    Raises `TruncationError` if the check is not True.
    """

    def __init__(self, msg=""):
        super().__init__(TruncationError, msg)

    def __str__(self):
        """Return a string representation of the object."""
        return f"TruncationCheck{{{self.msg}}}"


class Truncated(Distribution):
    r"""
    Truncated distribution.

    The pdf of a Truncated distribution is

    .. math::

        \begin{cases}
            0 & \text{for } x < lower, \\
            \frac{\text{PDF}(x, dist)}{\text{CDF}(upper, dist) - \text{CDF}(lower, dist)}
            & \text{for } lower <= x <= upper, \\
            0 & \text{for } x > upper,
        \end{cases}


    Parameters
    ----------
    dist: unnamed distribution
        Univariate distribution created via the `.dist()` API, which will be truncated.
        This distribution must be a pure RandomVariable and have a logcdf method
        implemented for MCMC sampling.

        .. warning:: dist will be cloned, rendering it independent of the one passed as input.

    lower: tensor_like of float or None
        Lower (left) truncation point. If `None` the distribution will not be left truncated.
    upper: tensor_like of float or None
        Upper (right) truncation point. If `None`, the distribution will not be right truncated.
    max_n_steps: int, defaults 10_000
        Maximum number of resamples that are attempted when performing rejection sampling.
        A `TruncationError` is raised if convergence is not reached after that many steps.

    Returns
    -------
    truncated_distribution: TensorVariable
        Graph representing a truncated `RandomVariable`. A specialized `Op` may be used
        if the `Op` of the dist has a dispatched `_truncated` function. Otherwise, a
        `SymbolicRandomVariable` graph representing the truncation process, via inverse
        CDF sampling (if the underlying dist has a logcdf method), or rejection sampling
        is returned.

    Examples
    --------
    .. code-block:: python

        with pm.Model():
            normal_dist = pm.Normal.dist(mu=0.0, sigma=1.0)
            truncated_normal = pm.Truncated("truncated_normal", normal_dist, lower=-1, upper=1)

    """

    rv_type = TruncatedRV
    rv_op = rv_type.rv_op

    @classmethod
    def dist(cls, dist, lower=None, upper=None, max_n_steps: int = 10_000, **kwargs):
        if not (
            isinstance(dist, TensorVariable)
            and dist.owner is not None
            and isinstance(dist.owner.op, RandomVariable | SymbolicRandomVariable)
        ):
            raise ValueError(
                f"Truncation dist must be a distribution created via the `.dist()` API, got {type(dist)}"
            )

        if (
            isinstance(dist.owner.op, SymbolicRandomVariable)
            and "[size]" not in dist.owner.op.extended_signature
        ):
            # Truncation needs to wrap the underlying dist, but not all SymbolicRandomVariables encapsulate the whole
            # random graph and as such we don't know where the actual inputs begin. This happens mostly for
            # distribution factories like `Censored` and `Mixture` which would have a very complex signature if they
            # encapsulated the random components instead of taking them as inputs like they do now.
            # SymbolicRandomVariables that encapsulate the whole random graph can be identified for having a size parameter.
            raise NotImplementedError(f"Truncation not implemented for {dist.owner.op}")

        if dist.owner.op.ndim_supp > 0:
            raise NotImplementedError("Truncation not implemented for multivariate distributions")

        check_dist_not_registered(dist)

        if lower is None and upper is None:
            raise ValueError("lower and upper cannot both be None")

        return super().dist([dist, lower, upper, max_n_steps], **kwargs)


@_change_dist_size.register(TruncatedRV)
def change_truncated_size(op: TruncatedRV, truncated_rv, new_size, expand):
    *rv_inputs, lower, upper = truncated_rv.owner.inputs
    untruncated_rv = op.base_rv_op.make_node(*rv_inputs).default_output()

    if expand:
        new_size = to_tuple(new_size) + tuple(truncated_rv.shape)

    return Truncated.rv_op(
        untruncated_rv,
        lower=lower,
        upper=upper,
        size=new_size,
        max_n_steps=op.max_n_steps,
    )


@_support_point.register(TruncatedRV)
def truncated_support_point(op: TruncatedRV, truncated_rv, *inputs):
    *rv_inputs, lower, upper = inputs

    # recreate untruncated rv and respective support_point
    untruncated_rv = op.base_rv_op.make_node(*rv_inputs).default_output()
    untruncated_support_point = support_point(untruncated_rv)

    fallback_support_point = pt.switch(
        pt.and_(pt.bitwise_not(pt.isinf(lower)), pt.bitwise_not(pt.isinf(upper))),
        (upper - lower) / 2,  # lower and upper are finite
        pt.switch(
            pt.isinf(upper),
            lower + 1,  # only lower is finite
            upper - 1,  # only upper is finite
        ),
    )

    return pt.switch(
        pt.and_(pt.ge(untruncated_support_point, lower), pt.le(untruncated_support_point, upper)),
        untruncated_support_point,  # untruncated support_point is between lower and upper
        fallback_support_point,
    )


@_default_transform.register(TruncatedRV)
def truncated_default_transform(op, truncated_rv):
    # Don't transform discrete truncated distributions
    if truncated_rv.type.dtype.startswith("int"):
        return None
    # Lower and Upper are the arguments -2 and -1
    return bounded_cont_transform(op, truncated_rv, bound_args_indices=(-2, -1))


@_logprob.register(TruncatedRV)
def truncated_logprob(op, values, *inputs, **kwargs):
    (value,) = values
    *rv_inputs, lower, upper = inputs

    base_rv_op = op.base_rv_op
    base_rv = base_rv_op.make_node(*rv_inputs).default_output()
    base_logp = logp(base_rv, value)
    lower_logcdf, upper_logcdf = TruncatedRV._create_logcdf_exprs(base_rv, value, lower, upper)
    if base_rv_op.name:
        base_logp.name = f"{base_rv_op}_logprob"
        lower_logcdf.name = f"{base_rv_op}_lower_logcdf"
        upper_logcdf.name = f"{base_rv_op}_upper_logcdf"

    is_lower_bounded = not (isinstance(lower, TensorConstant) and np.all(np.isneginf(lower.value)))
    is_upper_bounded = not (isinstance(upper, TensorConstant) and np.all(np.isinf(upper.value)))

    lognorm = 0
    if is_lower_bounded and is_upper_bounded:
        lognorm = logdiffexp(upper_logcdf, lower_logcdf)
    elif is_lower_bounded:
        lognorm = TruncatedRV._create_lower_logccdf_expr(base_rv, value, lower)
    elif is_upper_bounded:
        lognorm = upper_logcdf

    truncated_logp = base_logp - lognorm

    if is_lower_bounded:
        truncated_logp = pt.switch(value < lower, -np.inf, truncated_logp)

    if is_upper_bounded:
        truncated_logp = pt.switch(value <= upper, truncated_logp, -np.inf)

    if is_lower_bounded and is_upper_bounded:
        truncated_logp = check_parameters(
            truncated_logp,
            pt.le(lower, upper),
            msg="lower_bound <= upper_bound",
        )

    return truncated_logp


@_logcdf.register(TruncatedRV)
def truncated_logcdf(op: TruncatedRV, value, *inputs, **kwargs):
    *rv_inputs, lower, upper = inputs

    base_rv = op.base_rv_op.make_node(*rv_inputs).default_output()
    base_logcdf = logcdf(base_rv, value)
    lower_logcdf, upper_logcdf = TruncatedRV._create_logcdf_exprs(base_rv, value, lower, upper)

    is_lower_bounded = not (isinstance(lower, TensorConstant) and np.all(np.isneginf(lower.value)))
    is_upper_bounded = not (isinstance(upper, TensorConstant) and np.all(np.isinf(upper.value)))

    lognorm = 0
    if is_lower_bounded and is_upper_bounded:
        lognorm = logdiffexp(upper_logcdf, lower_logcdf)
    elif is_lower_bounded:
        lognorm = TruncatedRV._create_lower_logccdf_expr(base_rv, value, lower)
    elif is_upper_bounded:
        lognorm = upper_logcdf

    logcdf_numerator = logdiffexp(base_logcdf, lower_logcdf) if is_lower_bounded else base_logcdf
    logcdf_trunc = logcdf_numerator - lognorm

    if is_lower_bounded:
        logcdf_trunc = pt.switch(value < lower, -np.inf, logcdf_trunc)

    if is_upper_bounded:
        logcdf_trunc = pt.switch(value <= upper, logcdf_trunc, 0.0)

    if is_lower_bounded and is_upper_bounded:
        logcdf_trunc = check_parameters(
            logcdf_trunc,
            pt.le(lower, upper),
            msg="lower_bound <= upper_bound",
        )

    return logcdf_trunc


@_truncated.register(NormalRV)
def _truncated_normal(op, lower, upper, size, rng, old_size, mu, sigma):
    return TruncatedNormal.dist(
        mu=mu,
        sigma=sigma,
        lower=lower,
        upper=upper,
        rng=None,  # Do not reuse rng to avoid weird dependencies
        size=size,
        dtype=op.dtype,
    )
