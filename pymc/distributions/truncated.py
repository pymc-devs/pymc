from functools import singledispatch

import aesara
import aesara.tensor as at
import numpy as np

from aeppl.abstract import MeasurableVariable
from aeppl.logprob import _logcdf, _logprob, icdf, logcdf
from aesara import scan
from aesara.graph import Op
from aesara.graph.basic import Node
from aesara.raise_op import CheckAndRaise
from aesara.scan import until
from aesara.tensor import TensorConstant, TensorVariable
from aesara.tensor.random.basic import NormalRV
from aesara.tensor.random.op import RandomVariable

from pymc.distributions.continuous import TruncatedNormal, bounded_cont_transform
from pymc.distributions.dist_math import check_parameters
from pymc.distributions.distribution import (
    Distribution,
    SymbolicRandomVariable,
    _moment,
    moment,
)
from pymc.distributions.shape_utils import _change_dist_size, change_dist_size, to_tuple
from pymc.distributions.transforms import _default_transform
from pymc.exceptions import TruncationError
from pymc.math import logdiffexp
from pymc.util import check_dist_not_registered


class TruncatedRV(SymbolicRandomVariable):
    """An `Op` constructed from an Aesara graph that represents a truncated univariate
    random variable."""

    default_output = 1
    base_rv_op = None
    max_n_steps = None

    def __init__(self, *args, base_rv_op: Op, max_n_steps: int, **kwargs):
        self.base_rv_op = base_rv_op
        self.max_n_steps = max_n_steps
        super().__init__(*args, **kwargs)

    def update(self, node: Node):
        """Return the update mapping for the noise RV."""
        # Since RNG is a shared variable it shows up as the last node input
        return {node.inputs[-1]: node.outputs[0]}


MeasurableVariable.register(TruncatedRV)


@singledispatch
def _truncated(op: Op, lower, upper, *params):
    """Return the truncated equivalent of another `RandomVariable`."""
    raise NotImplementedError(f"{op} does not have an equivalent truncated version implemented")


class TruncationCheck(CheckAndRaise):
    """Implements a check in truncated graphs.
    Raises `TruncationError` if the check is not True.
    """

    def __init__(self, msg=""):
        super().__init__(TruncationError, msg)

    def __str__(self):
        return f"TruncationCheck{{{self.msg}}}"


class Truncated(Distribution):
    r"""
    Truncated distribution

    The pdf of a censored distribution is

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
    """

    rv_type = TruncatedRV

    @classmethod
    def dist(cls, dist, lower=None, upper=None, max_n_steps: int = 10_000, **kwargs):
        if not (isinstance(dist, TensorVariable) and isinstance(dist.owner.op, RandomVariable)):
            if isinstance(dist.owner.op, SymbolicRandomVariable):
                raise NotImplementedError(
                    f"Truncation not implemented for SymbolicRandomVariable {dist.owner.op}"
                )
            raise ValueError(
                f"Truncation dist must be a distribution created via the `.dist()` API, got {type(dist)}"
            )

        if dist.owner.op.ndim_supp > 0:
            raise NotImplementedError("Truncation not implemented for multivariate distributions")

        check_dist_not_registered(dist)

        if lower is None and upper is None:
            raise ValueError("lower and upper cannot both be None")

        return super().dist([dist, lower, upper, max_n_steps], **kwargs)

    @classmethod
    def rv_op(cls, dist, lower, upper, max_n_steps, size=None):

        # Try to use specialized Op
        try:
            return _truncated(dist.owner.op, lower, upper, *dist.owner.inputs)
        except NotImplementedError:
            pass

        lower = at.as_tensor_variable(lower) if lower is not None else at.constant(-np.inf)
        upper = at.as_tensor_variable(upper) if upper is not None else at.constant(np.inf)

        if size is None:
            size = at.broadcast_shape(dist, lower, upper)
        dist = change_dist_size(dist, new_size=size)

        # Variables with `_` suffix identify dummy inputs for the OpFromGraph
        graph_inputs = [*dist.owner.inputs[1:], lower, upper]
        graph_inputs_ = [inp.type() for inp in graph_inputs]
        *rv_inputs_, lower_, upper_ = graph_inputs_

        # We will use a Shared RNG variable because Scan demands it, even though it
        # would not be necessary for the OpFromGraph inverse cdf.
        rng = aesara.shared(np.random.default_rng())
        rv_ = dist.owner.op.make_node(rng, *rv_inputs_).default_output()

        # Try to use inverted cdf sampling
        try:
            # For left truncated discrete RVs, we need to include the whole lower bound.
            # This may result in draws below the truncation range, if any uniform == 0
            lower_value = lower_ - 1 if dist.owner.op.dtype.startswith("int") else lower_
            cdf_lower_ = at.exp(logcdf(rv_, lower_value))
            cdf_upper_ = at.exp(logcdf(rv_, upper_))
            # It's okay to reuse the same rng here, because the rng in rv_ will not be
            # used by either the logcdf of icdf functions
            uniform_ = at.random.uniform(
                cdf_lower_,
                cdf_upper_,
                rng=rng,
                size=rv_inputs_[0],
            )
            truncated_rv_ = icdf(rv_, uniform_)
            return TruncatedRV(
                base_rv_op=dist.owner.op,
                inputs=graph_inputs_,
                outputs=[uniform_.owner.outputs[0], truncated_rv_],
                ndim_supp=0,
                max_n_steps=max_n_steps,
            )(*graph_inputs)
        except NotImplementedError:
            pass

        # Fallback to rejection sampling
        def loop_fn(truncated_rv, reject_draws, lower, upper, rng, *rv_inputs):
            next_rng, new_truncated_rv = dist.owner.op.make_node(rng, *rv_inputs).outputs
            truncated_rv = at.set_subtensor(
                truncated_rv[reject_draws],
                new_truncated_rv[reject_draws],
            )
            reject_draws = at.or_((truncated_rv < lower), (truncated_rv > upper))

            return (
                (truncated_rv, reject_draws),
                [(rng, next_rng)],
                until(~at.any(reject_draws)),
            )

        (truncated_rv_, reject_draws_), updates = scan(
            loop_fn,
            outputs_info=[
                at.zeros_like(rv_),
                at.ones_like(rv_, dtype=bool),
            ],
            non_sequences=[lower_, upper_, rng, *rv_inputs_],
            n_steps=max_n_steps,
            strict=True,
        )

        truncated_rv_ = truncated_rv_[-1]
        convergence_ = ~at.any(reject_draws_[-1])
        truncated_rv_ = TruncationCheck(f"Truncation did not converge in {max_n_steps} steps")(
            truncated_rv_, convergence_
        )

        return TruncatedRV(
            base_rv_op=dist.owner.op,
            inputs=graph_inputs_,
            outputs=[tuple(updates.values())[0], truncated_rv_],
            ndim_supp=0,
            max_n_steps=max_n_steps,
        )(*graph_inputs)


@_change_dist_size.register(TruncatedRV)
def change_truncated_size(op, dist, new_size, expand):
    *rv_inputs, lower, upper, rng = dist.owner.inputs
    # Recreate the original untruncated RV
    untruncated_rv = op.base_rv_op.make_node(rng, *rv_inputs).default_output()
    if expand:
        new_size = to_tuple(new_size) + tuple(dist.shape)

    return Truncated.rv_op(
        untruncated_rv,
        lower=lower,
        upper=upper,
        size=new_size,
        max_n_steps=op.max_n_steps,
    )


@_moment.register(TruncatedRV)
def truncated_moment(op, rv, *inputs):
    *rv_inputs, lower, upper, rng = inputs

    # recreate untruncated rv and respective moment
    untruncated_rv = op.base_rv_op.make_node(rng, *rv_inputs).default_output()
    untruncated_moment = moment(untruncated_rv)

    fallback_moment = at.switch(
        at.and_(at.bitwise_not(at.isinf(lower)), at.bitwise_not(at.isinf(upper))),
        (upper - lower) / 2,  # lower and upper are finite
        at.switch(
            at.isinf(upper),
            lower + 1,  # only lower is finite
            upper - 1,  # only upper is finite
        ),
    )

    return at.switch(
        at.and_(at.ge(untruncated_moment, lower), at.le(untruncated_moment, upper)),
        untruncated_moment,  # untruncated moment is between lower and upper
        fallback_moment,
    )


@_default_transform.register(TruncatedRV)
def truncated_default_transform(op, rv):
    # Don't transform discrete truncated distributions
    if op.base_rv_op.dtype.startswith("int"):
        return None
    # Lower and Upper are the arguments -3 and -2
    return bounded_cont_transform(op, rv, bound_args_indices=(-3, -2))


@_logprob.register(TruncatedRV)
def truncated_logprob(op, values, *inputs, **kwargs):
    (value,) = values

    *rv_inputs, lower, upper, rng = inputs
    rv_inputs = [rng, *rv_inputs]

    base_rv_op = op.base_rv_op
    logp = _logprob(base_rv_op, (value,), *rv_inputs, **kwargs)
    # For left truncated RVs, we don't want to include the lower bound in the
    # normalization term
    lower_value = lower - 1 if base_rv_op.dtype.startswith("int") else lower
    lower_logcdf = _logcdf(base_rv_op, lower_value, *rv_inputs, **kwargs)
    upper_logcdf = _logcdf(base_rv_op, upper, *rv_inputs, **kwargs)

    if base_rv_op.name:
        logp.name = f"{base_rv_op}_logprob"
        lower_logcdf.name = f"{base_rv_op}_lower_logcdf"
        upper_logcdf.name = f"{base_rv_op}_upper_logcdf"

    is_lower_bounded = not (isinstance(lower, TensorConstant) and np.all(np.isneginf(lower.value)))
    is_upper_bounded = not (isinstance(upper, TensorConstant) and np.all(np.isinf(upper.value)))

    lognorm = 0
    if is_lower_bounded and is_upper_bounded:
        lognorm = logdiffexp(upper_logcdf, lower_logcdf)
    elif is_lower_bounded:
        lognorm = at.log1mexp(lower_logcdf)
    elif is_upper_bounded:
        lognorm = upper_logcdf

    logp = logp - lognorm

    if is_lower_bounded:
        logp = at.switch(value < lower, -np.inf, logp)

    if is_upper_bounded:
        logp = at.switch(value <= upper, logp, -np.inf)

    if is_lower_bounded and is_upper_bounded:
        logp = check_parameters(
            logp,
            at.le(lower, upper),
            msg="lower_bound <= upper_bound",
        )

    return logp


@_truncated.register(NormalRV)
def _truncated_normal(op, lower, upper, rng, size, dtype, mu, sigma):
    return TruncatedNormal.dist(
        mu=mu,
        sigma=sigma,
        lower=lower,
        upper=upper,
        rng=None,  # Do not reuse rng to avoid weird dependencies
        size=size,
        dtype=dtype,
    )
