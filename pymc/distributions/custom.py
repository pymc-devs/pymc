#   Copyright 2024 The PyMC Developers
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
import functools
import re
import warnings

from collections.abc import Callable, Sequence

from pytensor import Variable, clone_replace
from pytensor import tensor as pt
from pytensor.graph.basic import io_toposort
from pytensor.graph.features import ReplaceValidate
from pytensor.graph.rewriting.basic import GraphRewriter
from pytensor.scan.op import Scan
from pytensor.tensor import TensorVariable, as_tensor_variable
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.random.type import RandomGeneratorType, RandomType
from pytensor.tensor.random.utils import normalize_size_param
from pytensor.tensor.utils import safe_signature

from pymc.distributions.distribution import (
    Distribution,
    SymbolicRandomVariable,
    _support_point,
    support_point,
)
from pymc.distributions.shape_utils import _change_dist_size, rv_size_is_none
from pymc.exceptions import BlockModelAccessError
from pymc.logprob.abstract import _logcdf, _logprob
from pymc.model.core import new_or_existing_block_model_access
from pymc.pytensorf import collect_default_updates


def default_not_implemented(rv_name, method_name):
    message = (
        f"Attempted to run {method_name} on the CustomDist '{rv_name}', "
        f"but this method had not been provided when the distribution was "
        f"constructed. Please re-build your model and provide a callable "
        f"to '{rv_name}'s {method_name} keyword argument.\n"
    )

    def func(*args, **kwargs):
        raise NotImplementedError(message)

    return func


def default_support_point(rv, size, *rv_inputs, rv_name=None, has_fallback=False):
    if None not in rv.type.shape:
        return pt.zeros(rv.type.shape)
    elif rv.owner.op.ndim_supp == 0 and not rv_size_is_none(size):
        return pt.zeros(size)
    elif has_fallback:
        return pt.zeros_like(rv)
    else:
        raise TypeError(
            "Cannot safely infer the size of a multivariate random variable's support_point. "
            f"Please provide a support_point function when instantiating the {rv_name} "
            "random variable."
        )


class CustomDistRV(RandomVariable):
    """
    Base class for CustomDistRV.

    This should be subclassed when defining CustomDist objects.
    """

    name = "CustomDistRV"
    _print_name = ("CustomDist", "\\operatorname{CustomDist}")

    @classmethod
    def rng_fn(cls, rng, *args):
        args = list(args)
        size = args.pop(-1)
        return cls._random_fn(*args, rng=rng, size=size)


class _CustomDist(Distribution):
    """A distribution that returns a subclass of CustomDistRV."""

    rv_type = CustomDistRV

    @classmethod
    def dist(
        cls,
        *dist_params,
        logp: Callable | None = None,
        logcdf: Callable | None = None,
        random: Callable | None = None,
        support_point: Callable | None = None,
        ndim_supp: int | None = None,
        ndims_params: Sequence[int] | None = None,
        signature: str | None = None,
        dtype: str = "floatX",
        class_name: str = "CustomDist",
        **kwargs,
    ):
        if ndim_supp is None and signature is None:
            # Assume a scalar distribution
            signature = safe_signature([0] * len(dist_params), [0])

        dist_params = [as_tensor_variable(param) for param in dist_params]

        if logp is None:
            logp = default_not_implemented(class_name, "logp")

        if logcdf is None:
            logcdf = default_not_implemented(class_name, "logcdf")

        if support_point is None:
            support_point = functools.partial(
                default_support_point,
                rv_name=class_name,
                has_fallback=random is not None,
            )

        if random is None:
            random = default_not_implemented(class_name, "random")

        return super().dist(
            dist_params,
            logp=logp,
            logcdf=logcdf,
            random=random,
            support_point=support_point,
            ndim_supp=ndim_supp,
            ndims_params=ndims_params,
            signature=signature,
            dtype=dtype,
            class_name=class_name,
            **kwargs,
        )

    @classmethod
    def rv_op(
        cls,
        *dist_params,
        logp: Callable | None,
        logcdf: Callable | None,
        random: Callable | None,
        support_point: Callable | None,
        signature: str | None,
        ndim_supp: int | None,
        ndims_params: Sequence[int] | None,
        dtype: str,
        class_name: str,
        **kwargs,
    ):
        rv_type = type(
            class_name,
            (CustomDistRV,),
            {
                "name": class_name,
                "inplace": False,
                "ndim_supp": ndim_supp,
                "ndims_params": ndims_params,
                "signature": signature,
                "dtype": dtype,
                "_print_name": (class_name, f"\\operatorname{{{class_name}}}"),
                # Specific to CustomDist
                "_random_fn": random,
            },
        )

        # Dispatch custom methods
        @_logprob.register(rv_type)
        def custom_dist_logp(op, values, rng, size, *dist_params, **kwargs):
            return logp(values[0], *dist_params)

        @_logcdf.register(rv_type)
        def custom_dist_logcdf(op, value, rng, size, *dist_params, **kwargs):
            return logcdf(value, *dist_params, **kwargs)

        @_support_point.register(rv_type)
        def custom_dist_support_point(op, rv, rng, size, *dist_params):
            return support_point(rv, size, *dist_params)

        rv_op = rv_type()
        return rv_op(*dist_params, **kwargs)


class CustomSymbolicDistRV(SymbolicRandomVariable):
    """
    Base class for CustomSymbolicDist.

    This should be subclassed when defining custom CustomDist objects that have
    symbolic random methods.
    """

    default_output = 0

    _print_name = ("CustomSymbolicDist", "\\operatorname{CustomSymbolicDist}")


class _CustomSymbolicDist(Distribution):
    rv_type = CustomSymbolicDistRV

    @classmethod
    def dist(
        cls,
        *dist_params,
        dist: Callable,
        logp: Callable | None = None,
        logcdf: Callable | None = None,
        support_point: Callable | None = None,
        ndim_supp: int | None = None,
        ndims_params: Sequence[int] | None = None,
        signature: str | None = None,
        dtype: str = "floatX",
        class_name: str = "CustomDist",
        **kwargs,
    ):
        dist_params = [as_tensor_variable(param) for param in dist_params]

        if logcdf is None:
            logcdf = default_not_implemented(class_name, "logcdf")

        if signature is None:
            if ndim_supp is None:
                ndim_supp = 0
            if ndims_params is None:
                ndims_params = [0] * len(dist_params)
            signature = safe_signature(
                core_inputs_ndim=ndims_params,
                core_outputs_ndim=[ndim_supp],
            )

        return super().dist(
            dist_params,
            class_name=class_name,
            logp=logp,
            logcdf=logcdf,
            dist=dist,
            support_point=support_point,
            signature=signature,
            **kwargs,
        )

    @classmethod
    def rv_op(
        cls,
        *dist_params,
        dist: Callable,
        logp: Callable | None,
        logcdf: Callable | None,
        support_point: Callable | None,
        size=None,
        signature: str,
        class_name: str,
    ):
        size = normalize_size_param(size)
        # If it's NoneConst, just use that as the dummy
        dummy_size_param = size.type() if isinstance(size, TensorVariable) else size
        dummy_dist_params = [dist_param.type() for dist_param in dist_params]
        with new_or_existing_block_model_access(
            error_msg_on_access="Model variables cannot be created in the dist function. Use the `.dist` API"
        ):
            dummy_rv = dist(*dummy_dist_params, dummy_size_param)
        dummy_params = [dummy_size_param, *dummy_dist_params]
        # RNGs are not passed as explicit inputs (because we usually don't know how many are needed)
        # We retrieve them here. This will also raise if the user forgot to specify some update in a Scan Op
        dummy_updates_dict = collect_default_updates(inputs=dummy_params, outputs=(dummy_rv,))

        rv_type = type(
            class_name,
            (CustomSymbolicDistRV,),
            # If logp is not provided, we try to infer it from the dist graph
            {
                "inline_logprob": logp is None,
                "_print_name": (class_name, f"\\operatorname{{{class_name}}}"),
            },
        )

        # Dispatch custom methods
        if logp is not None:

            @_logprob.register(rv_type)
            def custom_dist_logp(op, values, size, *inputs, **kwargs):
                [value] = values
                rv_params = inputs[: len(dist_params)]
                return logp(value, *rv_params)

        if logcdf is not None:

            @_logcdf.register(rv_type)
            def custom_dist_logcdf(op, value, size, *inputs, **kwargs):
                rv_params = inputs[: len(dist_params)]
                return logcdf(value, *rv_params)

        if support_point is not None:

            @_support_point.register(rv_type)
            def custom_dist_support_point(op, rv, size, *params):
                return support_point(
                    rv,
                    size,
                    *[
                        p
                        for p in params
                        if not isinstance(p.type, RandomType | RandomGeneratorType)
                    ],
                )

        @_change_dist_size.register(rv_type)
        def change_custom_dist_size(op, rv, new_size, expand):
            node = rv.owner

            if expand:
                shape = tuple(rv.shape)
                old_size = shape[: len(shape) - node.op.ndim_supp]
                new_size = tuple(new_size) + tuple(old_size)
            new_size = pt.as_tensor(new_size, dtype="int64", ndim=1)

            old_size, *old_dist_params = node.inputs[: len(dist_params) + 1]

            # OpFromGraph has to be recreated if the size type changes!
            dummy_size_param = new_size.type()
            dummy_dist_params = [dist_param.type() for dist_param in old_dist_params]
            dummy_rv = dist(*dummy_dist_params, dummy_size_param)
            dummy_params = [dummy_size_param, *dummy_dist_params]
            updates_dict = collect_default_updates(inputs=dummy_params, outputs=(dummy_rv,))
            rngs = updates_dict.keys()
            rngs_updates = updates_dict.values()
            new_rv_op = rv_type(
                inputs=[*dummy_params, *rngs],
                outputs=[dummy_rv, *rngs_updates],
                extended_signature=extended_signature,
            )
            new_rv = new_rv_op(new_size, *dist_params, *rngs)

            return new_rv

        if dummy_updates_dict:
            rngs, rngs_updates = zip(*dummy_updates_dict.items())
        else:
            rngs, rngs_updates = (), ()

        inputs = [*dummy_params, *rngs]
        outputs = [dummy_rv, *rngs_updates]
        extended_signature = cls._infer_final_signature(
            signature, n_inputs=len(inputs), n_outputs=len(outputs), n_rngs=len(rngs)
        )
        rv_op = rv_type(
            inputs=inputs,
            outputs=outputs,
            extended_signature=extended_signature,
        )
        return rv_op(size, *dist_params, *rngs)

    @staticmethod
    def _infer_final_signature(signature: str, n_inputs, n_outputs, n_rngs) -> str:
        """Add size and updates to user provided gufunc signature if they are missing."""
        # Regex to split across outer commas
        # Copied from https://stackoverflow.com/a/26634150
        outer_commas = re.compile(r",\s*(?![^()]*\))")

        input_sig, output_sig = signature.split("->")
        # It's valid to have a signature without params inputs, as in a Flat RV
        n_inputs_sig = len(outer_commas.split(input_sig)) if input_sig.strip() else 0
        n_outputs_sig = len(outer_commas.split(output_sig))

        if n_inputs_sig == n_inputs and n_outputs_sig == n_outputs:
            # User provided a signature with no missing parts
            return signature

        size_sig = "[size]"
        rngs_sig = ("[rng]",) * n_rngs
        if n_inputs_sig == (n_inputs - n_rngs - 1):
            # Assume size and rngs are missing
            if input_sig.strip():
                input_sig = ",".join((size_sig, input_sig, *rngs_sig))
            else:
                input_sig = ",".join((size_sig, *rngs_sig))
        if n_outputs_sig == (n_outputs - n_rngs):
            # Assume updates are missing
            output_sig = ",".join((output_sig, *rngs_sig))
        signature = "->".join((input_sig, output_sig))
        return signature


class SupportPointRewrite(GraphRewriter):
    def rewrite_support_point_scan_node(self, node):
        if not isinstance(node.op, Scan):
            return

        node_inputs, node_outputs = node.op.inner_inputs, node.op.inner_outputs
        op = node.op

        local_fgraph_topo = io_toposort(node_inputs, node_outputs)

        replace_with_support_point = []
        to_replace_set = set()

        for nd in local_fgraph_topo:
            if nd not in to_replace_set and isinstance(
                nd.op, RandomVariable | SymbolicRandomVariable
            ):
                replace_with_support_point.append(nd.out)
                to_replace_set.add(nd)
        givens = {}
        if len(replace_with_support_point) > 0:
            for item in replace_with_support_point:
                givens[item] = support_point(item)
        else:
            return
        op_outs = clone_replace(node_outputs, replace=givens)

        nwScan = Scan(
            node_inputs,
            op_outs,
            op.info,
            mode=op.mode,
            profile=op.profile,
            truncate_gradient=op.truncate_gradient,
            name=op.name,
            allow_gc=op.allow_gc,
        )
        nw_node = nwScan(*(node.inputs), return_list=True)[0].owner
        return nw_node

    def add_requirements(self, fgraph):
        fgraph.attach_feature(ReplaceValidate())

    def apply(self, fgraph):
        for node in fgraph.toposort():
            if isinstance(node.op, RandomVariable | SymbolicRandomVariable):
                fgraph.replace(node.out, support_point(node.out))
            elif isinstance(node.op, Scan):
                new_node = self.rewrite_support_point_scan_node(node)
                if new_node is not None:
                    fgraph.replace_all(tuple(zip(node.outputs, new_node.outputs)))


@_support_point.register(CustomSymbolicDistRV)
def dist_support_point(op, rv, *args):
    node = rv.owner
    rv_out_idx = node.outputs.index(rv)

    fgraph = op.fgraph.clone()
    replace_support_point = SupportPointRewrite()
    replace_support_point.rewrite(fgraph)
    # Replace dummy inner inputs by outer inputs
    fgraph.replace_all(tuple(zip(op.inner_inputs, args)), import_missing=True)
    support_point = fgraph.outputs[rv_out_idx]
    return support_point


class CustomDist:
    """A helper class to create custom distributions.

    This class can be used to wrap black-box random and logp methods for use in
    forward and mcmc sampling.

    A user can provide a `dist` function that returns a PyTensor graph built from
    simpler PyMC distributions, which represents the distribution. This graph is
    used to take random draws, and to infer the logp expression automatically
    when not provided by the user.

    Alternatively, a user can provide a `random` function that returns numerical
    draws (e.g., via NumPy routines), and a `logp` function that must return a
    PyTensor graph that represents the logp graph when evaluated. This is used for
    mcmc sampling.

    Additionally, a user can provide a `logcdf` and `support_point` functions that must return
    PyTensor graphs that computes those quantities. These may be used by other PyMC
    routines.

    Parameters
    ----------
    name : str
    dist_params : Tuple
        A sequence of the distribution's parameter. These will be converted into
        Pytensor tensor variables internally.
    dist: Optional[Callable]
        A callable that returns a PyTensor graph built from simpler PyMC distributions
        which represents the distribution. This can be used by PyMC to take random draws
        as well as to infer the logp of the distribution in some cases. In that case
        it's not necessary to implement ``random`` or ``logp`` functions.

        It must have the following signature: ``dist(*dist_params, size)``.
        The symbolic tensor distribution parameters are passed as positional arguments in
        the same order as they are supplied when the ``CustomDist`` is constructed.

    random : Optional[Callable]
        A callable that can be used to generate random draws from the distribution

        It must have the following signature: ``random(*dist_params, rng=None, size=None)``.
        The numerical distribution parameters are passed as positional arguments in the
        same order as they are supplied when the ``CustomDist`` is constructed.
        The keyword arguments are ``rng``, which will provide the random variable's
        associated :py:class:`~numpy.random.Generator`, and ``size``, that will represent
        the desired size of the random draw. If ``None``, a ``NotImplemented``
        error will be raised when trying to draw random samples from the distribution's
        prior or posterior predictive.

    logp : Optional[Callable]
        A callable that calculates the log probability of some given ``value``
        conditioned on certain distribution parameter values. It must have the
        following signature: ``logp(value, *dist_params)``, where ``value`` is
        a PyTensor tensor that represents the distribution value, and ``dist_params``
        are the tensors that hold the values of the distribution parameters.
        This function must return a PyTensor tensor.

        When the `dist` function is specified, PyMC will try to automatically
        infer the `logp` when this is not provided.

        Otherwise, a ``NotImplementedError`` will be raised when trying to compute the
        distribution's logp.
    logcdf : Optional[Callable]
        A callable that calculates the log cumulative log probability of some given
        ``value`` conditioned on certain distribution parameter values. It must have the
        following signature: ``logcdf(value, *dist_params)``, where ``value`` is
        a PyTensor tensor that represents the distribution value, and ``dist_params``
        are the tensors that hold the values of the distribution parameters.
        This function must return a PyTensor tensor. If ``None``, a ``NotImplementedError``
        will be raised when trying to compute the distribution's logcdf.
    support_point : Optional[Callable]
        A callable that can be used to compute the finete logp point of the distribution.
        It must have the following signature: ``support_point(rv, size, *rv_inputs)``.
        The distribution's variable is passed as the first argument ``rv``. ``size``
        is the random variable's size implied by the ``dims``, ``size`` and parameters
        supplied to the distribution. Finally, ``rv_inputs`` is the sequence of the
        distribution parameters, in the same order as they were supplied when the
        CustomDist was created. If ``None``, a default  ``support_point`` function will be
        assigned that will always return 0, or an array of zeros.
    ndim_supp : Optional[int]
        The number of dimensions in the support of the distribution.
        Inferred from signature, if provided. Defaults to assuming
        a scalar distribution, i.e. ``ndim_supp = 0``
    ndims_params : Optional[Sequence[int]]
        The list of number of dimensions in the support of each of the distribution's
        parameters. Inferred from signature, if provided. Defaults to assuming
        all parameters are scalars, i.e. ``ndims_params=[0, ...]``.
    signature : Optional[str]
        A numpy vectorize-like signature that indicates the number and core dimensionality
        of the input parameters and sample outputs of the CustomDist.
        When specified, `ndim_supp` and `ndims_params` are not needed. See examples below.
    dtype : str
        The dtype of the distribution. All draws and observations passed into the
        distribution will be cast onto this dtype. This is not needed if a PyTensor
        dist function is provided, which should already return the right dtype!
    class_name : str
        Name for the class which will wrap the CustomDist methods. When not specified,
        it will be given the name of the model variable.
    kwargs :
        Extra keyword arguments are passed to the parent's class ``__new__`` method.


    Examples
    --------
    Create a CustomDist that wraps a black-box logp function. This variable cannot be
    used in prior or posterior predictive sampling because no random function was provided

    .. code-block:: python

        import numpy as np
        import pymc as pm
        from pytensor.tensor import TensorVariable


        def logp(value: TensorVariable, mu: TensorVariable) -> TensorVariable:
            return -((value - mu) ** 2)


        with pm.Model():
            mu = pm.Normal("mu", 0, 1)
            pm.CustomDist(
                "custom_dist",
                mu,
                logp=logp,
                observed=np.random.randn(100),
            )
            idata = pm.sample(100)

    Provide a random function that return numerical draws. This allows one to use a
    CustomDist in prior and posterior predictive sampling.
    A gufunc signature was also provided, which may be used by other routines.

    .. code-block:: python

        from typing import Optional, Tuple

        import numpy as np
        import pymc as pm
        from pytensor.tensor import TensorVariable


        def logp(value: TensorVariable, mu: TensorVariable) -> TensorVariable:
            return -((value - mu) ** 2)


        def random(
            mu: np.ndarray | float,
            rng: Optional[np.random.Generator] = None,
            size: Optional[Tuple[int]] = None,
        ) -> np.ndarray | float:
            return rng.normal(loc=mu, scale=1, size=size)


        with pm.Model():
            mu = pm.Normal("mu", 0, 1)
            pm.CustomDist(
                "custom_dist",
                mu,
                logp=logp,
                random=random,
                signature="()->()",
                observed=np.random.randn(100, 3),
                size=(100, 3),
            )
            prior = pm.sample_prior_predictive(10)

    Provide a dist function that creates a PyTensor graph built from other
    PyMC distributions. PyMC can automatically infer that the logp of this
    variable corresponds to a shifted Exponential distribution.
    A gufunc signature was also provided, which may be used by other routines.

    .. code-block:: python

        import pymc as pm
        from pytensor.tensor import TensorVariable


        def dist(
            lam: TensorVariable,
            shift: TensorVariable,
            size: TensorVariable,
        ) -> TensorVariable:
            return pm.Exponential.dist(lam, size=size) + shift


        with pm.Model() as m:
            lam = pm.HalfNormal("lam")
            shift = -1
            pm.CustomDist(
                "custom_dist",
                lam,
                shift,
                dist=dist,
                signature="(),()->()",
                observed=[-1, -1, 0],
            )

            prior = pm.sample_prior_predictive()
            posterior = pm.sample()

    Provide a dist function that creates a PyTensor graph built from other
    PyMC distributions. PyMC can automatically infer that the logp of
    this variable corresponds to a modified-PERT distribution.

    .. code-block:: python

       import pymc as pm
       from pytensor.tensor import TensorVariable

        def pert(
            low: TensorVariable,
            peak: TensorVariable,
            high: TensorVariable,
            lmbda: TensorVariable,
            size: TensorVariable,
        ) -> TensorVariable:
            range = (high - low)
            s_alpha = 1 + lmbda * (peak - low) / range
            s_beta = 1 + lmbda * (high - peak) / range
            return pm.Beta.dist(s_alpha, s_beta, size=size) * range + low

        with pm.Model() as m:
            low = pm.Normal("low", 0, 10)
            peak = pm.Normal("peak", 50, 10)
            high = pm.Normal("high", 100, 10)
            lmbda = 4
            pm.CustomDist("pert", low, peak, high, lmbda, dist=pert, observed=[30, 35, 73])

        m.point_logps()

    """

    def __new__(
        cls,
        name,
        *dist_params,
        dist: Callable | None = None,
        random: Callable | None = None,
        logp: Callable | None = None,
        logcdf: Callable | None = None,
        support_point: Callable | None = None,
        # TODO: Deprecate ndim_supp / ndims_params in favor of signature?
        ndim_supp: int | None = None,
        ndims_params: Sequence[int] | None = None,
        signature: str | None = None,
        dtype: str = "floatX",
        **kwargs,
    ):
        if isinstance(kwargs.get("observed", None), dict):
            raise TypeError(
                "Since ``v4.0.0`` the ``observed`` parameter should be of type"
                " ``pd.Series``, ``np.array``, or ``pm.Data``."
                " Previous versions allowed passing distribution parameters as"
                " a dictionary in ``observed``, in the current version these "
                "parameters are positional arguments."
            )
        dist_params = cls.parse_dist_params(dist_params)
        cls.check_valid_dist_random(dist, random, dist_params)
        moment = kwargs.pop("moment", None)
        if moment is not None:
            warnings.warn(
                "`moment` argument is deprecated. Use `support_point` instead.",
                FutureWarning,
            )
            support_point = moment
        if dist is not None:
            kwargs.setdefault("class_name", f"CustomDist_{name}")
            return _CustomSymbolicDist(
                name,
                *dist_params,
                dist=dist,
                logp=logp,
                logcdf=logcdf,
                support_point=support_point,
                ndim_supp=ndim_supp,
                ndims_params=ndims_params,
                signature=signature,
                **kwargs,
            )
        else:
            kwargs.setdefault("class_name", f"CustomDist_{name}")
            return _CustomDist(
                name,
                *dist_params,
                random=random,
                logp=logp,
                logcdf=logcdf,
                support_point=support_point,
                ndim_supp=ndim_supp,
                ndims_params=ndims_params,
                signature=signature,
                dtype=dtype,
                **kwargs,
            )

    @classmethod
    def dist(
        cls,
        *dist_params,
        dist: Callable | None = None,
        random: Callable | None = None,
        logp: Callable | None = None,
        logcdf: Callable | None = None,
        support_point: Callable | None = None,
        ndim_supp: int | None = None,
        ndims_params: Sequence[int] | None = None,
        signature: str | None = None,
        dtype: str = "floatX",
        **kwargs,
    ):
        dist_params = cls.parse_dist_params(dist_params)
        cls.check_valid_dist_random(dist, random, dist_params)
        if dist is not None:
            return _CustomSymbolicDist.dist(
                *dist_params,
                dist=dist,
                logp=logp,
                logcdf=logcdf,
                support_point=support_point,
                ndim_supp=ndim_supp,
                ndims_params=ndims_params,
                signature=signature,
                **kwargs,
            )
        else:
            return _CustomDist.dist(
                *dist_params,
                random=random,
                logp=logp,
                logcdf=logcdf,
                support_point=support_point,
                ndim_supp=ndim_supp,
                ndims_params=ndims_params,
                signature=signature,
                dtype=dtype,
                **kwargs,
            )

    @classmethod
    def parse_dist_params(cls, dist_params):
        if len(dist_params) > 0 and callable(dist_params[0]):
            raise TypeError(
                "The DensityDist API has changed, you are using the old API "
                "where logp was the first positional argument. In the current API, "
                "the logp is a keyword argument, amongst other changes. Please refer "
                "to the API documentation for more information on how to use the "
                "new DensityDist API."
            )
        return [as_tensor_variable(param) for param in dist_params]

    @classmethod
    def check_valid_dist_random(cls, dist, random, dist_params):
        if dist is not None and random is not None:
            raise ValueError("Cannot provide both dist and random functions")
        if random is not None and cls.is_symbolic_random(random, dist_params):
            raise TypeError(
                "API change: function passed to `random` argument should no longer return a PyTensor graph. "
                "Pass such function to the `dist` argument instead."
            )

    @classmethod
    def is_symbolic_random(self, random, dist_params):
        if random is None:
            return False
        # Try calling random with symbolic inputs
        try:
            size = normalize_size_param(None)
            with new_or_existing_block_model_access(
                error_msg_on_access="Model variables cannot be created in the random function. Use the `.dist` API to create such variables."
            ):
                out = random(*dist_params, size)
        except BlockModelAccessError:
            raise
        except Exception:
            # If it fails we assume it was not
            return False
        # Confirm the output is symbolic
        return isinstance(out, Variable)


DensityDist = CustomDist
