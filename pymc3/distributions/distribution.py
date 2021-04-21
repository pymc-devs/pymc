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
import contextvars
import inspect
import multiprocessing
import sys
import types
import warnings

from abc import ABCMeta
from copy import copy
from typing import Any, Optional, Sequence, Tuple, Union

import aesara
import aesara.tensor as at
import dill

from aesara.graph.basic import Variable
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.shape import SpecifyShape, specify_shape

from pymc3.aesaraf import change_rv_size, pandas_to_array
from pymc3.distributions import _logcdf, _logp
from pymc3.util import UNSET, get_repr_for_variable
from pymc3.vartypes import string_types

__all__ = [
    "DensityDist",
    "Distribution",
    "Continuous",
    "Discrete",
    "NoDistribution",
]

vectorized_ppc = contextvars.ContextVar(
    "vectorized_ppc", default=None
)  # type: contextvars.ContextVar[Optional[Callable]]

PLATFORM = sys.platform

Shape = Union[int, Sequence[Union[str, type(Ellipsis)]], Variable]
Dims = Union[str, Sequence[Union[str, None, type(Ellipsis)]]]
Size = Union[int, Tuple[int, ...]]


class _Unpickling:
    pass


class DistributionMeta(ABCMeta):
    def __new__(cls, name, bases, clsdict):

        # Forcefully deprecate old v3 `Distribution`s
        if "random" in clsdict:

            def _random(*args, **kwargs):
                warnings.warn(
                    "The old `Distribution.random` interface is deprecated.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                return clsdict["random"](*args, **kwargs)

            clsdict["random"] = _random

        rv_op = clsdict.setdefault("rv_op", None)
        rv_type = None

        if isinstance(rv_op, RandomVariable):
            if not rv_op.inplace:
                # TODO: This is a temporary work-around.
                # Remove this once we know what we want regarding RNG states
                # and their propagation.
                rv_op = copy(rv_op)
                rv_op.inplace = True
                clsdict["rv_op"] = rv_op

            rv_type = type(rv_op)

        new_cls = super().__new__(cls, name, bases, clsdict)

        if rv_type is not None:
            # Create dispatch functions

            class_logp = clsdict.get("logp")
            if class_logp:

                @_logp.register(rv_type)
                def logp(op, var, rvs_to_values, *dist_params, **kwargs):
                    value_var = rvs_to_values.get(var, var)
                    return class_logp(value_var, *dist_params, **kwargs)

            class_logcdf = clsdict.get("logcdf")
            if class_logcdf:

                @_logcdf.register(rv_type)
                def logcdf(op, var, rvs_to_values, *dist_params, **kwargs):
                    value_var = rvs_to_values.get(var, var)
                    return class_logcdf(value_var, *dist_params, **kwargs)

            # class_transform = clsdict.get("transform")
            # if class_transform:
            #
            #     @logp_transform.register(rv_type)
            #     def transform(op, *args, **kwargs):
            #         return class_transform(*args, **kwargs)

            # Register the Aesara `RandomVariable` type as a subclass of this
            # `Distribution` type.
            new_cls.register(rv_type)

        return new_cls


def _valid_ellipsis_position(items: Union[None, Shape, Dims, Size]) -> bool:
    if items is not None and not isinstance(items, Variable) and Ellipsis in items:
        if any(i == Ellipsis for i in items[:-1]):
            return False
    return True


def _validate_shape_dims_size(
    shape: Any = None, dims: Any = None, size: Any = None
) -> Tuple[Optional[Shape], Optional[Dims], Optional[Size]]:
    # Raise on unsupported parametrization
    if shape is not None and dims is not None:
        raise ValueError(f"Passing both `shape` ({shape}) and `dims` ({dims}) is not supported!")
    if dims is not None and size is not None:
        raise ValueError(f"Passing both `dims` ({dims}) and `size` ({size}) is not supported!")
    if shape is not None and size is not None:
        raise ValueError(f"Passing both `shape` ({shape}) and `size` ({size}) is not supported!")

    # Raise on invalid types
    if not isinstance(shape, (type(None), int, list, tuple, Variable)):
        raise ValueError("The `shape` parameter must be an int, list or tuple.")
    if not isinstance(dims, (type(None), str, list, tuple)):
        raise ValueError("The `dims` parameter must be a str, list or tuple.")
    if not isinstance(size, (type(None), int, list, tuple)):
        raise ValueError("The `size` parameter must be an int, list or tuple.")

    # Auto-convert non-tupled parameters
    if isinstance(shape, int):
        shape = (shape,)
    if isinstance(dims, str):
        dims = (dims,)
    if isinstance(size, int):
        size = (size,)

    # Convert to actual tuples
    if not isinstance(shape, (type(None), tuple, Variable)):
        shape = tuple(shape)
    if not isinstance(dims, (type(None), tuple)):
        dims = tuple(dims)
    if not isinstance(size, (type(None), tuple)):
        size = tuple(size)

    if not _valid_ellipsis_position(shape):
        raise ValueError(
            f"Ellipsis in `shape` may only appear in the last position. Actual: {shape}"
        )
    if not _valid_ellipsis_position(dims):
        raise ValueError(f"Ellipsis in `dims` may only appear in the last position. Actual: {dims}")
    if size is not None and Ellipsis in size:
        raise ValueError(f"The `size` parameter cannot contain an Ellipsis. Actual: {size}")
    return shape, dims, size


class Distribution(metaclass=DistributionMeta):
    """Statistical distribution"""

    rv_class = None
    rv_op = None

    def __new__(
        cls,
        name: str,
        *args,
        rng=None,
        dims: Optional[Dims] = None,
        testval=None,
        observed=None,
        total_size=None,
        transform=UNSET,
        **kwargs,
    ) -> RandomVariable:
        """Adds a RandomVariable corresponding to a PyMC3 distribution to the current model.

        Note that all remaining kwargs must be compatible with ``.dist()``

        Parameters
        ----------
        cls : type
            A PyMC3 distribution.
        name : str
            Name for the new model variable.
        rng : optional
            Random number generator to use with the RandomVariable.
        dims : tuple, optional
            A tuple of dimension names known to the model.
        testval : optional
            Test value to be attached to the output RV.
            Must match its shape exactly.
        observed : optional
            Observed data to be passed when registering the random variable in the model.
            See ``Model.register_rv``.
        total_size : float, optional
            See ``Model.register_rv``.
        transform : optional
            See ``Model.register_rv``.
        **kwargs
            Keyword arguments that will be forwarded to ``.dist()``.
            Most prominently: ``shape`` and ``size``

        Returns
        -------
        rv : RandomVariable
            The created RV, registered in the Model.
        """

        try:
            from pymc3.model import Model

            model = Model.get_context()
        except TypeError:
            raise TypeError(
                "No model on context stack, which is needed to "
                "instantiate distributions. Add variable inside "
                "a 'with model:' block, or use the '.dist' syntax "
                "for a standalone distribution."
            )

        if not isinstance(name, string_types):
            raise TypeError(f"Name needs to be a string but got: {name}")

        if rng is None:
            rng = model.default_rng

        _, dims, _ = _validate_shape_dims_size(dims=dims)
        resize = None

        # Create the RV without specifying testval, because the testval may have a shape
        # that only matches after replicating with a size implied by dims (see below).
        rv_out = cls.dist(*args, rng=rng, testval=None, **kwargs)
        n_implied = rv_out.ndim

        # The `.dist()` can wrap automatically with a SpecifyShape Op which brings informative
        # error messages earlier in model construction.
        # Here, however, the underyling RV must be used - a new SpecifyShape Op can be added at the end.
        assert_shape = None
        if isinstance(rv_out.owner.op, SpecifyShape):
            rv_out, assert_shape = rv_out.owner.inputs

        # `dims` are only available with this API, because `.dist()` can be used
        # without a modelcontext and dims are not tracked at the Aesara level.
        if dims is not None:
            if Ellipsis in dims:
                # Auto-complete the dims tuple to the full length
                dims = (*dims[:-1], *[None] * rv_out.ndim)

            n_resize = len(dims) - n_implied

            # All resize dims must be known already (numerically or symbolically).
            unknown_resize_dims = set(dims[:n_resize]) - set(model.dim_lengths)
            if unknown_resize_dims:
                raise KeyError(
                    f"Dimensions {unknown_resize_dims} are unknown to the model and cannot be used to specify a `size`."
                )

            # The numeric/symbolic resize tuple can be created using model.RV_dim_lengths
            resize = tuple(model.dim_lengths[dname] for dname in dims[:n_resize])
        elif observed is not None:
            if not hasattr(observed, "shape"):
                observed = pandas_to_array(observed)
            n_resize = observed.ndim - n_implied
            resize = tuple(observed.shape[d] for d in range(n_resize))

        if resize:
            # A batch size was specified through `dims`, or implied by `observed`.
            rv_out = change_rv_size(rv_var=rv_out, new_size=resize, expand=True)

        if dims is not None:
            # Now that we have a handle on the output RV, we can register named implied dimensions that
            # were not yet known to the model, such that they can be used for size further downstream.
            for di, dname in enumerate(dims[n_resize:]):
                if not dname in model.dim_lengths:
                    model.add_coord(dname, values=None, length=rv_out.shape[n_resize + di])

        if testval is not None:
            # Assigning the testval earlier causes trouble because the RV may not be created with the final shape already.
            rv_out.tag.test_value = testval

        rv_registered = model.register_rv(
            rv_out, name, observed, total_size, dims=dims, transform=transform
        )

        # Wrapping in specify_shape now does not break transforms:
        if assert_shape is not None:
            rv_registered = specify_shape(rv_registered, assert_shape)

        return rv_registered

    @classmethod
    def dist(
        cls,
        dist_params,
        *,
        shape: Optional[Shape] = None,
        size: Optional[Size] = None,
        testval=None,
        **kwargs,
    ) -> RandomVariable:
        """Creates a RandomVariable corresponding to the `cls` distribution.

        Parameters
        ----------
        dist_params
        shape : tuple, optional
            A tuple of sizes for each dimension of the new RV.

            Ellipsis (...) may be used in the last position of the tuple,
            and automatically expand to the shape implied by RV inputs.

            Without Ellipsis, a `SpecifyShape` Op is automatically applied,
            constraining this model variable to exactly the specified shape.
        size : int, tuple, Variable, optional
            A scalar or tuple for replicating the RV in addition
            to its implied shape/dimensionality.
        testval : optional
            Test value to be attached to the output RV.
            Must match its shape exactly.

        Returns
        -------
        rv : RandomVariable
            The created RV.
        """
        if "dims" in kwargs:
            raise NotImplementedError("The use of a `.dist(dims=...)` API is not yet supported.")

        shape, _, size = _validate_shape_dims_size(shape=shape, size=size)
        assert_shape = None

        # Create the RV without specifying size or testval.
        # The size will be expanded later (if necessary) and only then the testval fits.
        rv_native = cls.rv_op(*dist_params, size=None, **kwargs)

        if shape is None and size is None:
            size = ()
        elif shape is not None:
            # SpecifyShape is automatically applied for symbolic and non-Ellipsis shapes
            if isinstance(shape, Variable):
                assert_shape = shape
                size = ()
            else:
                if Ellipsis in shape:
                    size = tuple(shape[:-1])
                else:
                    size = tuple(shape[: len(shape) - rv_native.ndim])
                    assert_shape = shape
        # no-op conditions:
        # `elif size is not None` (User already specified how to expand the RV)
        # `else` (Unreachable)

        if size:
            rv_out = change_rv_size(rv_var=rv_native, new_size=size, expand=True)
        else:
            rv_out = rv_native

        if assert_shape is not None:
            rv_out = specify_shape(rv_out, shape=assert_shape)

        if testval is not None:
            rv_out.tag.test_value = testval

        return rv_out

    def _distr_parameters_for_repr(self):
        """Return the names of the parameters for this distribution (e.g. "mu"
        and "sigma" for Normal). Used in generating string (and LaTeX etc.)
        representations of Distribution objects. By default based on inspection
        of __init__, but can be overwritten if necessary (e.g. to avoid including
        "sd" and "tau").
        """
        return inspect.getfullargspec(self.__init__).args[1:]

    def _distr_name_for_repr(self):
        return self.__class__.__name__

    def _str_repr(self, name=None, dist=None, formatting="plain"):
        """
        Generate string representation for this distribution, optionally
        including LaTeX markup (formatting='latex').

        Parameters
        ----------
        name : str
            name of the distribution
        dist : Distribution
            the distribution object
        formatting : str
            one of { "latex", "plain", "latex_with_params", "plain_with_params" }
        """
        if dist is None:
            dist = self
        if name is None:
            name = "[unnamed]"
        supported_formattings = {"latex", "plain", "latex_with_params", "plain_with_params"}
        if not formatting in supported_formattings:
            raise ValueError(f"Unsupported formatting ''. Choose one of {supported_formattings}.")

        param_names = self._distr_parameters_for_repr()
        param_values = [
            get_repr_for_variable(getattr(dist, x), formatting=formatting) for x in param_names
        ]

        if "latex" in formatting:
            param_string = ",~".join(
                [fr"\mathit{{{name}}}={value}" for name, value in zip(param_names, param_values)]
            )
            if formatting == "latex_with_params":
                return r"$\text{{{var_name}}} \sim \text{{{distr_name}}}({params})$".format(
                    var_name=name, distr_name=dist._distr_name_for_repr(), params=param_string
                )
            return r"$\text{{{var_name}}} \sim \text{{{distr_name}}}$".format(
                var_name=name, distr_name=dist._distr_name_for_repr()
            )
        else:
            # one of the plain formattings
            param_string = ", ".join(
                [f"{name}={value}" for name, value in zip(param_names, param_values)]
            )
            if formatting == "plain_with_params":
                return f"{name} ~ {dist._distr_name_for_repr()}({param_string})"
            return f"{name} ~ {dist._distr_name_for_repr()}"

    def __str__(self, **kwargs):
        try:
            return self._str_repr(formatting="plain", **kwargs)
        except:
            return super().__str__()

    def _repr_latex_(self, *, formatting="latex_with_params", **kwargs):
        """Magic method name for IPython to use for LaTeX formatting."""
        return self._str_repr(formatting=formatting, **kwargs)

    __latex__ = _repr_latex_


class NoDistribution(Distribution):
    def __init__(
        self,
        shape,
        dtype,
        testval=None,
        defaults=(),
        parent_dist=None,
        *args,
        **kwargs,
    ):
        super().__init__(
            shape=shape, dtype=dtype, testval=testval, defaults=defaults, *args, **kwargs
        )
        self.parent_dist = parent_dist

    def __getattr__(self, name):
        # Do not use __getstate__ and __setstate__ from parent_dist
        # to avoid infinite recursion during unpickling
        if name.startswith("__"):
            raise AttributeError("'NoDistribution' has no attribute '%s'" % name)
        return getattr(self.parent_dist, name)

    def logp(self, x):
        """Calculate log probability.

        Parameters
        ----------
        x: numeric
            Value for which log-probability is calculated.

        Returns
        -------
        TensorVariable
        """
        return at.zeros_like(x)

    def _distr_parameters_for_repr(self):
        return []


class Discrete(Distribution):
    """Base class for discrete distributions"""

    def __new__(cls, name, *args, **kwargs):

        if kwargs.get("transform", None):
            raise ValueError("Transformations for discrete distributions")

        return super().__new__(cls, name, *args, **kwargs)


class Continuous(Distribution):
    """Base class for continuous distributions"""


class DensityDist(Distribution):
    """Distribution based on a given log density function.

    A distribution with the passed log density function is created.
    Requires a custom random function passed as kwarg `random` to
    enable prior or posterior predictive sampling.

    """

    def __init__(
        self,
        logp,
        shape=(),
        dtype=None,
        testval=0,
        random=None,
        wrap_random_with_dist_shape=True,
        check_shape_in_random=True,
        *args,
        **kwargs,
    ):
        """
        Parameters
        ----------

        logp: callable
            A callable that has the following signature ``logp(value)`` and
            returns an Aesara tensor that represents the distribution's log
            probability density.
        shape: tuple (Optional): defaults to `()`
            The shape of the distribution. The default value indicates a scalar.
            If the distribution is *not* scalar-valued, the programmer should pass
            a value here.
        dtype: None, str (Optional)
            The dtype of the distribution.
        testval: number or array (Optional)
            The ``testval`` of the RV's tensor that follow the ``DensityDist``
            distribution.
        args, kwargs: (Optional)
            These are passed to the parent class' ``__init__``.

        Examples
        --------
            .. code-block:: python

                with pm.Model():
                    mu = pm.Normal('mu',0,1)
                    normal_dist = pm.Normal.dist(mu, 1)
                    pm.DensityDist(
                        'density_dist',
                        normal_dist.logp,
                        observed=np.random.randn(100),
                    )
                    trace = pm.sample(100)

            .. code-block:: python

                with pm.Model():
                    mu = pm.Normal('mu', 0 , 1)
                    normal_dist = pm.Normal.dist(mu, 1, shape=3)
                    dens = pm.DensityDist(
                        'density_dist',
                        normal_dist.logp,
                        observed=np.random.randn(100, 3),
                        shape=3,
                    )
                    prior = pm.sample_prior_predictive(10)['density_dist']
                assert prior.shape == (10, 100, 3)

        """
        if dtype is None:
            dtype = aesara.config.floatX
        super().__init__(shape, dtype, testval, *args, **kwargs)
        self.logp = logp
        if type(self.logp) == types.MethodType:
            if PLATFORM != "linux":
                warnings.warn(
                    "You are passing a bound method as logp for DensityDist, this can lead to "
                    "errors when sampling on platforms other than Linux. Consider using a "
                    "plain function instead, or subclass Distribution."
                )
            elif type(multiprocessing.get_context()) != multiprocessing.context.ForkContext:
                warnings.warn(
                    "You are passing a bound method as logp for DensityDist, this can lead to "
                    "errors when sampling when multiprocessing cannot rely on forking. Consider using a "
                    "plain function instead, or subclass Distribution."
                )
        self.rand = random
        self.wrap_random_with_dist_shape = wrap_random_with_dist_shape
        self.check_shape_in_random = check_shape_in_random

    def __getstate__(self):
        # We use dill to serialize the logp function, as this is almost
        # always defined in the notebook and won't be pickled correctly.
        # Fix https://github.com/pymc-devs/pymc3/issues/3844
        try:
            logp = dill.dumps(self.logp)
        except RecursionError as err:
            if type(self.logp) == types.MethodType:
                raise ValueError(
                    "logp for DensityDist is a bound method, leading to RecursionError while serializing"
                ) from err
            else:
                raise err
        vals = self.__dict__.copy()
        vals["logp"] = logp
        return vals

    def __setstate__(self, vals):
        vals["logp"] = dill.loads(vals["logp"])
        self.__dict__ = vals

    def _distr_parameters_for_repr(self):
        return []
