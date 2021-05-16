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
from typing import Optional, Sequence, Tuple, Union

import aesara
import aesara.tensor as at
import dill
import numpy as np

from aesara.graph.basic import Variable
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.random.var import RandomStateSharedVariable
from aesara.tensor.var import TensorVariable

from pymc3.aesaraf import change_rv_size, pandas_to_array
from pymc3.distributions import _logcdf, _logp
from pymc3.exceptions import ShapeError, ShapeWarning
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

# User-provided can be lazily specified as scalars
Shape = Union[int, TensorVariable, Sequence[Union[int, TensorVariable, type(Ellipsis)]]]
Dims = Union[str, Sequence[Union[str, None, type(Ellipsis)]]]
Size = Union[int, TensorVariable, Sequence[Union[int, TensorVariable]]]

# After conversion to vectors
WeakShape = Union[TensorVariable, Tuple[Union[int, TensorVariable, type(Ellipsis)], ...]]
WeakDims = Tuple[Union[str, None, type(Ellipsis)], ...]

# After Ellipsis were substituted
StrongShape = Union[TensorVariable, Tuple[Union[int, TensorVariable], ...]]
StrongDims = Sequence[Union[str, None]]
StrongSize = Union[TensorVariable, Tuple[Union[int, TensorVariable], ...]]


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

            # Register the Aesara `RandomVariable` type as a subclass of this
            # `Distribution` type.
            new_cls.register(rv_type)

        return new_cls


def _convert_dims(dims: Dims) -> Optional[WeakDims]:
    """ Process a user-provided dims variable into None or a valid dims tuple. """
    if dims is None:
        return None

    if isinstance(dims, str):
        dims = (dims,)
    elif isinstance(dims, (list, tuple)):
        dims = tuple(dims)
    else:
        raise ValueError(f"The `dims` parameter must be a tuple, str or list. Actual: {type(dims)}")

    if any(d == Ellipsis for d in dims[:-1]):
        raise ValueError(f"Ellipsis in `dims` may only appear in the last position. Actual: {dims}")

    return dims


def _convert_shape(shape: Shape) -> Optional[WeakShape]:
    """ Process a user-provided shape variable into None or a valid shape object. """
    if shape is None:
        return None

    if isinstance(shape, int) or (isinstance(shape, TensorVariable) and shape.ndim == 0):
        shape = (shape,)
    elif isinstance(shape, (list, tuple)):
        shape = tuple(shape)
    else:
        raise ValueError(
            f"The `shape` parameter must be a tuple, TensorVariable, int or list. Actual: {type(shape)}"
        )

    if isinstance(shape, tuple) and any(s == Ellipsis for s in shape[:-1]):
        raise ValueError(
            f"Ellipsis in `shape` may only appear in the last position. Actual: {shape}"
        )

    return shape


def _convert_size(size: Size) -> Optional[StrongSize]:
    """ Process a user-provided size variable into None or a valid size object. """
    if size is None:
        return None

    if isinstance(size, int) or (isinstance(size, TensorVariable) and size.ndim == 0):
        size = (size,)
    elif isinstance(size, (list, tuple)):
        size = tuple(size)
    else:
        raise ValueError(
            f"The `size` parameter must be a tuple, TensorVariable, int or list. Actual: {type(size)}"
        )

    if isinstance(size, tuple) and Ellipsis in size:
        raise ValueError(f"The `size` parameter cannot contain an Ellipsis. Actual: {size}")

    return size


def _resize_from_dims(
    dims: WeakDims, ndim_implied: int, model
) -> Tuple[int, StrongSize, StrongDims]:
    """Determines a potential resize shape from a `dims` tuple.

    Parameters
    ----------
    dims : array-like
        A vector of dimension names, None or Ellipsis.
    ndim_implied : int
        Number of RV dimensions that were implied from its inputs alone.
    model : pm.Model
        The current model on stack.

    Returns
    -------
    ndim_resize : int
        Number of dimensions that should be added through resizing.
    resize_shape : array-like
        The shape of the new dimensions.
    """
    if Ellipsis in dims:
        # Auto-complete the dims tuple to the full length.
        # We don't have a way to know the names of implied
        # dimensions, so they will be `None`.
        dims = (*dims[:-1], *[None] * ndim_implied)

    ndim_resize = len(dims) - ndim_implied

    # All resize dims must be known already (numerically or symbolically).
    unknowndim_resize_dims = set(dims[:ndim_resize]) - set(model.dim_lengths)
    if unknowndim_resize_dims:
        raise KeyError(
            f"Dimensions {unknowndim_resize_dims} are unknown to the model and cannot be used to specify a `size`."
        )

    # The numeric/symbolic resize tuple can be created using model.RV_dim_lengths
    resize_shape = tuple(model.dim_lengths[dname] for dname in dims[:ndim_resize])
    return ndim_resize, resize_shape, dims


def _resize_from_observed(
    observed, ndim_implied: int
) -> Tuple[int, StrongSize, Union[np.ndarray, Variable]]:
    """Determines a potential resize shape from observations.

    Parameters
    ----------
    observed : scalar, array-like
        The value of the `observed` kwarg to the RV creation.
    ndim_implied : int
        Number of RV dimensions that were implied from its inputs alone.

    Returns
    -------
    ndim_resize : int
        Number of dimensions that should be added through resizing.
    resize_shape : array-like
        The shape of the new dimensions.
    observed : scalar, array-like
        Observations as numpy array or `Variable`.
    """
    if not hasattr(observed, "shape"):
        observed = pandas_to_array(observed)
    ndim_resize = observed.ndim - ndim_implied
    resize_shape = tuple(observed.shape[d] for d in range(ndim_resize))
    return ndim_resize, resize_shape, observed


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
        initval=None,
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
        initval : optional
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

        if "testval" in kwargs:
            initval = kwargs.pop("testval")
            warnings.warn(
                "The `testval` argument is deprecated; use `initval`.",
                DeprecationWarning,
                stacklevel=2,
            )

        if not isinstance(name, string_types):
            raise TypeError(f"Name needs to be a string but got: {name}")

        if rng is None:
            rng = model.next_rng()

        if dims is not None and "shape" in kwargs:
            raise ValueError(
                f"Passing both `dims` ({dims}) and `shape` ({kwargs['shape']}) is not supported!"
            )
        if dims is not None and "size" in kwargs:
            raise ValueError(
                f"Passing both `dims` ({dims}) and `size` ({kwargs['size']}) is not supported!"
            )
        dims = _convert_dims(dims)

        # Create the RV without specifying testval, because the testval may have a shape
        # that only matches after replicating with a size implied by dims (see below).
        rv_out = cls.dist(*args, rng=rng, testval=None, **kwargs)
        ndim_actual = rv_out.ndim
        resize_shape = None

        # `dims` are only available with this API, because `.dist()` can be used
        # without a modelcontext and dims are not tracked at the Aesara level.
        if dims is not None:
            ndim_resize, resize_shape, dims = _resize_from_dims(dims, ndim_actual, model)
        elif observed is not None:
            ndim_resize, resize_shape, observed = _resize_from_observed(observed, ndim_actual)

        if resize_shape:
            # A batch size was specified through `dims`, or implied by `observed`.
            rv_out = change_rv_size(rv_var=rv_out, new_size=resize_shape, expand=True)

        if initval is not None:
            # Assigning the testval earlier causes trouble because the RV may not be created with the final shape already.
            rv_out.tag.test_value = initval

        return model.register_rv(rv_out, name, observed, total_size, dims=dims, transform=transform)

    @classmethod
    def dist(
        cls,
        dist_params,
        *,
        shape: Optional[Shape] = None,
        size: Optional[Size] = None,
        initval=None,
        **kwargs,
    ) -> RandomVariable:
        """Creates a RandomVariable corresponding to the `cls` distribution.

        Parameters
        ----------
        dist_params : array-like
            The inputs to the `RandomVariable` `Op`.
        shape : int, tuple, Variable, optional
            A tuple of sizes for each dimension of the new RV.

            An Ellipsis (...) may be inserted in the last position to short-hand refer to
            all the dimensions that the RV would get if no shape/size/dims were passed at all.
        size : int, tuple, Variable, optional
            For creating the RV like in Aesara/NumPy.
        initival : optional
            Test value to be attached to the output RV.
            Must match its shape exactly.

        Returns
        -------
        rv : RandomVariable
            The created RV.
        """
        if "testval" in kwargs:
            initval = kwargs.pop("testval")
            warnings.warn(
                "The `testval` argument is deprecated. "
                "Use `initval` to set initial values for a `Model`; "
                "otherwise, set test values on Aesara parameters explicitly "
                "when attempting to use Aesara's test value debugging features.",
                DeprecationWarning,
                stacklevel=2,
            )
        if "dims" in kwargs:
            raise NotImplementedError("The use of a `.dist(dims=...)` API is not supported.")
        if shape is not None and size is not None:
            raise ValueError(
                f"Passing both `shape` ({shape}) and `size` ({size}) is not supported!"
            )
        shape = _convert_shape(shape)
        size = _convert_size(size)

        ndim_supp = cls.rv_op.ndim_supp
        ndim_expected = None
        ndim_batch = None
        create_size = None

        if shape is not None:
            if Ellipsis in shape:
                # Ellipsis short-hands all implied dimensions. Therefore
                # we don't know how many dimensions to expect.
                ndim_expected = ndim_batch = None
                # Create the RV with its implied shape and resize later
                create_size = None
            else:
                ndim_expected = len(tuple(shape))
                ndim_batch = ndim_expected - ndim_supp
                create_size = tuple(shape)[:ndim_batch]
        elif size is not None:
            ndim_expected = ndim_supp + len(tuple(size))
            ndim_batch = ndim_expected - ndim_supp
            create_size = size

        # Create the RV with a `size` right away.
        # This is not necessarily the final result.
        rv_out = cls.rv_op(*dist_params, size=create_size, **kwargs)
        ndim_actual = rv_out.ndim
        ndims_unexpected = ndim_actual != ndim_expected

        if shape is not None and ndims_unexpected:
            if Ellipsis in shape:
                # Resize and we're done!
                rv_out = change_rv_size(rv_var=rv_out, new_size=shape[:-1], expand=True)
            else:
                # This is rare, but happens, for example, with MvNormal(np.ones((2, 3)), np.eye(3), shape=(2, 3)).
                # Recreate the RV without passing `size` to created it with just the implied dimensions.
                rv_out = cls.rv_op(*dist_params, size=None, **kwargs)

                # Now resize by any remaining "extra" dimensions that were not implied from support and parameters
                if rv_out.ndim < ndim_expected:
                    expand_shape = shape[: ndim_expected - rv_out.ndim]
                    rv_out = change_rv_size(rv_var=rv_out, new_size=expand_shape, expand=True)
                if not rv_out.ndim == ndim_expected:
                    raise ShapeError(
                        f"Failed to create the RV with the expected dimensionality. "
                        f"This indicates a severe problem. Please open an issue.",
                        actual=ndim_actual,
                        expected=ndim_batch + ndim_supp,
                    )

        # Warn about the edge cases where the RV Op creates more dimensions than
        # it should based on `size` and `RVOp.ndim_supp`.
        if size is not None and ndims_unexpected:
            warnings.warn(
                f"You may have expected a ({len(tuple(size))}+{ndim_supp})-dimensional RV, but the resulting RV will be {ndim_actual}-dimensional."
                ' To silence this warning use `warnings.simplefilter("ignore", pm.ShapeWarning)`.',
                ShapeWarning,
            )

        rng = kwargs.pop("rng", None)
        if (
            rv_out.owner
            and isinstance(rv_out.owner.op, RandomVariable)
            and isinstance(rng, RandomStateSharedVariable)
            and not getattr(rng, "default_update", None)
        ):
            # This tells `aesara.function` that the shared RNG variable
            # is mutable, which--in turn--tells the `FunctionGraph`
            # `Supervisor` feature to allow in-place updates on the variable.
            # Without it, the `RandomVariable`s could not be optimized to allow
            # in-place RNG updates, forcing all sample results from compiled
            # functions to be the same on repeated evaluations.
            new_rng = rv_out.owner.outputs[0]
            rv_out.update = (rng, new_rng)
            rng.default_update = new_rng

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
        initval=None,
        defaults=(),
        parent_dist=None,
        *args,
        **kwargs,
    ):
        super().__init__(
            shape=shape, dtype=dtype, initval=initval, defaults=defaults, *args, **kwargs
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
        initval=0,
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
        initval: number or array (Optional)
            The ``initval`` of the RV's tensor that follow the ``DensityDist``
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
        super().__init__(shape, dtype, initval, *args, **kwargs)
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
