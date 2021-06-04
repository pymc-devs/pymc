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
from typing import Optional

import aesara
import aesara.tensor as at
import dill

from aesara.tensor.random.op import RandomVariable
from aesara.tensor.random.var import RandomStateSharedVariable

from pymc3.aesaraf import change_rv_size
from pymc3.distributions import _logcdf, _logp
from pymc3.distributions.shape_utils import (
    Dims,
    Shape,
    Size,
    convert_dims,
    convert_shape,
    convert_size,
    find_size,
    maybe_resize,
    resize_from_dims,
    resize_from_observed,
)
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
        dims = convert_dims(dims)

        # Create the RV without specifying initval, because the initval may have a shape
        # that only matches after replicating with a size implied by dims (see below).
        rv_out = cls.dist(*args, rng=rng, initval=None, **kwargs)
        ndim_actual = rv_out.ndim
        resize_shape = None

        # `dims` are only available with this API, because `.dist()` can be used
        # without a modelcontext and dims are not tracked at the Aesara level.
        if dims is not None:
            ndim_resize, resize_shape, dims = resize_from_dims(dims, ndim_actual, model)
        elif observed is not None:
            ndim_resize, resize_shape, observed = resize_from_observed(observed, ndim_actual)

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

        shape = convert_shape(shape)
        size = convert_size(size)

        create_size, ndim_expected, ndim_batch, ndim_supp = find_size(
            shape=shape, size=size, ndim_supp=cls.rv_op.ndim_supp
        )
        # Create the RV with a `size` right away.
        # This is not necessarily the final result.
        rv_out = cls.rv_op(*dist_params, size=create_size, **kwargs)
        rv_out = maybe_resize(
            rv_out,
            cls.rv_op,
            dist_params,
            ndim_expected,
            ndim_batch,
            ndim_supp,
            shape,
            size,
            **kwargs,
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
