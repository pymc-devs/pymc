#   Copyright 2023 The PyMC Developers
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
import warnings

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import arviz
import cloudpickle
import numpy as np
import xarray

from cachetools import LRUCache, cachedmethod
from pytensor import Variable
from pytensor.compile import SharedVariable
from pytensor.graph.utils import ValidatingScratchpad

from pymc.exceptions import BlockModelAccessError


class _UnsetType:
    """Type for the `UNSET` object to make it look nice in `help(...)` outputs."""

    def __str__(self):
        return "UNSET"

    def __repr__(self):
        return str(self)


UNSET = _UnsetType()


def withparent(meth):
    """Helper wrapper that passes calls to parent's instance"""

    def wrapped(self, *args, **kwargs):
        res = meth(self, *args, **kwargs)
        if getattr(self, "parent", None) is not None:
            getattr(self.parent, meth.__name__)(*args, **kwargs)
        return res

    # Unfortunately functools wrapper fails
    # when decorating built-in methods so we
    # need to fix that improper behaviour
    wrapped.__name__ = meth.__name__
    return wrapped


class treelist(list):
    """A list that passes mutable extending operations used in Model
    to parent list instance.
    Extending treelist you will also extend its parent
    """

    def __init__(self, iterable=(), parent=None):
        super().__init__(iterable)
        assert isinstance(parent, list) or parent is None
        self.parent = parent
        if self.parent is not None:
            self.parent.extend(self)

    # typechecking here works bad
    append = withparent(list.append)
    __iadd__ = withparent(list.__iadd__)
    extend = withparent(list.extend)

    def tree_contains(self, item):
        if isinstance(self.parent, treedict):
            return list.__contains__(self, item) or self.parent.tree_contains(item)
        elif isinstance(self.parent, list):
            return list.__contains__(self, item) or self.parent.__contains__(item)
        else:
            return list.__contains__(self, item)

    def __setitem__(self, key, value):
        raise NotImplementedError(
            "Method is removed as we are not able to determine appropriate logic for it"
        )

    # Added this because mypy didn't like having __imul__ without __mul__
    # This is my best guess about what this should do.  I might be happier
    # to kill both of these if they are not used.
    def __mul__(self, other) -> "treelist":
        return cast("treelist", super().__mul__(other))

    def __imul__(self, other) -> "treelist":
        t0 = len(self)
        super().__imul__(other)
        if self.parent is not None:
            self.parent.extend(self[t0:])
        return self  # python spec says should return the result.


class treedict(dict):
    """A dict that passes mutable extending operations used in Model
    to parent dict instance.
    Extending treedict you will also extend its parent
    """

    def __init__(self, iterable=(), parent=None, **kwargs):
        super().__init__(iterable, **kwargs)
        assert isinstance(parent, dict) or parent is None
        self.parent = parent
        if self.parent is not None:
            self.parent.update(self)

    # typechecking here works bad
    __setitem__ = withparent(dict.__setitem__)
    update = withparent(dict.update)

    def tree_contains(self, item):
        # needed for `add_named_variable` method
        if isinstance(self.parent, treedict):
            return dict.__contains__(self, item) or self.parent.tree_contains(item)
        elif isinstance(self.parent, dict):
            return dict.__contains__(self, item) or self.parent.__contains__(item)
        else:
            return dict.__contains__(self, item)


def get_transformed_name(name, transform):
    r"""
    Consistent way of transforming names

    Parameters
    ----------
    name: str
        Name to transform
    transform: transforms.Transform
        Should be a subclass of `transforms.Transform`

    Returns
    -------
    str
        A string to use for the transformed variable
    """
    return f"{name}_{transform.name}__"


def is_transformed_name(name):
    r"""
    Quickly check if a name was transformed with `get_transformed_name`

    Parameters
    ----------
    name: str
        Name to check

    Returns
    -------
    bool
        Boolean, whether the string could have been produced by `get_transformed_name`
    """
    return name.endswith("__") and name.count("_") >= 3


def get_untransformed_name(name):
    r"""
    Undo transformation in `get_transformed_name`. Throws ValueError if name wasn't transformed

    Parameters
    ----------
    name: str
        Name to untransform

    Returns
    -------
    str
        String with untransformed version of the name.
    """
    if not is_transformed_name(name):
        raise ValueError(f"{name} does not appear to be a transformed name")
    return "_".join(name.split("_")[:-3])


def get_default_varnames(var_iterator, include_transformed):
    r"""Helper to extract default varnames from a trace.

    Parameters
    ----------
    varname_iterator: iterator
        Elements will be cast to string to check whether it is transformed, and optionally filtered
    include_transformed: boolean
        Should transformed variable names be included in return value

    Returns
    -------
    list
        List of variables, possibly filtered
    """
    if include_transformed:
        return list(var_iterator)
    else:
        return [var for var in var_iterator if not is_transformed_name(get_var_name(var))]


def get_var_name(var) -> str:
    """Get an appropriate, plain variable name for a variable."""
    return str(getattr(var, "name", var))


def get_transformed(z):
    if hasattr(z, "transformed"):
        z = z.transformed
    return z


def biwrap(wrapper):
    @functools.wraps(wrapper)
    def enhanced(*args, **kwargs):
        is_bound_method = hasattr(args[0], wrapper.__name__) if args else False
        if is_bound_method:
            count = 1
        else:
            count = 0
        if len(args) > count:
            newfn = wrapper(*args, **kwargs)
            return newfn
        else:
            newwrapper = functools.partial(wrapper, *args, **kwargs)
            return newwrapper

    return enhanced


def dataset_to_point_list(
    ds: xarray.Dataset, sample_dims: Sequence[str]
) -> Tuple[List[Dict[str, np.ndarray]], Dict[str, Any]]:
    # All keys of the dataset must be a str
    var_names = list(ds.keys())
    for vn in var_names:
        if not isinstance(vn, str):
            raise ValueError(f"Variable names must be str, but dataset key {vn} is a {type(vn)}.")
    num_sample_dims = len(sample_dims)
    stacked_dims = {dim_name: ds[dim_name] for dim_name in sample_dims}
    ds = ds.transpose(*sample_dims, ...)
    stacked_dict = {
        vn: da.values.reshape((-1, *da.shape[num_sample_dims:])) for vn, da in ds.items()
    }
    points = [
        {vn: stacked_dict[vn][i, ...] for vn in var_names}
        for i in range(np.prod([len(coords) for coords in stacked_dims.values()]))
    ]
    # use the list of points
    return cast(List[Dict[str, np.ndarray]], points), stacked_dims


def drop_warning_stat(idata: arviz.InferenceData) -> arviz.InferenceData:
    """Returns a new ``InferenceData`` object with the "warning" stat removed from sample stats groups.

    This function should be applied to an ``InferenceData`` object obtained with
    ``pm.sample(keep_warning_stat=True)`` before trying to ``.to_netcdf()`` or ``.to_zarr()`` it.
    """
    nidata = arviz.InferenceData(attrs=idata.attrs)
    for gname, group in idata.items():
        if "sample_stat" in gname:
            group = group.drop_vars(names=["warning", "warning_dim_0"], errors="ignore")
        nidata.add_groups({gname: group}, coords=group.coords, dims=group.dims)
    return nidata


def chains_and_samples(data: Union[xarray.Dataset, arviz.InferenceData]) -> Tuple[int, int]:
    """Extract and return number of chains and samples in xarray or arviz traces."""
    dataset: xarray.Dataset
    if isinstance(data, xarray.Dataset):
        dataset = data
    elif isinstance(data, arviz.InferenceData):
        dataset = data["posterior"]
    else:
        raise ValueError(
            "Argument must be xarray Dataset or arviz InferenceData. Got %s",
            data.__class__,
        )

    coords = dataset.coords
    nchains = coords["chain"].sizes["chain"]
    nsamples = coords["draw"].sizes["draw"]
    return nchains, nsamples


def hashable(a=None) -> int:
    """
    Hashes many kinds of objects, including some that are unhashable through the builtin `hash` function.
    Lists and tuples are hashed based on their elements.
    """
    if isinstance(a, dict):
        # first hash the keys and values with hashable
        # then hash the tuple of int-tuples with the builtin
        return hash(tuple((hashable(k), hashable(v)) for k, v in a.items()))
    if isinstance(a, (tuple, list)):
        # lists are mutable and not hashable by default
        # for memoization, we need the hash to depend on the items
        return hash(tuple(hashable(i) for i in a))
    try:
        return hash(a)
    except TypeError:
        pass
    # Not hashable >>>
    try:
        return hash(cloudpickle.dumps(a))
    except Exception:
        if hasattr(a, "__dict__"):
            return hashable(a.__dict__)
        else:
            return id(a)


def hash_key(*args, **kwargs):
    return tuple(HashableWrapper(a) for a in args + tuple(kwargs.items()))


class HashableWrapper:
    __slots__ = ("obj",)

    def __init__(self, obj):
        self.obj = obj

    def __hash__(self):
        return hashable(self.obj)

    def __eq__(self, other):
        return self.obj == other

    def __repr__(self):
        return f"{type(self).__name__}({self.obj})"


class WithMemoization:
    def __hash__(self):
        return hash(id(self))

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_cache", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


def locally_cachedmethod(f):
    from collections import defaultdict

    def self_cache_fn(f_name):
        def cf(self):
            return self.__dict__.setdefault("_cache", defaultdict(lambda: LRUCache(128)))[f_name]

        return cf

    return cachedmethod(self_cache_fn(f.__name__), key=hash_key)(f)


def check_dist_not_registered(dist, model=None):
    """Check that a dist is not registered in the model already"""
    from pymc.model import modelcontext

    try:
        model = modelcontext(None)
    except (TypeError, BlockModelAccessError):
        pass
    else:
        if dist in model.basic_RVs:
            raise ValueError(
                f"The dist {dist} was already registered in the current model.\n"
                f"You should use an unregistered (unnamed) distribution created via "
                f"the `.dist()` API instead, such as:\n`dist=pm.Normal.dist(0, 1)`"
            )


def point_wrapper(core_function):
    """Wrap an pytensor compiled function to be able to ingest point dictionaries whilst
    ignoring the keys that are not valid inputs to the core function.
    """
    ins = [i.name for i in core_function.maker.fgraph.inputs if not isinstance(i, SharedVariable)]

    def wrapped(**kwargs):
        input_point = {k: v for k, v in kwargs.items() if k in ins}
        return core_function(**input_point)

    return wrapped


RandomSeed = Optional[Union[int, Sequence[int], np.ndarray]]
RandomState = Union[RandomSeed, np.random.RandomState, np.random.Generator]


def _get_seeds_per_chain(
    random_state: RandomState,
    chains: int,
) -> Union[Sequence[int], np.ndarray]:
    """Obtain or validate specified integer seeds per chain.

    This function process different possible sources of seeding and returns one integer
    seed per chain:
    1. If the input is an integer and a single chain is requested, the input is
        returned inside a tuple.
    2. If the input is a sequence or NumPy array with as many entries as chains,
        the input is returned.
    3. If the input is an integer and multiple chains are requested, new unique seeds
        are generated from NumPy default Generator seeded with that integer.
    4. If the input is None new unique seeds are generated from an unseeded NumPy default
        Generator.
    5. If a RandomState or Generator is provided, new unique seeds are generated from it.

    Raises
    ------
    ValueError
        If none of the conditions above are met
    """

    def _get_unique_seeds_per_chain(integers_fn):
        seeds = []
        while len(set(seeds)) != chains:
            seeds = [int(seed) for seed in integers_fn(2**30, dtype=np.int64, size=chains)]
        return seeds

    if random_state is None or isinstance(random_state, int):
        if chains == 1 and isinstance(random_state, int):
            return (random_state,)
        return _get_unique_seeds_per_chain(np.random.default_rng(random_state).integers)
    if isinstance(random_state, np.random.Generator):
        return _get_unique_seeds_per_chain(random_state.integers)
    if isinstance(random_state, np.random.RandomState):
        return _get_unique_seeds_per_chain(random_state.randint)

    if not isinstance(random_state, (list, tuple, np.ndarray)):
        raise ValueError(f"The `seeds` must be array-like. Got {type(random_state)} instead.")

    if len(random_state) != chains:
        raise ValueError(
            f"Number of seeds ({len(random_state)}) does not match the number of chains ({chains})."
        )

    return random_state


def get_value_vars_from_user_vars(
    vars: Union[Variable, Sequence[Variable]], model
) -> List[Variable]:
    """Converts user "vars" input into value variables.

    More often than not, users will pass random variables, and we will extract the
    respective value variables, but we also allow for the input to already be value
    variables, in case the function is called internally or by a "super-user"

    Returns
    -------
    value_vars: list of TensorVariable
        List of model value variables that correspond to the input vars

    Raises
    ------
    ValueError:
        If any of the provided variables do not correspond to any model value variable
    """
    if not isinstance(vars, Sequence):
        # Single var was passed
        value_vars = [model.rvs_to_values.get(vars, vars)]
    else:
        value_vars = [model.rvs_to_values.get(var, var) for var in vars]

    # Check that we only have value vars from the model
    model_value_vars = model.value_vars
    notin = [v for v in value_vars if v not in model_value_vars]
    if notin:
        notin = list(map(get_var_name, notin))
        # We mention random variables, even though the input may be a wrong value variable
        # because most users don't know about that duality
        raise ValueError(
            "The following variables are not random variables in the model: " + str(notin)
        )

    return value_vars


class _FutureWarningValidatingScratchpad(ValidatingScratchpad):
    def __getattribute__(self, name):
        for deprecated_names, alternative in (
            (("value_var", "observations"), "model.rvs_to_values[rv]"),
            (("transform",), "model.rvs_to_transforms[rv]"),
        ):
            if name in deprecated_names:
                try:
                    super().__getattribute__(name)
                except AttributeError:
                    pass
                else:
                    warnings.warn(
                        f"The tag attribute {name} is deprecated. Use {alternative} instead",
                        FutureWarning,
                    )
        return super().__getattribute__(name)


def _add_future_warning_tag(var) -> None:
    old_tag = var.tag
    if not isinstance(old_tag, _FutureWarningValidatingScratchpad):
        new_tag = _FutureWarningValidatingScratchpad("test_value", var.type.filter)
        for k, v in old_tag.__dict__.items():
            new_tag.__dict__.setdefault(k, v)
        var.tag = new_tag


def makeiter(a):
    if isinstance(a, (tuple, list)):
        return a
    else:
        return [a]
