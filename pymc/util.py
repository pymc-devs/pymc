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

import functools
import re

from collections import namedtuple
from collections.abc import Sequence
from copy import deepcopy
from typing import cast

import arviz
import cloudpickle
import numpy as np
import xarray

from cachetools import LRUCache, cachedmethod
from pytensor import Variable
from pytensor.compile import SharedVariable

from pymc.exceptions import BlockModelAccessError


class _UnsetType:
    """Type for the `UNSET` object to make it look nice in `help(...)` outputs."""

    def __str__(self):
        return "UNSET"

    def __repr__(self):
        return str(self)


UNSET = _UnsetType()


def withparent(meth):
    """Pass calls to parent's instance."""

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
    """A list that passes mutable extending operations used in Model to parent list instance.

    Extending treelist you will also extend its parent.
    """

    def __init__(self, iterable=(), parent=None):
        super().__init__(iterable)
        assert isinstance(parent, list) or parent is None
        self.parent = parent
        if self.parent is not None:
            self.parent.extend(self)

    # here typechecking works bad
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
        """Set value at index `key` with value `value`."""
        raise NotImplementedError(
            "Method is removed as we are not able to determine appropriate logic for it"
        )

    # Added this because mypy didn't like having __imul__ without __mul__
    # This is my best guess about what this should do.  I might be happier
    # to kill both of these if they are not used.
    def __mul__(self, other) -> "treelist":
        """Multiplication."""
        return cast("treelist", super().__mul__(other))

    def __imul__(self, other) -> "treelist":
        """Inplace multiplication."""
        t0 = len(self)
        super().__imul__(other)
        if self.parent is not None:
            self.parent.extend(self[t0:])
        return self  # python spec says should return the result.


class treedict(dict):
    """A dict that passes mutable extending operations used in Model to parent dict instance.

    Extending treedict you will also extend its parent.
    """

    def __init__(self, iterable=(), parent=None, **kwargs):
        super().__init__(iterable, **kwargs)
        assert isinstance(parent, dict) or parent is None
        self.parent = parent
        if self.parent is not None:
            self.parent.update(self)

    # here typechecking works bad
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
    Consistent way of transforming names.

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
    Quickly check if a name was transformed with `get_transformed_name`.

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
    Undo transformation in `get_transformed_name`. Throws ValueError if name wasn't transformed.

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
    r"""Extract default varnames from a trace.

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
    return var if isinstance(var, str) else str(var.name)


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


def drop_warning_stat(idata: arviz.InferenceData) -> arviz.InferenceData:
    """Return a new ``InferenceData`` object with the "warning" stat removed from sample stats groups.

    This function should be applied to an ``InferenceData`` object obtained with
    ``pm.sample(keep_warning_stat=True)`` before trying to ``.to_netcdf()`` or ``.to_zarr()`` it.
    """
    nidata = arviz.InferenceData(attrs=idata.attrs)
    for gname, group in idata.items():
        if "sample_stat" in gname:
            warning_vars = [
                name
                for name in group.data_vars
                if name == "warning" or re.match(r"sampler_\d+__warning", str(name))
            ]
            group = group.drop_vars(names=[*warning_vars, "warning_dim_0"], errors="ignore")
        nidata.add_groups({gname: group}, coords=group.coords, dims=group.dims)
    return nidata


def chains_and_samples(data: xarray.Dataset | arviz.InferenceData) -> tuple[int, int]:
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
    Hash many kinds of objects, including some that are unhashable through the builtin `hash` function.

    Lists and tuples are hashed based on their elements.
    """
    if isinstance(a, dict):
        # first hash the keys and values with hashable
        # then hash the tuple of int-tuples with the builtin
        return hash(tuple((hashable(k), hashable(v)) for k, v in a.items()))
    if isinstance(a, tuple | list):
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
        """Return a hash of the object."""
        return hashable(self.obj)

    def __eq__(self, other):
        """Compare this object with `other`."""
        return self.obj == other

    def __repr__(self):
        """Return a string representation of the object."""
        return f"{type(self).__name__}({self.obj})"


class WithMemoization:
    def __hash__(self):
        """Return a hash of the object."""
        return hash(id(self))

    def __getstate__(self):
        """Return an object to pickle."""
        state = self.__dict__.copy()
        state.pop("_cache", None)
        return state

    def __setstate__(self, state):
        """Set the object from a pickled object."""
        self.__dict__.update(state)


def locally_cachedmethod(f):
    from collections import defaultdict

    def self_cache_fn(f_name):
        def cf(self):
            return self.__dict__.setdefault("_cache", defaultdict(lambda: LRUCache(128)))[f_name]

        return cf

    return cachedmethod(self_cache_fn(f.__name__), key=hash_key)(f)


def check_dist_not_registered(dist, model=None):
    """Check that a dist is not registered in the model already."""
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
    """
    Wrap a pytensor compiled function to ingest point dictionaries.

    It ignores the keys that are not valid inputs to the core function.
    """
    ins = [i.name for i in core_function.maker.fgraph.inputs if not isinstance(i, SharedVariable)]

    def wrapped(**kwargs):
        input_point = {k: v for k, v in kwargs.items() if k in ins}
        return core_function(**input_point)

    return wrapped


RandomSeed = None | int | Sequence[int] | np.ndarray
RandomState = RandomSeed | np.random.RandomState | np.random.Generator
RandomGenerator = RandomSeed | np.random.Generator | np.random.BitGenerator


def _get_seeds_per_chain(
    random_state: RandomState,
    chains: int,
) -> Sequence[int] | np.ndarray:
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

    try:
        int_random_state = int(random_state)  # type: ignore[arg-type]
    except Exception:
        int_random_state = None

    if random_state is None or int_random_state is not None:
        if chains == 1 and int_random_state is not None:
            return (int_random_state,)
        return _get_unique_seeds_per_chain(np.random.default_rng(int_random_state).integers)
    if isinstance(random_state, np.random.Generator):
        return _get_unique_seeds_per_chain(random_state.integers)
    if isinstance(random_state, np.random.RandomState):
        return _get_unique_seeds_per_chain(random_state.randint)

    if not isinstance(random_state, list | tuple | np.ndarray):
        raise ValueError(f"The `seeds` must be array-like. Got {type(random_state)} instead.")

    if len(random_state) != chains:
        raise ValueError(
            f"Number of seeds ({len(random_state)}) does not match the number of chains ({chains})."
        )

    return random_state


def get_value_vars_from_user_vars(vars: Variable | Sequence[Variable], model) -> list[Variable]:
    """Convert user "vars" input into value variables.

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


def makeiter(a):
    if isinstance(a, tuple | list):
        return a
    else:
        return [a]


RandomGeneratorState = namedtuple("RandomGeneratorState", ["bit_generator_state", "seed_seq_state"])


def get_state_from_generator(
    rng: np.random.Generator | np.random.BitGenerator,
) -> RandomGeneratorState:
    assert isinstance(rng, (np.random.Generator | np.random.BitGenerator))
    bit_gen: np.random.BitGenerator = (
        rng.bit_generator if isinstance(rng, np.random.Generator) else rng
    )

    return RandomGeneratorState(
        bit_generator_state=bit_gen.state,
        seed_seq_state=bit_gen.seed_seq.state,  # type: ignore[attr-defined]
    )


def random_generator_from_state(state: RandomGeneratorState) -> np.random.Generator:
    seed_seq = np.random.SeedSequence(**state.seed_seq_state)
    bit_generator_class = getattr(np.random, state.bit_generator_state["bit_generator"])
    bit_generator = bit_generator_class(seed_seq)
    bit_generator.state = state.bit_generator_state
    return np.random.Generator(bit_generator)


def get_random_generator(
    seed: RandomGenerator | np.random.RandomState = None, copy: bool = True
) -> np.random.Generator:
    """Build a :py:class:`~numpy.random.Generator` object from a suitable seed.

    Parameters
    ----------
    seed : None | int | Sequence[int] | numpy.random.Generator | numpy.random.BitGenerator | numpy.random.RandomState
        A suitable seed to use to generate the :py:class:`~numpy.random.Generator` object.
        For more details on suitable seeds, refer to :py:func:`numpy.random.default_rng`.
    copy : bool
        Boolean flag that indicates whether to copy the seed object before feeding
        it to :py:func:`numpy.random.default_rng`. If `copy` is `False`, and the seed
        object is a ``BitGenerator`` or ``Generator`` object, the returned
        ``Generator`` will use the ``seed`` object where possible. This means that it
        will return the ``seed`` input object if it is a ``Generator`` or that it
        will return a new ``Generator`` whose ``bit_generator`` attribute will be the
        input ``seed`` object. To avoid this potential object sharing, you must set
        ``copy`` to ``True``.

    Returns
    -------
    rng : numpy.random.Generator
        The result of passing the input ``seed`` (or a copy of it) through
        :py:func:`numpy.random.default_rng`.

    Raises
    ------
    TypeError:
        If the supplied ``seed`` is a :py:class:`~numpy.random.RandomState` object. We
        do not support using these legacy objects because their seeding strategy is not
        amenable to spawning new independent random streams.
    """
    if isinstance(seed, np.random.RandomState):
        raise TypeError(
            "Cannot create a random Generator from a RandomStream object. "
            "Please provide a random seed, BitGenerator or Generator instead."
        )
    if copy:
        # If seed is a numpy.random.Generator or numpy.random.BitGenerator,
        # numpy.random.default_rng will use the exact same object to return.
        # In the former case, it will return seed, in the latter it will return
        # a new Generator object that has the same BitGenerator. This would potentially
        # make the new generator be shared across many users. To avoid this, we
        # copy by default.
        # Also, because of https://github.com/numpy/numpy/issues/27727, we can't use
        # deepcopy. We must rebuild a Generator without losing the SeedSequence information
        if isinstance(seed, np.random.Generator | np.random.BitGenerator):
            return random_generator_from_state(get_state_from_generator(seed))
        seed = deepcopy(seed)
    return np.random.default_rng(seed)
