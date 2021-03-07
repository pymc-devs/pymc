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

import functools
import re
import warnings

from typing import Dict, List, Tuple, Union

import arviz
import dill
import numpy as np
import xarray

from cachetools import LRUCache, cachedmethod
from theano.tensor import TensorVariable

from pymc3.exceptions import SamplingError

LATEX_ESCAPE_RE = re.compile(r"(%|_|\$|#|&)", re.MULTILINE)


def escape_latex(strng):
    r"""Consistently escape LaTeX special characters for _repr_latex_ in IPython

    Implementation taken from the IPython magic `format_latex`

    Examples
    --------
        escape_latex('disease_rate')  # 'disease\_rate'

    Parameters
    ----------
    strng: str
        string to escape LaTeX characters

    Returns
    -------
    str
        A string with LaTeX escaped
    """
    if strng is None:
        return "None"
    return LATEX_ESCAPE_RE.sub(r"\\\1", strng)


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


def get_repr_for_variable(variable, formatting="plain"):
    """Build a human-readable string representation for a variable."""
    if variable is not None and hasattr(variable, "name"):
        name = variable.name
    elif type(variable) in [float, int, str]:
        name = str(variable)
    else:
        name = None

    if name is None and variable is not None:
        if hasattr(variable, "get_parents"):
            try:
                names = [
                    get_repr_for_variable(item, formatting=formatting)
                    for item in variable.get_parents()[0].inputs
                ]
                # do not escape_latex these, since it is not idempotent
                if "latex" in formatting:
                    return "f({args})".format(
                        args=",~".join([n for n in names if isinstance(n, str)])
                    )
                else:
                    return "f({args})".format(
                        args=", ".join([n for n in names if isinstance(n, str)])
                    )
            except IndexError:
                pass
        value = variable.eval()
        if not value.shape or value.shape == (1,):
            return value.item()
        return "array"

    if "latex" in formatting:
        return fr"\text{{{name}}}"
    else:
        return name


def get_var_name(var):
    """Get an appropriate, plain variable name for a variable. Necessary
    because we override theano.tensor.TensorVariable.__str__ to give informative
    string representations to our pymc3.PyMC3Variables, yet we want to use the
    plain name as e.g. keys in dicts.
    """
    if isinstance(var, TensorVariable):
        return super(TensorVariable, var).__str__()
    else:
        return str(var)


def update_start_vals(a, b, model):
    r"""Update a with b, without overwriting existing keys. Values specified for
    transformed variables on the original scale are also transformed and inserted.
    """
    if model is not None:
        for free_RV in model.free_RVs:
            tname = free_RV.name
            for name in a:
                if is_transformed_name(tname) and get_untransformed_name(tname) == name:
                    transform_func = [
                        d.transformation for d in model.deterministics if d.name == name
                    ]
                    if transform_func:
                        b[tname] = transform_func[0].forward_val(a[name], point=b)

    a.update({k: v for k, v in b.items() if k not in a})


def check_start_vals(start, model):
    r"""Check that the starting values for MCMC do not cause the relevant log probability
    to evaluate to something invalid (e.g. Inf or NaN)

    Parameters
    ----------
    start : dict, or array of dict
        Starting point in parameter space (or partial point)
        Defaults to ``trace.point(-1))`` if there is a trace provided and model.test_point if not
        (defaults to empty dict). Initialization methods for NUTS (see ``init`` keyword) can
        overwrite the default.
    model : Model object
    Raises
    ______
    KeyError if the parameters provided by `start` do not agree with the parameters contained
        within `model`
    pymc3.exceptions.SamplingError if the evaluation of the parameters in `start` leads to an
        invalid (i.e. non-finite) state
    Returns
    -------
    None
    """
    start_points = [start] if isinstance(start, dict) else start
    for elem in start_points:
        if not set(elem.keys()).issubset(model.named_vars.keys()):
            extra_keys = ", ".join(set(elem.keys()) - set(model.named_vars.keys()))
            valid_keys = ", ".join(model.named_vars.keys())
            raise KeyError(
                "Some start parameters do not appear in the model!\n"
                "Valid keys are: {}, but {} was supplied".format(valid_keys, extra_keys)
            )

        initial_eval = model.check_test_point(test_point=elem)

        if not np.all(np.isfinite(initial_eval)):
            raise SamplingError(
                "Initial evaluation of model at starting point failed!\n"
                "Starting values:\n{}\n\n"
                "Initial evaluation results:\n{}".format(elem, str(initial_eval))
            )


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


# FIXME: this function is poorly named, because it returns a LIST of
# points, not a dictionary of points.
def dataset_to_point_dict(ds: xarray.Dataset) -> List[Dict[str, np.ndarray]]:
    warnings.warn(
        "dataset_to_point_dict was renamed to dataset_to_point_list and will be removed!",
        DeprecationWarning,
    )
    return dataset_to_point_list(ds)


def dataset_to_point_list(ds: xarray.Dataset) -> List[Dict[str, np.ndarray]]:
    # grab posterior samples for each variable
    _samples: Dict[str, np.ndarray] = {vn: ds[vn].values for vn in ds.keys()}
    # make dicts
    points: List[Dict[str, np.ndarray]] = []
    vn: str
    s: np.ndarray
    for c in ds.chain:
        for d in ds.draw:
            points.append({vn: s[c, d] for vn, s in _samples.items()})
    # use the list of points
    return points


def chains_and_samples(data: Union[xarray.Dataset, arviz.InferenceData]) -> Tuple[int, int]:
    """Extract and return number of chains and samples in xarray or arviz traces."""
    dataset: xarray.Dataset
    if isinstance(data, xarray.Dataset):
        dataset = data
    elif isinstance(data, arviz.InferenceData):
        dataset = data.posterior
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
        return hash(dill.dumps(a))
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
