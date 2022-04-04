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
import warnings

from typing import (
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import aesara
import aesara.tensor as at
import numpy as np
import scipy.sparse as sps

from aeppl.logprob import CheckParameterValue
from aesara import config, scalar
from aesara.compile.mode import Mode, get_mode
from aesara.gradient import grad
from aesara.graph import local_optimizer
from aesara.graph.basic import (
    Apply,
    Constant,
    Variable,
    clone_get_equiv,
    graph_inputs,
    vars_between,
    walk,
)
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import Op, compute_test_value
from aesara.sandbox.rng_mrg import MRG_RandomStream as RandomStream
from aesara.scalar.basic import Cast
from aesara.tensor.elemwise import Elemwise
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.shape import SpecifyShape
from aesara.tensor.sharedvar import SharedVariable
from aesara.tensor.subtensor import AdvancedIncSubtensor, AdvancedIncSubtensor1
from aesara.tensor.var import TensorConstant, TensorVariable

from pymc.exceptions import ShapeError
from pymc.vartypes import continuous_types, int_types, isgenerator, typefilter

PotentialShapeType = Union[
    int, np.ndarray, Tuple[Union[int, Variable], ...], List[Union[int, Variable]], Variable
]


__all__ = [
    "gradient",
    "hessian",
    "hessian_diag",
    "inputvars",
    "cont_inputs",
    "floatX",
    "intX",
    "smartfloatX",
    "jacobian",
    "CallableTensor",
    "join_nonshared_inputs",
    "make_shared_replacements",
    "generator",
    "set_at_rng",
    "at_rng",
    "take_along_axis",
    "pandas_to_array",
]


def pandas_to_array(data):
    """Convert a pandas object to a NumPy array.

    XXX: When `data` is a generator, this will return an Aesara tensor!

    """
    if hasattr(data, "to_numpy") and hasattr(data, "isnull"):
        # typically, but not limited to pandas objects
        vals = data.to_numpy()
        null_data = data.isnull()
        if hasattr(null_data, "to_numpy"):
            # pandas Series
            mask = null_data.to_numpy()
        else:
            # pandas Index
            mask = null_data
        if mask.any():
            # there are missing values
            ret = np.ma.MaskedArray(vals, mask)
        else:
            ret = vals
    elif isinstance(data, np.ndarray):
        if isinstance(data, np.ma.MaskedArray):
            if not data.mask.any():
                # empty mask
                ret = data.filled()
            else:
                # already masked and rightly so
                ret = data
        else:
            # already a ndarray, but not masked
            mask = np.isnan(data)
            if np.any(mask):
                ret = np.ma.MaskedArray(data, mask)
            else:
                # no masking required
                ret = data
    elif isinstance(data, Variable):
        ret = data
    elif sps.issparse(data):
        ret = data
    elif isgenerator(data):
        ret = generator(data)
    else:
        ret = np.asarray(data)

    # type handling to enable index variables when data is int:
    if hasattr(data, "dtype"):
        if "int" in str(data.dtype):
            return intX(ret)
        # otherwise, assume float:
        else:
            return floatX(ret)
    # needed for uses of this function other than with pm.Data:
    else:
        return floatX(ret)


def change_rv_size(
    rv: TensorVariable,
    new_size: PotentialShapeType,
    expand: Optional[bool] = False,
) -> TensorVariable:
    """Change or expand the size of a `RandomVariable`.

    Parameters
    ==========
    rv
        The old `RandomVariable` output.
    new_size
        The new size.
    expand:
        Expand the existing size by `new_size`.

    """
    # Check the dimensionality of the `new_size` kwarg
    new_size_ndim = np.ndim(new_size)
    if new_size_ndim > 1:
        raise ShapeError("The `new_size` must be ≤1-dimensional.", actual=new_size_ndim)
    elif new_size_ndim == 0:
        new_size = (new_size,)

    # Extract the RV node that is to be resized, together with its inputs, name and tag
    if isinstance(rv.owner.op, SpecifyShape):
        rv = rv.owner.inputs[0]
    rv_node = rv.owner
    rng, size, dtype, *dist_params = rv_node.inputs
    name = rv.name
    tag = rv.tag

    if expand:
        shape = tuple(rv_node.op._infer_shape(size, dist_params))
        size = shape[: len(shape) - rv_node.op.ndim_supp]
        new_size = tuple(new_size) + tuple(size)

    # Make sure the new size is a tensor. This dtype-aware conversion helps
    # to not unnecessarily pick up a `Cast` in some cases (see #4652).
    new_size = at.as_tensor(new_size, ndim=1, dtype="int64")

    new_rv_node = rv_node.op.make_node(rng, new_size, dtype, *dist_params)
    new_rv = new_rv_node.outputs[-1]
    new_rv.name = name
    for k, v in tag.__dict__.items():
        new_rv.tag.__dict__.setdefault(k, v)

    # Update "traditional" rng default_update, if that was set for old RV
    default_update = getattr(rng, "default_update", None)
    if default_update is not None and default_update is rv_node.outputs[0]:
        rng.default_update = new_rv_node.outputs[0]

    if config.compute_test_value != "off":
        compute_test_value(new_rv_node)

    return new_rv


def extract_rv_and_value_vars(
    var: TensorVariable,
) -> Tuple[TensorVariable, TensorVariable]:
    """Return a random variable and it's observations or value variable, or ``None``.

    Parameters
    ==========
    var
        A variable corresponding to a ``RandomVariable``.

    Returns
    =======
    The first value in the tuple is the ``RandomVariable``, and the second is the
    measure/log-likelihood value variable that corresponds with the latter.

    """
    if not var.owner:
        return None, None

    if isinstance(var.owner.op, RandomVariable):
        rv_value = getattr(var.tag, "observations", getattr(var.tag, "value_var", None))
        return var, rv_value

    return None, None


def extract_obs_data(x: TensorVariable) -> np.ndarray:
    """Extract data from observed symbolic variables.

    Raises
    ------
    TypeError

    """
    if isinstance(x, Constant):
        return x.data
    if isinstance(x, SharedVariable):
        return x.get_value()
    if x.owner and isinstance(x.owner.op, Elemwise) and isinstance(x.owner.op.scalar_op, Cast):
        array_data = extract_obs_data(x.owner.inputs[0])
        return array_data.astype(x.type.dtype)
    if x.owner and isinstance(x.owner.op, (AdvancedIncSubtensor, AdvancedIncSubtensor1)):
        array_data = extract_obs_data(x.owner.inputs[0])
        mask_idx = tuple(extract_obs_data(i) for i in x.owner.inputs[2:])
        mask = np.zeros_like(array_data)
        mask[mask_idx] = 1
        return np.ma.MaskedArray(array_data, mask)

    raise TypeError(f"Data cannot be extracted from {x}")


def walk_model(
    graphs: Iterable[TensorVariable],
    walk_past_rvs: bool = False,
    stop_at_vars: Optional[Set[TensorVariable]] = None,
    expand_fn: Callable[[TensorVariable], Iterable[TensorVariable]] = lambda var: [],
) -> Generator[TensorVariable, None, None]:
    """Walk model graphs and yield their nodes.

    By default, these walks will not go past ``RandomVariable`` nodes.

    Parameters
    ==========
    graphs
        The graphs to walk.
    walk_past_rvs
        If ``True``, the walk will not terminate at ``RandomVariable``s.
    stop_at_vars
        A list of variables at which the walk will terminate.
    expand_fn
        A function that returns the next variable(s) to be traversed.
    """
    if stop_at_vars is None:
        stop_at_vars = set()

    def expand(var):
        new_vars = expand_fn(var)

        if (
            var.owner
            and (walk_past_rvs or not isinstance(var.owner.op, RandomVariable))
            and (var not in stop_at_vars)
        ):
            new_vars.extend(reversed(var.owner.inputs))

        return new_vars

    yield from walk(graphs, expand, False)


def replace_rvs_in_graphs(
    graphs: Iterable[TensorVariable],
    replacement_fn: Callable[[TensorVariable], Dict[TensorVariable, TensorVariable]],
    initial_replacements: Optional[Dict[TensorVariable, TensorVariable]] = None,
    **kwargs,
) -> Tuple[TensorVariable, Dict[TensorVariable, TensorVariable]]:
    """Replace random variables in graphs

    This will *not* recompute test values.

    Parameters
    ==========
    graphs
        The graphs in which random variables are to be replaced.

    Returns
    =======
    Tuple containing the transformed graphs and a ``dict`` of the replacements
    that were made.
    """
    replacements = {}
    if initial_replacements:
        replacements.update(initial_replacements)

    def expand_replace(var):
        new_nodes = []
        if var.owner and isinstance(var.owner.op, RandomVariable):
            new_nodes.extend(replacement_fn(var, replacements))
        return new_nodes

    for var in walk_model(graphs, expand_fn=expand_replace, **kwargs):
        pass

    if replacements:
        inputs = [i for i in graph_inputs(graphs) if not isinstance(i, Constant)]
        equiv = {k: k for k in replacements.keys()}
        equiv = clone_get_equiv(inputs, graphs, False, False, equiv)

        fg = FunctionGraph(
            [equiv[i] for i in inputs],
            [equiv[o] for o in graphs],
            clone=False,
        )

        fg.replace_all(replacements.items(), import_missing=True)

        graphs = list(fg.outputs)

    return graphs, replacements


def rvs_to_value_vars(
    graphs: Iterable[TensorVariable],
    apply_transforms: bool = False,
    initial_replacements: Optional[Dict[TensorVariable, TensorVariable]] = None,
    **kwargs,
) -> Tuple[TensorVariable, Dict[TensorVariable, TensorVariable]]:
    """Clone and replace random variables in graphs with their value variables.

    This will *not* recompute test values in the resulting graphs.

    Parameters
    ==========
    graphs
        The graphs in which to perform the replacements.
    apply_transforms
        If ``True``, apply each value variable's transform.
    initial_replacements
        A ``dict`` containing the initial replacements to be made.

    """

    # Avoid circular dependency
    from pymc.distributions import NoDistribution

    def transform_replacements(var, replacements):
        rv_var, rv_value_var = extract_rv_and_value_vars(var)

        if rv_value_var is None:
            # If RandomVariable does not have a value_var and corresponds to
            # a NoDistribution, we allow further replacements in upstream graph
            if isinstance(rv_var.owner.op, NoDistribution):
                return rv_var.owner.inputs

            else:
                warnings.warn(
                    f"No value variable found for {rv_var}; "
                    "the random variable will not be replaced."
                )
                return []

        transform = getattr(rv_value_var.tag, "transform", None)

        if transform is None or not apply_transforms:
            replacements[var] = rv_value_var
            # In case the value variable is itself a graph, we walk it for
            # potential replacements
            return [rv_value_var]

        trans_rv_value = transform.backward(rv_value_var, *rv_var.owner.inputs)
        replacements[var] = trans_rv_value

        # Walk the transformed variable and make replacements
        return [trans_rv_value]

    # Clone original graphs
    inputs = [i for i in graph_inputs(graphs) if not isinstance(i, Constant)]
    equiv = clone_get_equiv(inputs, graphs, False, False, {})
    graphs = [equiv[n] for n in graphs]

    if initial_replacements:
        initial_replacements = {
            equiv.get(k, k): equiv.get(v, v) for k, v in initial_replacements.items()
        }

    return replace_rvs_in_graphs(graphs, transform_replacements, initial_replacements, **kwargs)


def inputvars(a):
    """
    Get the inputs into Aesara variables

    Parameters
    ----------
        a: Aesara variable

    Returns
    -------
        r: list of tensor variables that are inputs
    """
    return [
        v
        for v in graph_inputs(makeiter(a))
        if isinstance(v, TensorVariable) and not isinstance(v, TensorConstant)
    ]


def cont_inputs(a):
    """
    Get the continuous inputs into Aesara variables

    Parameters
    ----------
        a: Aesara variable

    Returns
    -------
        r: list of tensor variables that are continuous inputs
    """
    return typefilter(inputvars(a), continuous_types)


def floatX(X):
    """
    Convert an Aesara tensor or numpy array to aesara.config.floatX type.
    """
    try:
        return X.astype(aesara.config.floatX)
    except AttributeError:
        # Scalar passed
        return np.asarray(X, dtype=aesara.config.floatX)


_conversion_map = {"float64": "int32", "float32": "int16", "float16": "int8", "float8": "int8"}


def intX(X):
    """
    Convert a aesara tensor or numpy array to aesara.tensor.int32 type.
    """
    intX = _conversion_map[aesara.config.floatX]
    try:
        return X.astype(intX)
    except AttributeError:
        # Scalar passed
        return np.asarray(X, dtype=intX)


def smartfloatX(x):
    """
    Converts numpy float values to floatX and leaves values of other types unchanged.
    """
    if str(x.dtype).startswith("float"):
        x = floatX(x)
    return x


"""
Aesara derivative functions
"""


def gradient1(f, v):
    """flat gradient of f wrt v"""
    return at.flatten(grad(f, v, disconnected_inputs="warn"))


empty_gradient = at.zeros(0, dtype="float32")


def gradient(f, vars=None):
    if vars is None:
        vars = cont_inputs(f)

    if vars:
        return at.concatenate([gradient1(f, v) for v in vars], axis=0)
    else:
        return empty_gradient


def jacobian1(f, v):
    """jacobian of f wrt v"""
    f = at.flatten(f)
    idx = at.arange(f.shape[0], dtype="int32")

    def grad_i(i):
        return gradient1(f[i], v)

    return aesara.map(grad_i, idx)[0]


def jacobian(f, vars=None):
    if vars is None:
        vars = cont_inputs(f)

    if vars:
        return at.concatenate([jacobian1(f, v) for v in vars], axis=1)
    else:
        return empty_gradient


def jacobian_diag(f, x):
    idx = at.arange(f.shape[0], dtype="int32")

    def grad_ii(i, f, x):
        return grad(f[i], x)[i]

    return aesara.scan(
        grad_ii, sequences=[idx], n_steps=f.shape[0], non_sequences=[f, x], name="jacobian_diag"
    )[0]


@aesara.config.change_flags(compute_test_value="ignore")
def hessian(f, vars=None):
    return -jacobian(gradient(f, vars), vars)


@aesara.config.change_flags(compute_test_value="ignore")
def hessian_diag1(f, v):
    g = gradient1(f, v)
    idx = at.arange(g.shape[0], dtype="int32")

    def hess_ii(i):
        return gradient1(g[i], v)[i]

    return aesara.map(hess_ii, idx)[0]


@aesara.config.change_flags(compute_test_value="ignore")
def hessian_diag(f, vars=None):
    if vars is None:
        vars = cont_inputs(f)

    if vars:
        return -at.concatenate([hessian_diag1(f, v) for v in vars], axis=0)
    else:
        return empty_gradient


def makeiter(a):
    if isinstance(a, (tuple, list)):
        return a
    else:
        return [a]


class IdentityOp(scalar.UnaryScalarOp):
    @staticmethod
    def st_impl(x):
        return x

    def impl(self, x):
        return x

    def grad(self, inp, grads):
        return grads

    def c_code(self, node, name, inp, out, sub):
        return f"{out[0]} = {inp[0]};"

    def __eq__(self, other):
        return isinstance(self, type(other))

    def __hash__(self):
        return hash(type(self))


def make_shared_replacements(point, vars, model):
    """
    Makes shared replacements for all *other* variables than the ones passed.

    This way functions can be called many times without setting unchanging variables. Allows us
    to use func.trust_input by removing the need for DictToArrayBijection and kwargs.

    Parameters
    ----------
    point: dictionary mapping variable names to sample values
    vars: list of variables not to make shared
    model: model

    Returns
    -------
    Dict of variable -> new shared variable
    """
    othervars = set(model.value_vars) - set(vars)
    return {
        var: aesara.shared(point[var.name], var.name + "_shared", broadcastable=var.broadcastable)
        for var in othervars
    }


def join_nonshared_inputs(
    point: Dict[str, np.ndarray],
    xs: List[TensorVariable],
    vars: List[TensorVariable],
    shared,
    make_shared: bool = False,
):
    """
    Takes a list of Aesara Variables and joins their non shared inputs into a single input.

    Parameters
    ----------
    point: a sample point
    xs: list of Aesara tensors
    vars: list of variables to join

    Returns
    -------
    tensors, inarray
    tensors: list of same tensors but with inarray as input
    inarray: vector of inputs
    """
    if not vars:
        raise ValueError("Empty list of variables.")

    joined = at.concatenate([var.ravel() for var in vars])

    if not make_shared:
        tensor_type = joined.type
        inarray = tensor_type("inarray")
    else:
        if point is None:
            raise ValueError("A point is required when `make_shared` is True")
        joined_values = np.concatenate([point[var.name].ravel() for var in vars])
        inarray = aesara.shared(joined_values, "inarray")

    if aesara.config.compute_test_value != "off":
        inarray.tag.test_value = joined.tag.test_value

    replace = {}
    last_idx = 0
    for var in vars:
        shape = point[var.name].shape
        arr_len = np.prod(shape, dtype=int)
        replace[var] = reshape_t(inarray[last_idx : last_idx + arr_len], shape).astype(var.dtype)
        last_idx += arr_len

    replace.update(shared)

    xs_special = [aesara.clone_replace(x, replace, strict=False) for x in xs]
    return xs_special, inarray


def reshape_t(x, shape):
    """Work around fact that x.reshape(()) doesn't work"""
    if shape != ():
        return x.reshape(shape)
    else:
        return x[0]


class CallableTensor:
    """Turns a symbolic variable with one input into a function that returns symbolic arguments
    with the one variable replaced with the input.
    """

    def __init__(self, tensor):
        self.tensor = tensor

    def __call__(self, input):
        """Replaces the single input of symbolic variable to be the passed argument.

        Parameters
        ----------
        input: TensorVariable
        """
        (oldinput,) = inputvars(self.tensor)
        return aesara.clone_replace(self.tensor, {oldinput: input}, strict=False)


scalar_identity = IdentityOp(scalar.upgrade_to_float, name="scalar_identity")
identity = Elemwise(scalar_identity, name="identity")


class GeneratorOp(Op):
    """
    Generator Op is designed for storing python generators inside aesara graph.

    __call__ creates TensorVariable
        It has 2 new methods
        - var.set_gen(gen): sets new generator
        - var.set_default(value): sets new default value (None erases default value)

    If generator is exhausted, variable will produce default value if it is not None,
    else raises `StopIteration` exception that can be caught on runtime.

    Parameters
    ----------
    gen: generator that implements __next__ (py3) or next (py2) method
        and yields np.arrays with same types
    default: np.array with the same type as generator produces
    """

    __props__ = ("generator",)

    def __init__(self, gen, default=None):
        from pymc.data import GeneratorAdapter

        super().__init__()
        if not isinstance(gen, GeneratorAdapter):
            gen = GeneratorAdapter(gen)
        self.generator = gen
        self.set_default(default)

    def make_node(self, *inputs):
        gen_var = self.generator.make_variable(self)
        return Apply(self, [], [gen_var])

    def perform(self, node, inputs, output_storage, params=None):
        if self.default is not None:
            output_storage[0][0] = next(self.generator, self.default)
        else:
            output_storage[0][0] = next(self.generator)

    def do_constant_folding(self, fgraph, node):
        return False

    __call__ = aesara.config.change_flags(compute_test_value="off")(Op.__call__)

    def set_gen(self, gen):
        from pymc.data import GeneratorAdapter

        if not isinstance(gen, GeneratorAdapter):
            gen = GeneratorAdapter(gen)
        if not gen.tensortype == self.generator.tensortype:
            raise ValueError("New generator should yield the same type")
        self.generator = gen

    def set_default(self, value):
        if value is None:
            self.default = None
        else:
            value = np.asarray(value, self.generator.tensortype.dtype)
            t1 = (False,) * value.ndim
            t2 = self.generator.tensortype.broadcastable
            if not t1 == t2:
                raise ValueError("Default value should have the same type as generator")
            self.default = value


def generator(gen, default=None):
    """
    Generator variable with possibility to set default value and new generator.
    If generator is exhausted variable will produce default value if it is not None,
    else raises `StopIteration` exception that can be caught on runtime.

    Parameters
    ----------
    gen: generator that implements __next__ (py3) or next (py2) method
        and yields np.arrays with same types
    default: np.array with the same type as generator produces

    Returns
    -------
    TensorVariable
        It has 2 new methods
        - var.set_gen(gen): sets new generator
        - var.set_default(value): sets new default value (None erases default value)
    """
    return GeneratorOp(gen, default)()


_at_rng = RandomStream()


def at_rng(random_seed=None):
    """
    Get the package-level random number generator or new with specified seed.

    Parameters
    ----------
    random_seed: int
        If not None
        returns *new* aesara random generator without replacing package global one

    Returns
    -------
    `aesara.tensor.random.utils.RandomStream` instance
        `aesara.tensor.random.utils.RandomStream`
        instance passed to the most recent call of `set_at_rng`
    """
    if random_seed is None:
        return _at_rng
    else:
        ret = RandomStream(random_seed)
        return ret


def set_at_rng(new_rng):
    """
    Set the package-level random number generator.

    Parameters
    ----------
    new_rng: `aesara.tensor.random.utils.RandomStream` instance
        The random number generator to use.
    """
    # pylint: disable=global-statement
    global _at_rng
    # pylint: enable=global-statement
    if isinstance(new_rng, int):
        new_rng = RandomStream(new_rng)
    _at_rng = new_rng


def floatX_array(x):
    return floatX(np.array(x))


def ix_(*args):
    """
    Aesara np.ix_ analog

    See numpy.lib.index_tricks.ix_ for reference
    """
    out = []
    nd = len(args)
    for k, new in enumerate(args):
        if new is None:
            out.append(slice(None))
        new = at.as_tensor(new)
        if new.ndim != 1:
            raise ValueError("Cross index must be 1 dimensional")
        new = new.reshape((1,) * k + (new.size,) + (1,) * (nd - k - 1))
        out.append(new)
    return tuple(out)


def largest_common_dtype(tensors):
    dtypes = {
        str(t.dtype) if hasattr(t, "dtype") else smartfloatX(np.asarray(t)).dtype for t in tensors
    }
    return np.stack([np.ones((), dtype=dtype) for dtype in dtypes]).dtype


def _make_along_axis_idx(arr_shape, indices, axis):
    # compute dimensions to iterate over
    if str(indices.dtype) not in int_types:
        raise IndexError("`indices` must be an integer array")
    shape_ones = (1,) * indices.ndim
    dest_dims = list(range(axis)) + [None] + list(range(axis + 1, indices.ndim))

    # build a fancy index, consisting of orthogonal aranges, with the
    # requested index inserted at the right location
    fancy_index = []
    for dim, n in zip(dest_dims, arr_shape):
        if dim is None:
            fancy_index.append(indices)
        else:
            ind_shape = shape_ones[:dim] + (-1,) + shape_ones[dim + 1 :]
            fancy_index.append(at.arange(n).reshape(ind_shape))

    return tuple(fancy_index)


def take_along_axis(arr, indices, axis=0):
    """Take values from the input array by matching 1d index and data slices.

    This iterates over matching 1d slices oriented along the specified axis in
    the index and data arrays, and uses the former to look up values in the
    latter. These slices can be different lengths.

    Functions returning an index along an axis, like argsort and argpartition,
    produce suitable indices for this function.
    """
    arr = at.as_tensor_variable(arr)
    indices = at.as_tensor_variable(indices)
    # normalize inputs
    if axis is None:
        arr = arr.flatten()
        arr_shape = (len(arr),)  # flatiter has no .shape
        _axis = 0
    else:
        if axis < 0:
            _axis = arr.ndim + axis
        else:
            _axis = axis
        if _axis < 0 or _axis >= arr.ndim:
            raise ValueError(
                "Supplied `axis` value {} is out of bounds of an array with "
                "ndim = {}".format(axis, arr.ndim)
            )
        arr_shape = arr.shape
    if arr.ndim != indices.ndim:
        raise ValueError("`indices` and `arr` must have the same number of dimensions")

    # use the fancy index
    return arr[_make_along_axis_idx(arr_shape, indices, _axis)]


@local_optimizer(tracks=[CheckParameterValue])
def local_remove_check_parameter(fgraph, node):
    """Rewrite that removes Aeppl's CheckParameterValue

    This is used when compile_rv_inplace
    """
    if isinstance(node.op, CheckParameterValue):
        return [node.inputs[0]]


@local_optimizer(tracks=[CheckParameterValue])
def local_check_parameter_to_ninf_switch(fgraph, node):
    if isinstance(node.op, CheckParameterValue):
        logp_expr, *logp_conds = node.inputs
        if len(logp_conds) > 1:
            logp_cond = at.all(logp_conds)
        else:
            (logp_cond,) = logp_conds
        out = at.switch(logp_cond, logp_expr, -np.inf)
        out.name = node.op.msg

        if out.dtype != node.outputs[0].dtype:
            out = at.cast(out, node.outputs[0].dtype)

        return [out]


aesara.compile.optdb["canonicalize"].register(
    "local_remove_check_parameter",
    local_remove_check_parameter,
    use_db_name_as_tag=False,
)

aesara.compile.optdb["canonicalize"].register(
    "local_check_parameter_to_ninf_switch",
    local_check_parameter_to_ninf_switch,
    use_db_name_as_tag=False,
)


def compile_pymc(
    inputs, outputs, mode=None, **kwargs
) -> Callable[..., Union[np.ndarray, List[np.ndarray]]]:
    """Use ``aesara.function`` with specialized pymc rewrites always enabled.

    Included rewrites
    -----------------
    random_make_inplace
        Ensures that compiled functions containing random variables will produce new
        samples on each call.
    local_check_parameter_to_ninf_switch
        Replaces Aeppl's CheckParameterValue assertions is logp expressions with Switches
        that return -inf in place of the assert.

    Optional rewrites
    -----------------
    local_remove_check_parameter
        Replaces Aeppl's CheckParameterValue assertions is logp expressions. This is used
        as an alteranative to the default local_check_parameter_to_ninf_switch whenenver
        this function is called within a model context and the model `check_bounds` flag
        is set to False.
    """
    # Create an update mapping of RandomVariable's RNG so that it is automatically
    # updated after every function call
    # TODO: This won't work for variables with InnerGraphs (Scan and OpFromGraph)
    rng_updates = {}
    output_to_list = outputs if isinstance(outputs, (list, tuple)) else [outputs]
    for rv in (
        node
        for node in vars_between(inputs, output_to_list)
        if node.owner and isinstance(node.owner.op, RandomVariable) and node not in inputs
    ):
        rng = rv.owner.inputs[0]
        if not hasattr(rng, "default_update"):
            rng_updates[rng] = rv.owner.outputs[0]

    # If called inside a model context, see if check_bounds flag is set to False
    try:
        from pymc.model import modelcontext

        model = modelcontext(None)
        check_bounds = model.check_bounds
    except TypeError:
        check_bounds = True
    check_parameter_opt = (
        "local_check_parameter_to_ninf_switch" if check_bounds else "local_remove_check_parameter"
    )

    mode = get_mode(mode)
    opt_qry = mode.provided_optimizer.including("random_make_inplace", check_parameter_opt)
    mode = Mode(linker=mode.linker, optimizer=opt_qry)
    aesara_function = aesara.function(
        inputs,
        outputs,
        updates={**rng_updates, **kwargs.pop("updates", {})},
        mode=mode,
        **kwargs,
    )
    return aesara_function
