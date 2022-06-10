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
    Sequence,
    Set,
    Tuple,
    Union,
)

import aesara
import aesara.tensor as at
import numpy as np
import scipy.sparse as sps

from aeppl.abstract import MeasurableVariable
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
from aesara.tensor.random.var import (
    RandomGeneratorSharedVariable,
    RandomStateSharedVariable,
)
from aesara.tensor.shape import SpecifyShape
from aesara.tensor.sharedvar import SharedVariable
from aesara.tensor.subtensor import AdvancedIncSubtensor, AdvancedIncSubtensor1
from aesara.tensor.var import TensorConstant, TensorVariable

from pymc.exceptions import ShapeError
from pymc.vartypes import continuous_types, isgenerator, typefilter

PotentialShapeType = Union[int, np.ndarray, Sequence[Union[int, Variable]], TensorVariable]


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
    "convert_observed_data",
]


def convert_observed_data(data):
    """Convert user provided dataset to accepted formats."""

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
    assert rv.owner.op is not None
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


scalar_identity = IdentityOp(scalar.upgrade_to_float, name="scalar_identity")
identity = Elemwise(scalar_identity, name="identity")


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


def find_rng_nodes(
    variables: Iterable[Variable],
) -> List[Union[RandomStateSharedVariable, RandomGeneratorSharedVariable]]:
    """Return RNG variables in a graph"""
    return [
        node
        for node in graph_inputs(variables)
        if isinstance(node, (RandomStateSharedVariable, RandomGeneratorSharedVariable))
    ]


SeedSequenceSeed = Optional[Union[int, Sequence[int], np.ndarray, np.random.SeedSequence]]


def reseed_rngs(
    rngs: Sequence[SharedVariable],
    seed: SeedSequenceSeed,
) -> None:
    """Create a new set of RandomState/Generator for each rng based on a seed"""
    bit_generators = [
        np.random.PCG64(sub_seed) for sub_seed in np.random.SeedSequence(seed).spawn(len(rngs))
    ]
    for rng, bit_generator in zip(rngs, bit_generators):
        new_rng: Union[np.random.RandomState, np.random.Generator]
        if isinstance(rng, at.random.var.RandomStateSharedVariable):
            new_rng = np.random.RandomState(bit_generator)
        else:
            new_rng = np.random.Generator(bit_generator)
        rng.set_value(new_rng, borrow=True)


def compile_pymc(
    inputs,
    outputs,
    random_seed: SeedSequenceSeed = None,
    mode=None,
    **kwargs,
) -> Callable[..., Union[np.ndarray, List[np.ndarray]]]:
    """Use ``aesara.function`` with specialized pymc rewrites always enabled.

    This function also ensures shared RandomState/Generator used by RandomVariables
    in the graph are updated across calls, to ensure independent draws.

    Parameters
    ----------
    inputs: list of TensorVariables, optional
        Inputs of the compiled Aesara function
    outputs: list of TensorVariables, optional
        Outputs of the compiled Aesara function
    random_seed: int, array-like of int or SeedSequence, optional
        Seed used to override any RandomState/Generator shared variables in the graph.
        If not specified, the value of original shared variables will still be overwritten.
    mode: optional
        Aesara mode used to compile the function

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
    rng_updates = {}
    output_to_list = outputs if isinstance(outputs, (list, tuple)) else [outputs]
    for random_var in (
        var
        for var in vars_between(inputs, output_to_list)
        if var.owner
        and isinstance(var.owner.op, (RandomVariable, MeasurableVariable))
        and var not in inputs
    ):
        # All nodes in `vars_between(inputs, outputs)` have owners.
        # But mypy doesn't know, so we just assert it:
        assert random_var.owner.op is not None
        if isinstance(random_var.owner.op, RandomVariable):
            rng = random_var.owner.inputs[0]
            if not hasattr(rng, "default_update"):
                rng_updates[rng] = random_var.owner.outputs[0]
            else:
                rng_updates[rng] = rng.default_update
        else:
            update_fn = getattr(random_var.owner.op, "update", None)
            if update_fn is not None:
                rng_updates.update(update_fn(random_var.owner))

    # We always reseed random variables as this provides RNGs with no chances of collision
    if rng_updates:
        reseed_rngs(rng_updates.keys(), random_seed)

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
