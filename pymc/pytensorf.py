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

import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
import scipy.sparse as sps

from pytensor import scalar
from pytensor.compile import Function, Mode, get_mode
from pytensor.gradient import grad
from pytensor.graph import Type, node_rewriter, rewrite_graph
from pytensor.graph.basic import (
    Apply,
    Constant,
    Variable,
    clone_get_equiv,
    graph_inputs,
    walk,
)
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import Op
from pytensor.scalar.basic import Cast
from pytensor.scan.op import Scan
from pytensor.tensor.basic import _as_tensor_variable
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.random.type import RandomType
from pytensor.tensor.random.var import (
    RandomGeneratorSharedVariable,
    RandomStateSharedVariable,
)
from pytensor.tensor.rewriting.basic import topo_constant_folding
from pytensor.tensor.rewriting.shape import ShapeFeature
from pytensor.tensor.sharedvar import SharedVariable, TensorSharedVariable
from pytensor.tensor.subtensor import AdvancedIncSubtensor, AdvancedIncSubtensor1
from pytensor.tensor.variable import TensorConstant, TensorVariable

from pymc.exceptions import NotConstantValueError
from pymc.logprob.transforms import RVTransform
from pymc.logprob.utils import CheckParameterValue
from pymc.util import makeiter
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
    "convert_observed_data",
    "compile_pymc",
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


@_as_tensor_variable.register(pd.Series)
@_as_tensor_variable.register(pd.DataFrame)
def dataframe_to_tensor_variable(df: pd.DataFrame, *args, **kwargs) -> TensorVariable:
    return pt.as_tensor_variable(df.to_numpy(), *args, **kwargs)


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
    stop_at_vars: Optional[Set[TensorVariable]] = None,
    expand_fn: Callable[[TensorVariable], Iterable[TensorVariable]] = lambda var: [],
) -> Generator[TensorVariable, None, None]:
    """Walk model graphs and yield their nodes.

    Parameters
    ----------
    graphs
        The graphs to walk.
    stop_at_vars
        A list of variables at which the walk will terminate.
    expand_fn
        A function that returns the next variable(s) to be traversed.
    """
    if stop_at_vars is None:
        stop_at_vars = set()

    def expand(var):
        new_vars = expand_fn(var)

        if var.owner and var not in stop_at_vars:
            new_vars.extend(reversed(var.owner.inputs))

        return new_vars

    yield from walk(graphs, expand, bfs=False)


def _replace_vars_in_graphs(
    graphs: Iterable[TensorVariable],
    replacement_fn: Callable[[TensorVariable], Dict[TensorVariable, TensorVariable]],
    **kwargs,
) -> Tuple[List[TensorVariable], Dict[TensorVariable, TensorVariable]]:
    """Replace variables in graphs.

    This will *not* recompute test values.

    Parameters
    ----------
    graphs
        The graphs in which random variables are to be replaced.
    replacement_fn
        A callable called on each graph output that populates a replacement dictionary and returns
        nodes that should be investigated further.

    Returns
    -------
    Tuple containing the transformed graphs and a ``dict`` of the replacements
    that were made.
    """
    replacements = {}

    def expand_replace(var):
        new_nodes = []
        if var.owner:
            # Call replacement_fn to update replacements dict inplace and, optionally,
            # specify new nodes that should also be walked for replacements. This
            # includes `value` variables that are not simple input variables, and may
            # contain other `random` variables in their graphs (e.g., IntervalTransform)
            new_nodes.extend(replacement_fn(var, replacements))
        return new_nodes

    # This iteration populates the replacements
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

        # replacements have to be done in reverse topological order so that nested
        # expressions get recursively replaced correctly
        toposort = fg.toposort()
        sorted_replacements = sorted(
            tuple(replacements.items()),
            # Root inputs don't have owner, we give them negative priority -1
            key=lambda pair: toposort.index(pair[0].owner) if pair[0].owner is not None else -1,
            reverse=True,
        )
        fg.replace_all(sorted_replacements, import_missing=True)

        graphs = list(fg.outputs)

    return graphs, replacements


def rvs_to_value_vars(
    graphs: Iterable[Variable],
    apply_transforms: bool = True,
    **kwargs,
) -> List[Variable]:
    """Clone and replace random variables in graphs with their value variables.

    This will *not* recompute test values in the resulting graphs.

    Parameters
    ----------
    graphs
        The graphs in which to perform the replacements.
    apply_transforms
        If ``True``, apply each value variable's transform.
    """
    warnings.warn(
        "rvs_to_value_vars is deprecated. Use model.replace_rvs_by_values instead",
        FutureWarning,
    )

    def populate_replacements(
        random_var: TensorVariable, replacements: Dict[TensorVariable, TensorVariable]
    ) -> List[TensorVariable]:
        # Populate replacements dict with {rv: value} pairs indicating which graph
        # RVs should be replaced by what value variables.

        value_var = getattr(
            random_var.tag, "observations", getattr(random_var.tag, "value_var", None)
        )

        # No value variable to replace RV with
        if value_var is None:
            return []

        transform = getattr(value_var.tag, "transform", None)
        if transform is not None and apply_transforms:
            # We want to replace uses of the RV by the back-transformation of its value
            value_var = transform.backward(value_var, *random_var.owner.inputs)

        replacements[random_var] = value_var

        # Also walk the graph of the value variable to make any additional replacements
        # if that is not a simple input variable
        return [value_var]

    # Clone original graphs
    inputs = [i for i in graph_inputs(graphs) if not isinstance(i, Constant)]
    equiv = clone_get_equiv(inputs, graphs, False, False, {})
    graphs = [equiv[n] for n in graphs]

    graphs, _ = _replace_vars_in_graphs(
        graphs,
        replacement_fn=populate_replacements,
        **kwargs,
    )

    return graphs


def replace_rvs_by_values(
    graphs: Sequence[TensorVariable],
    *,
    rvs_to_values: Dict[TensorVariable, TensorVariable],
    rvs_to_transforms: Optional[Dict[TensorVariable, RVTransform]] = None,
    **kwargs,
) -> List[TensorVariable]:
    """Clone and replace random variables in graphs with their value variables.

    This will *not* recompute test values in the resulting graphs.

    Parameters
    ----------
    graphs
        The graphs in which to perform the replacements.
    rvs_to_values
        Mapping between the original graph RVs and respective value variables
    rvs_to_transforms, optional
        Mapping between the original graph RVs and respective value transforms
    """

    # Clone original graphs so that we don't modify variables in place
    inputs = [i for i in graph_inputs(graphs) if not isinstance(i, Constant)]
    equiv = clone_get_equiv(inputs, graphs, False, False, {})
    graphs = [equiv[n] for n in graphs]

    # Get needed mappings for equivalent cloned variables
    equiv_rvs_to_values = {}
    equiv_rvs_to_transforms = {}
    for rv, value in rvs_to_values.items():
        equiv_rv = equiv.get(rv, rv)
        equiv_rvs_to_values[equiv_rv] = equiv.get(value, value)
        if rvs_to_transforms is not None:
            equiv_rvs_to_transforms[equiv_rv] = rvs_to_transforms[rv]

    def poulate_replacements(rv, replacements):
        # Populate replacements dict with {rv: value} pairs indicating which graph
        # RVs should be replaced by what value variables.

        # No value variable to replace RV with
        value = equiv_rvs_to_values.get(rv, None)
        if value is None:
            return []

        if rvs_to_transforms is not None:
            transform = equiv_rvs_to_transforms.get(rv, None)
            if transform is not None:
                # We want to replace uses of the RV by the back-transformation of its value
                value = transform.backward(value, *rv.owner.inputs)
                # The value may have a less precise type than the rv. In this case
                # filter_variable will add a SpecifyShape to ensure they are consistent
                value = rv.type.filter_variable(value, allow_convert=True)
                value.name = rv.name

        replacements[rv] = value
        # Also walk the graph of the value variable to make any additional
        # replacements if that is not a simple input variable
        return [value]

    graphs, _ = _replace_vars_in_graphs(
        graphs,
        replacement_fn=poulate_replacements,
        **kwargs,
    )

    return graphs


def inputvars(a):
    """
    Get the inputs into PyTensor variables

    Parameters
    ----------
        a: PyTensor variable

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
    Get the continuous inputs into PyTensor variables

    Parameters
    ----------
        a: PyTensor variable

    Returns
    -------
        r: list of tensor variables that are continuous inputs
    """
    return typefilter(inputvars(a), continuous_types)


def floatX(X):
    """
    Convert an PyTensor tensor or numpy array to pytensor.config.floatX type.
    """
    try:
        return X.astype(pytensor.config.floatX)
    except AttributeError:
        # Scalar passed
        return np.asarray(X, dtype=pytensor.config.floatX)


_conversion_map = {"float64": "int32", "float32": "int16", "float16": "int8", "float8": "int8"}


def intX(X):
    """
    Convert a pytensor tensor or numpy array to pytensor.tensor.int32 type.
    """
    intX = _conversion_map[pytensor.config.floatX]
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
PyTensor derivative functions
"""


def gradient1(f, v):
    """flat gradient of f wrt v"""
    return pt.flatten(grad(f, v, disconnected_inputs="warn"))


empty_gradient = pt.zeros(0, dtype="float32")


def gradient(f, vars=None):
    if vars is None:
        vars = cont_inputs(f)

    if vars:
        return pt.concatenate([gradient1(f, v) for v in vars], axis=0)
    else:
        return empty_gradient


def jacobian1(f, v):
    """jacobian of f wrt v"""
    f = pt.flatten(f)
    idx = pt.arange(f.shape[0], dtype="int32")

    def grad_i(i):
        return gradient1(f[i], v)

    return pytensor.map(grad_i, idx)[0]


def jacobian(f, vars=None):
    if vars is None:
        vars = cont_inputs(f)

    if vars:
        return pt.concatenate([jacobian1(f, v) for v in vars], axis=1)
    else:
        return empty_gradient


def jacobian_diag(f, x):
    idx = pt.arange(f.shape[0], dtype="int32")

    def grad_ii(i, f, x):
        return grad(f[i], x)[i]

    return pytensor.scan(
        grad_ii, sequences=[idx], n_steps=f.shape[0], non_sequences=[f, x], name="jacobian_diag"
    )[0]


@pytensor.config.change_flags(compute_test_value="ignore")
def hessian(f, vars=None):
    return -jacobian(gradient(f, vars), vars)


@pytensor.config.change_flags(compute_test_value="ignore")
def hessian_diag1(f, v):
    g = gradient1(f, v)
    idx = pt.arange(g.shape[0], dtype="int32")

    def hess_ii(i):
        return gradient1(g[i], v)[i]

    return pytensor.map(hess_ii, idx)[0]


@pytensor.config.change_flags(compute_test_value="ignore")
def hessian_diag(f, vars=None):
    if vars is None:
        vars = cont_inputs(f)

    if vars:
        return -pt.concatenate([hessian_diag1(f, v) for v in vars], axis=0)
    else:
        return empty_gradient


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
        var: pytensor.shared(point[var.name], var.name + "_shared", shape=var.type.shape)
        for var in othervars
    }


def join_nonshared_inputs(
    point: Dict[str, np.ndarray],
    outputs: List[TensorVariable],
    inputs: List[TensorVariable],
    shared_inputs: Optional[Dict[TensorVariable, TensorSharedVariable]] = None,
    make_inputs_shared: bool = False,
) -> Tuple[List[TensorVariable], TensorVariable]:
    """
    Create new outputs and input TensorVariables where the non-shared inputs are joined
    in a single raveled vector input.

    Parameters
    ----------
    point : dict of {str : array_like}
        Dictionary that maps each input variable name to a numerical variable. The values
        are used to extract the shape of each input variable to establish a correct
        mapping between joined and original inputs. The shape of each variable is
        assumed to be fixed.
    outputs : list of TensorVariable
        List of output TensorVariables whose non-shared inputs will be replaced
        by a joined vector input.
    inputs : list of TensorVariable
        List of input TensorVariables which will be replaced by a joined vector input.
    shared_inputs : dict of {TensorVariable : TensorSharedVariable}, optional
        Dict of TensorVariable and their associated TensorSharedVariable in
        subgraph replacement.
    make_inputs_shared : bool, default False
        Whether to make the joined vector input a shared variable.

    Returns
    -------
    new_outputs : list of TensorVariable
        List of new outputs `outputs` TensorVariables that depend on `joined_inputs` and new shared variables as inputs.
    joined_inputs : TensorVariable
        Joined input vector TensorVariable for the `new_outputs`

    Examples
    --------
    Join the inputs of a simple PyTensor graph.

    .. code-block:: python

        import pytensor.tensor as pt
        import numpy as np

        from pymc.pytensorf import join_nonshared_inputs

        # Original non-shared inputs
        x = pt.scalar("x")
        y = pt.vector("y")
        # Original output
        out = x + y
        print(out.eval({x: np.array(1), y: np.array([1, 2, 3])})) # [2, 3, 4]

        # New output and inputs
        [new_out], joined_inputs = join_nonshared_inputs(
            point={ # Only shapes matter
                "x": np.zeros(()),
                "y": np.zeros(3),
            },
            outputs=[out],
            inputs=[x, y],
        )
        print(new_out.eval({
            joined_inputs: np.array([1, 1, 2, 3]),
        })) # [2, 3, 4]

    Join the input value variables of a model logp.

    .. code-block:: python

        import pymc as pm

        with pm.Model() as model:
            mu_pop = pm.Normal("mu_pop")
            sigma_pop = pm.HalfNormal("sigma_pop")
            mu = pm.Normal("mu", mu_pop, sigma_pop, shape=(3, ))

            y = pm.Normal("y", mu, 1.0, observed=[0, 1, 2])

        print(model.compile_logp()({
            "mu_pop": 0,
            "sigma_pop_log__": 1,
            "mu": [0, 1, 2],
        })) # -12.691227342634292

        initial_point = model.initial_point()
        inputs = model.value_vars

        [logp], joined_inputs = join_nonshared_inputs(
            point=initial_point,
            outputs=[model.logp()],
            inputs=inputs,
        )

        print(logp.eval({
            joined_inputs: [0, 1, 0, 1, 2],
        })) # -12.691227342634292

    Same as above but with the `mu_pop` value variable being shared.

    .. code-block:: python

        from pytensor import shared

        mu_pop_input, *other_inputs = inputs
        shared_mu_pop_input = shared(0.0)

        [logp], other_joined_inputs = join_nonshared_inputs(
            point=initial_point,
            outputs=[model.logp()],
            inputs=other_inputs,
            shared_inputs={
                mu_pop_input: shared_mu_pop_input
            },
        )

        print(logp.eval({
            other_joined_inputs: [1, 0, 1, 2],
        })) # -12.691227342634292
    """
    if not inputs:
        raise ValueError("Empty list of input variables.")

    raveled_inputs = pt.concatenate([var.ravel() for var in inputs])

    if not make_inputs_shared:
        tensor_type = raveled_inputs.type
        joined_inputs = tensor_type("joined_inputs")
    else:
        joined_values = np.concatenate([point[var.name].ravel() for var in inputs])
        joined_inputs = pytensor.shared(joined_values, "joined_inputs")

    if pytensor.config.compute_test_value != "off":
        joined_inputs.tag.test_value = raveled_inputs.tag.test_value

    replace: Dict[TensorVariable, TensorVariable] = {}
    last_idx = 0
    for var in inputs:
        shape = point[var.name].shape
        arr_len = np.prod(shape, dtype=int)
        replace[var] = joined_inputs[last_idx : last_idx + arr_len].reshape(shape).astype(var.dtype)
        last_idx += arr_len

    if shared_inputs is not None:
        replace.update(shared_inputs)

    new_outputs = [
        pytensor.clone_replace(output, replace, rebuild_strict=False) for output in outputs
    ]
    return new_outputs, joined_inputs


class PointFunc:
    """Wraps so a function so it takes a dict of arguments instead of arguments."""

    def __init__(self, f):
        self.f = f

    def __call__(self, state):
        return self.f(**state)


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
        return pytensor.clone_replace(self.tensor, {oldinput: input}, rebuild_strict=False)


class GeneratorOp(Op):
    """
    Generator Op is designed for storing python generators inside pytensor graph.

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

    __call__ = pytensor.config.change_flags(compute_test_value="off")(Op.__call__)

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


def floatX_array(x):
    return floatX(np.array(x))


def ix_(*args):
    """
    PyTensor np.ix_ analog

    See numpy.lib.index_tricks.ix_ for reference
    """
    out = []
    nd = len(args)
    for k, new in enumerate(args):
        if new is None:
            out.append(slice(None))
        new = pt.as_tensor(new)
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


@node_rewriter(tracks=[CheckParameterValue])
def local_remove_check_parameter(fgraph, node):
    """Rewrite that removes CheckParameterValue

    This is used when compile_rv_inplace
    """
    if isinstance(node.op, CheckParameterValue):
        return [node.inputs[0]]


@node_rewriter(tracks=[CheckParameterValue])
def local_check_parameter_to_ninf_switch(fgraph, node):
    if not node.op.can_be_replaced_by_ninf:
        return None

    logp_expr, *logp_conds = node.inputs
    if len(logp_conds) > 1:
        logp_cond = pt.all(logp_conds)
    else:
        (logp_cond,) = logp_conds
    out = pt.switch(logp_cond, logp_expr, -np.inf)
    out.name = node.op.msg

    if out.dtype != node.outputs[0].dtype:
        out = pt.cast(out, node.outputs[0].dtype)

    return [out]


pytensor.compile.optdb["canonicalize"].register(
    "local_remove_check_parameter",
    local_remove_check_parameter,
    use_db_name_as_tag=False,
)

pytensor.compile.optdb["canonicalize"].register(
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


def replace_rng_nodes(outputs: Sequence[TensorVariable]) -> Sequence[TensorVariable]:
    """Replace any RNG nodes upstream of outputs by new RNGs of the same type

    This can be used when combining a pre-existing graph with a cloned one, to ensure
    RNGs are unique across the two graphs.
    """
    rng_nodes = find_rng_nodes(outputs)

    # Nothing to do here
    if not rng_nodes:
        return outputs

    graph = FunctionGraph(outputs=outputs, clone=False)
    new_rng_nodes: List[Union[np.random.RandomState, np.random.Generator]] = []
    for rng_node in rng_nodes:
        rng_cls: type
        if isinstance(rng_node, pt.random.var.RandomStateSharedVariable):
            rng_cls = np.random.RandomState
        else:
            rng_cls = np.random.Generator
        new_rng_nodes.append(pytensor.shared(rng_cls(np.random.PCG64())))
    graph.replace_all(zip(rng_nodes, new_rng_nodes), import_missing=True)
    return graph.outputs


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
        if isinstance(rng, pt.random.var.RandomStateSharedVariable):
            new_rng = np.random.RandomState(bit_generator)
        else:
            new_rng = np.random.Generator(bit_generator)
        rng.set_value(new_rng, borrow=True)


def collect_default_updates(
    outputs: Sequence[Variable],
    *,
    inputs: Optional[Sequence[Variable]] = None,
    must_be_shared: bool = True,
) -> Dict[Variable, Variable]:
    """Collect default update expression for shared-variable RNGs used by RVs between inputs and outputs.

    Parameters
    ----------
    outputs: list of PyTensor variables
        List of variables in which graphs default updates will be collected.
    inputs: list of PyTensor variables, optional
        Input nodes above which default updates should not be collected.
        When not provided, search will include top level inputs (roots).
    must_be_shared: bool, default True
        Used internally by PyMC. Whether updates should be collected for non-shared
        RNG input variables. This is used to collect update expressions for inner graphs.

    Examples
    --------
    .. code:: python
        import pymc as pm
        from pytensor.scan import scan
        from pymc.pytensorf import collect_default_updates

        def scan_step(xtm1):
            x = xtm1 + pm.Normal.dist()
            x_update = collect_default_updates([x])
            return x, x_update

        x0 = pm.Normal.dist()

        xs, updates = scan(
            fn=scan_step,
            outputs_info=[x0],
            n_steps=10,
        )

        # PyMC makes use of the updates to seed xs properly.
        # Without updates, it would raise an error.
        xs_draws = pm.draw(xs, draws=10)

    """
    # Avoid circular import
    from pymc.distributions.distribution import SymbolicRandomVariable

    def find_default_update(clients, rng: Variable) -> Union[None, Variable]:
        rng_clients = clients.get(rng, None)

        # Root case, RNG is not used elsewhere
        if not rng_clients:
            return rng

        if len(rng_clients) > 1:
            warnings.warn(
                f"RNG Variable {rng} has multiple clients. This is likely an inconsistent random graph.",
                UserWarning,
            )
            return None

        [client, _] = rng_clients[0]

        # RNG is an output of the function, this is not a problem
        if client == "output":
            return rng

        # RNG is used by another operator, which should output an update for the RNG
        if isinstance(client.op, RandomVariable):
            # RandomVariable first output is always the update of the input RNG
            next_rng = client.outputs[0]

        elif isinstance(client.op, SymbolicRandomVariable):
            # SymbolicRandomVariable have an explicit method that returns an
            # update mapping for their RNG(s)
            next_rng = client.op.update(client).get(rng)
            if next_rng is None:
                raise ValueError(
                    f"No update found for at least one RNG used in SymbolicRandomVariable Op {client.op}"
                )
        elif isinstance(client.op, Scan):
            # Check if any shared output corresponds to the RNG
            rng_idx = client.inputs.index(rng)
            io_map = client.op.get_oinp_iinp_iout_oout_mappings()["outer_out_from_outer_inp"]
            out_idx = io_map.get(rng_idx, -1)
            if out_idx != -1:
                next_rng = client.outputs[out_idx]
            else:  # No break
                raise ValueError(
                    f"No update found for at least one RNG used in Scan Op {client.op}.\n"
                    "You can use `pytensorf.collect_default_updates` inside the Scan function to return updates automatically."
                )
        else:
            # We don't know how this RNG should be updated (e.g., OpFromGraph).
            # The user should provide an update manually
            return None

        # Recurse until we find final update for RNG
        return find_default_update(clients, next_rng)

    if inputs is None:
        inputs = []

    outputs = makeiter(outputs)
    fg = FunctionGraph(outputs=outputs, clone=False)
    clients = fg.clients

    rng_updates = {}
    # Iterate over input RNGs. Only consider shared RNGs if `must_be_shared==True`
    for input_rng in (
        inp
        for inp in graph_inputs(outputs, blockers=inputs)
        if (
            (not must_be_shared or isinstance(inp, SharedVariable))
            and isinstance(inp.type, RandomType)
        )
    ):
        # Even if an explicit default update is provided, we call it to
        # issue any warnings about invalid random graphs.
        default_update = find_default_update(clients, input_rng)

        # Respect default update if provided
        if getattr(input_rng, "default_update", None):
            rng_updates[input_rng] = input_rng.default_update
        else:
            if default_update is not None:
                rng_updates[input_rng] = default_update

    return rng_updates


def compile_pymc(
    inputs,
    outputs,
    random_seed: SeedSequenceSeed = None,
    mode=None,
    **kwargs,
) -> Function:
    """Use ``pytensor.function`` with specialized pymc rewrites always enabled.

    This function also ensures shared RandomState/Generator used by RandomVariables
    in the graph are updated across calls, to ensure independent draws.

    Parameters
    ----------
    inputs: list of TensorVariables, optional
        Inputs of the compiled PyTensor function
    outputs: list of TensorVariables, optional
        Outputs of the compiled PyTensor function
    random_seed: int, array-like of int or SeedSequence, optional
        Seed used to override any RandomState/Generator shared variables in the graph.
        If not specified, the value of original shared variables will still be overwritten.
    mode: optional
        PyTensor mode used to compile the function

    Included rewrites
    -----------------
    random_make_inplace
        Ensures that compiled functions containing random variables will produce new
        samples on each call.
    local_check_parameter_to_ninf_switch
        Replaces CheckParameterValue assertions is logp expressions with Switches
        that return -inf in place of the assert.

    Optional rewrites
    -----------------
    local_remove_check_parameter
        Replaces CheckParameterValue assertions is logp expressions. This is used
        as an alteranative to the default local_check_parameter_to_ninf_switch whenenver
        this function is called within a model context and the model `check_bounds` flag
        is set to False.
    """
    # Create an update mapping of RandomVariable's RNG so that it is automatically
    # updated after every function call
    rng_updates = collect_default_updates(inputs=inputs, outputs=outputs)

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
    pytensor_function = pytensor.function(
        inputs,
        outputs,
        updates={**rng_updates, **kwargs.pop("updates", {})},
        mode=mode,
        **kwargs,
    )
    return pytensor_function


def constant_fold(
    xs: Sequence[TensorVariable], raise_not_constant: bool = True
) -> Tuple[np.ndarray, ...]:
    """Use constant folding to get constant values of a graph.

    Parameters
    ----------
    xs: Sequence of TensorVariable
        The variables that are to be constant folded
    raise_not_constant: bool, default True
        Raises NotConstantValueError if any of the variables cannot be constant folded.
        This should only be disabled with care, as the graphs are cloned before
        attempting constant folding, and any old non-shared inputs will not work with
        the returned outputs
    """
    fg = FunctionGraph(outputs=xs, features=[ShapeFeature()], clone=True)

    folded_xs = rewrite_graph(fg, custom_rewrite=topo_constant_folding).outputs

    if raise_not_constant and not all(isinstance(folded_x, Constant) for folded_x in folded_xs):
        raise NotConstantValueError

    return tuple(
        folded_x.data if isinstance(folded_x, Constant) else folded_x for folded_x in folded_xs
    )


def rewrite_pregrad(graph):
    """Apply simplifying or stabilizing rewrites to graph that are safe to use
    pre-grad.
    """
    return rewrite_graph(graph, include=("canonicalize", "stabilize"))


class StringType(Type[str]):
    def clone(self, **kwargs):
        return type(self)()

    def filter(self, x, strict=False, allow_downcast=None):
        if isinstance(x, str):
            return x
        else:
            raise TypeError("Expected a string!")

    def __str__(self):
        return "string"

    @staticmethod
    def may_share_memory(a, b):
        return isinstance(a, str) and a is b


stringtype = StringType()


class StringConstant(Constant):
    pass


@pytensor._as_symbolic.register(str)
def as_symbolic_string(x, **kwargs):
    return StringConstant(stringtype, x)


def toposort_replace(
    fgraph: FunctionGraph, replacements: Sequence[Tuple[Variable, Variable]], reverse: bool = False
) -> None:
    """Replace multiple variables in topological order."""
    toposort = fgraph.toposort()
    sorted_replacements = sorted(
        replacements,
        key=lambda pair: toposort.index(pair[0].owner) if pair[0].owner else -1,
        reverse=reverse,
    )
    fgraph.replace_all(sorted_replacements, import_missing=True)
