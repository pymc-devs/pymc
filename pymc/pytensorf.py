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
import warnings

from collections.abc import Iterable, Sequence
from typing import cast

import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
import scipy.sparse as sps

from pytensor.compile import Function, Mode, get_mode
from pytensor.compile.builders import OpFromGraph
from pytensor.gradient import grad
from pytensor.graph import Type, rewrite_graph
from pytensor.graph.basic import (
    Apply,
    Constant,
    Variable,
    clone_get_equiv,
    equal_computations,
)
from pytensor.graph.fg import FunctionGraph, Output
from pytensor.graph.op import HasInnerGraph
from pytensor.graph.traversal import explicit_graph_inputs, graph_inputs, walk
from pytensor.scalar.basic import Cast
from pytensor.scan.op import Scan
from pytensor.tensor.basic import _as_tensor_variable
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.random.op import RandomVariable, RNGConsumerOp
from pytensor.tensor.random.type import RandomType
from pytensor.tensor.random.var import RandomGeneratorSharedVariable
from pytensor.tensor.rewriting.basic import topo_unconditional_constant_folding
from pytensor.tensor.rewriting.shape import ShapeFeature
from pytensor.tensor.sharedvar import SharedVariable
from pytensor.tensor.subtensor import AdvancedIncSubtensor, AdvancedIncSubtensor1
from pytensor.tensor.variable import TensorVariable

from pymc.exceptions import NotConstantValueError
from pymc.util import makeiter
from pymc.vartypes import continuous_types, isgenerator, typefilter

PotentialShapeType = int | np.ndarray | Sequence[int | Variable] | TensorVariable

__all__ = [
    "CallableTensor",
    "compile",
    "cont_inputs",
    "convert_data",
    "convert_observed_data",
    "floatX",
    "gradient",
    "hessian",
    "hessian_diag",
    "inputvars",
    "intX",
    "jacobian",
    "join_nonshared_inputs",
    "make_shared_replacements",
]


def convert_observed_data(data) -> np.ndarray | Variable:
    """Convert user provided dataset to accepted formats."""
    if isgenerator(data):
        raise TypeError("Data passed to `observed` cannot be a generator.")
    return convert_data(data)


def convert_data(data) -> np.ndarray | Variable:
    ret: np.ndarray | Variable
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
    else:
        ret = np.asarray(data)

    # Data without dtype info is converted to float arrays by default.
    # This is the most common case for simple examples.
    if not hasattr(data, "dtype"):
        return floatX(ret)
    # Otherwise we only convert the precision.
    return smarttypeX(ret)


@_as_tensor_variable.register(pd.Series)
@_as_tensor_variable.register(pd.DataFrame)
def dataframe_to_tensor_variable(df: pd.DataFrame, *args, **kwargs) -> TensorVariable:
    return pt.as_tensor_variable(df.to_numpy(), *args, **kwargs)


_cheap_eval_mode = Mode(linker="py", optimizer="minimum_compile")


def extract_obs_data(x: TensorVariable) -> np.ndarray:
    """Extract data from observed symbolic variables.

    Raises
    ------
    TypeError

    """
    # TODO: These data functions should be in data.py or model/core.py
    from pymc.data import MinibatchOp

    if isinstance(x, Constant):
        return x.data
    if isinstance(x, SharedVariable):
        return x.get_value()
    if x.owner is not None:
        if isinstance(x.owner.op, Elemwise) and isinstance(x.owner.op.scalar_op, Cast):
            array_data = extract_obs_data(x.owner.inputs[0])
            return array_data.astype(x.type.dtype)
        if isinstance(x.owner.op, MinibatchOp):
            return extract_obs_data(x.owner.inputs[x.owner.outputs.index(x)])
        if isinstance(x.owner.op, AdvancedIncSubtensor | AdvancedIncSubtensor1):
            array_data = extract_obs_data(x.owner.inputs[0])
            mask_idx = tuple(extract_obs_data(i) for i in x.owner.inputs[2:])
            mask = np.zeros_like(array_data)
            mask[mask_idx] = 1
            return np.ma.MaskedArray(array_data, mask)

    if not len(list(explicit_graph_inputs(x))) and not rvs_in_graph(x):
        return x.eval(mode=_cheap_eval_mode)

    raise TypeError(f"Data cannot be extracted from {x}")


def expand_inner_graph(r):
    if (node := r.owner) is not None:
        inputs = list(reversed(node.inputs))

        if isinstance(node.op, HasInnerGraph):
            inputs += node.op.inner_outputs

        return inputs


def rvs_in_graph(vars: Variable | Sequence[Variable], rv_ops=None) -> set[Variable]:
    """Assert that there are no random nodes in a graph."""
    return {
        var
        for var in walk(makeiter(vars), expand_inner_graph, False)
        if (var.owner and isinstance(var.owner.op, RNGConsumerOp))
    }


def replace_vars_in_graphs(
    graphs: Iterable[Variable],
    replacements: dict[Variable, Variable],
) -> list[Variable]:
    """Replace variables in graphs.

    Graphs are cloned and not modified in place, unless the replacement expressions include variables from the original graphs.

    """
    # Clone graphs and get equivalences
    inputs = [i for i in graph_inputs(graphs) if not isinstance(i, Constant)]
    memo = {k: k for k in replacements.keys()}
    equiv = clone_get_equiv(inputs, graphs, False, False, memo)

    fg = FunctionGraph(
        [equiv[i] for i in inputs],
        [equiv[o] for o in graphs],
        clone=False,
    )

    # Filter replacement keys that are actually present in the graph
    vars = fg.variables
    final_replacements = tuple((k, v) for k, v in replacements.items() if k in vars)

    # Replacements have to be done in reverse topological order so that nested
    # expressions get recursively replaced correctly
    toposort_replace(fg, final_replacements, reverse=True)

    return list(fg.outputs)


def inputvars(a):
    """
    Get the inputs into PyTensor variables.

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
        if isinstance(v, Variable) and not isinstance(v, Constant | SharedVariable)
    ]


def cont_inputs(a):
    """
    Get the continuous inputs into PyTensor variables.

    NOTE: No particular order is guaranteed across PyTensor versions

    Parameters
    ----------
        a: PyTensor variable

    Returns
    -------
        r: list of tensor variables that are continuous inputs.
    """
    return typefilter(explicit_graph_inputs(a), continuous_types)


def floatX(X):
    """Convert a PyTensor tensor or numpy array to pytensor.config.floatX type."""
    try:
        return X.astype(pytensor.config.floatX)
    except AttributeError:
        # Scalar passed
        return np.asarray(X, dtype=pytensor.config.floatX)


_conversion_map = {"float64": "int32", "float32": "int16", "float16": "int8", "float8": "int8"}


def intX(X):
    """Convert a pytensor tensor or numpy array to pytensor.tensor.int32 type."""
    intX = _conversion_map[pytensor.config.floatX]
    try:
        return X.astype(intX)
    except AttributeError:
        # Scalar passed
        return np.asarray(X, dtype=intX)


def smartfloatX(x):
    """Convert numpy float values to floatX and leaves values of other types unchanged."""
    if str(x.dtype).startswith("float"):
        x = floatX(x)
    return x


def smarttypeX(x):
    if str(x.dtype).startswith("float"):
        x = floatX(x)
    elif str(x.dtype).startswith("int"):
        x = intX(x)
    return x


"""
PyTensor derivative functions
"""


def gradient1(f, v):
    """Flat gradient of f wrt v."""
    return pt.as_tensor(
        grad(f, v, disconnected_inputs="warn"), allow_xtensor_conversion=True
    ).ravel()


empty_gradient = pt.zeros(0, dtype="float32")


def gradient(f, vars=None):
    if vars is None:
        vars = cont_inputs(f)
        if len(vars) > 1:
            raise ValueError(
                "gradient requires vars to be specified when there is more than one input."
            )

    if vars:
        return pt.concatenate([gradient1(f, v) for v in vars], axis=0)
    else:
        return empty_gradient


def jacobian1(f, v):
    """Jacobian of f wrt v."""
    f = pt.flatten(f)
    idx = pt.arange(f.shape[0], dtype="int32")

    def grad_i(i):
        return gradient1(f[i], v)

    return pytensor.map(grad_i, idx, return_updates=False)


def jacobian(f, vars=None):
    if vars is None:
        vars = cont_inputs(f)
        if len(vars) > 1:
            raise ValueError(
                "jacobian requires vars to be specified when there is more than one input."
            )

    if vars:
        return pt.concatenate([jacobian1(f, v) for v in vars], axis=1)
    else:
        return empty_gradient


def jacobian_diag(f, x):
    idx = pt.arange(f.shape[0], dtype="int32")

    def grad_ii(i, f, x):
        return grad(f[i], x)[i]

    return pytensor.scan(
        grad_ii,
        sequences=[idx],
        n_steps=f.shape[0],
        non_sequences=[f, x],
        name="jacobian_diag",
        return_updates=False,
    )


@pytensor.config.change_flags(compute_test_value="ignore")
def hessian(f, vars=None, negate_output=True):
    res = jacobian(gradient(f, vars), vars)
    if negate_output:
        warnings.warn(
            "hessian will stop negating the output in a future version of PyMC.\n"
            "To suppress this warning set `negate_output=False`",
            FutureWarning,
            stacklevel=2,
        )
        res = -res
    return res


@pytensor.config.change_flags(compute_test_value="ignore")
def hessian_diag1(f, v):
    g = gradient1(f, v)
    idx = pt.arange(g.shape[0], dtype="int32")

    def hess_ii(i):
        return gradient1(g[i], v)[i]

    return pytensor.map(hess_ii, idx, return_updates=False)


@pytensor.config.change_flags(compute_test_value="ignore")
def hessian_diag(f, vars=None, negate_output=True):
    if vars is None:
        vars = cont_inputs(f)
        if len(vars) > 1:
            raise ValueError(
                "hessian_diag requires vars to be specified when there is more than one input."
            )

    if vars:
        res = pt.concatenate([hessian_diag1(f, v) for v in vars], axis=0)
        if negate_output:
            warnings.warn(
                "hessian_diag will stop negating the output in a future version of PyMC.\n"
                "To suppress this warning set `negate_output=False`",
                FutureWarning,
                stacklevel=2,
            )
            res = -res
        return res
    else:
        return empty_gradient


def make_shared_replacements(point, vars, model):
    """
    Make shared replacements for all *other* variables than the ones passed.

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
    vars_set = set(vars)
    return {
        var: pytensor.shared(point[var.name], var.name + "_shared", shape=var.type.shape)
        for var in model.value_vars
        if var not in vars_set
    }


def join_nonshared_inputs(
    point: dict[str, np.ndarray],
    outputs: Sequence[Variable],
    inputs: Sequence[Variable],
    shared_inputs: dict[Variable, Variable] | None = None,
    make_inputs_shared: bool = False,
) -> tuple[Sequence[Variable], TensorVariable]:
    """
    Create new outputs and input TensorVariables where the non-shared inputs are joined in a single raveled vector input.

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
        print(out.eval({x: np.array(1), y: np.array([1, 2, 3])}))  # [2, 3, 4]

        # New output and inputs
        [new_out], joined_inputs = join_nonshared_inputs(
            point={  # Only shapes matter
                "x": np.zeros(()),
                "y": np.zeros(3),
            },
            outputs=[out],
            inputs=[x, y],
        )
        print(new_out.eval({joined_inputs: np.array([1, 1, 2, 3])}))  # [2, 3, 4]

    Join the input value variables of a model logp.

    .. code-block:: python

        import pymc as pm

        with pm.Model() as model:
            mu_pop = pm.Normal("mu_pop")
            sigma_pop = pm.HalfNormal("sigma_pop")
            mu = pm.Normal("mu", mu_pop, sigma_pop, shape=(3,))

            y = pm.Normal("y", mu, 1.0, observed=[0, 1, 2])

        print(
            model.compile_logp()(
                {
                    "mu_pop": 0,
                    "sigma_pop_log__": 1,
                    "mu": [0, 1, 2],
                }
            )
        )  # -12.691227342634292

        initial_point = model.initial_point()
        inputs = model.value_vars

        [logp], joined_inputs = join_nonshared_inputs(
            point=initial_point,
            outputs=[model.logp()],
            inputs=inputs,
        )

        print(
            logp.eval(
                {
                    joined_inputs: [0, 1, 0, 1, 2],
                }
            )
        )  # -12.691227342634292

    Same as above but with the `mu_pop` value variable being shared.

    .. code-block:: python

        from pytensor import shared

        mu_pop_input, *other_inputs = inputs
        shared_mu_pop_input = shared(0.0)

        [logp], other_joined_inputs = join_nonshared_inputs(
            point=initial_point,
            outputs=[model.logp()],
            inputs=other_inputs,
            shared_inputs={mu_pop_input: shared_mu_pop_input},
        )

        print(
            logp.eval(
                {
                    other_joined_inputs: [1, 0, 1, 2],
                }
            )
        )  # -12.691227342634292
    """
    if not inputs:
        raise ValueError("Empty list of input variables.")

    raveled_inputs = pt.concatenate(
        [pt.as_tensor(var, allow_xtensor_conversion=True).ravel() for var in inputs]
    )

    if not make_inputs_shared:
        tensor_type = raveled_inputs.type
        joined_inputs = tensor_type("joined_inputs")
    else:
        joined_values = np.concatenate([point[var.name].ravel() for var in inputs])
        joined_inputs = pytensor.shared(joined_values, "joined_inputs")

    if pytensor.config.compute_test_value != "off":
        joined_inputs.tag.test_value = raveled_inputs.tag.test_value

    replace: dict[Variable, Variable] = {}
    last_idx = 0
    for var in inputs:
        shape = point[var.name].shape
        arr_len = np.prod(shape, dtype=int)
        replacement_var = (
            joined_inputs[last_idx : last_idx + arr_len].reshape(shape).astype(var.dtype)
        )
        replace[var] = var.type.filter_variable(replacement_var)
        last_idx += arr_len

    if shared_inputs is not None:
        replace.update(shared_inputs)

    new_outputs = [
        pytensor.clone_replace(output, replace, rebuild_strict=False) for output in outputs
    ]
    return new_outputs, joined_inputs


class PointFunc:
    """Wraps so a function so it takes a dict of arguments instead of arguments."""

    __slots__ = ("f",)

    def __init__(self, f):
        self.f = f

    def __call__(self, state):
        return self.f(**state)

    def dprint(self, **kwrags):
        return self.f.dprint(**kwrags)


class CallableTensor:
    """Turns a symbolic variable with one input into a function that returns symbolic arguments with the one variable replaced with the input."""

    def __init__(self, tensor):
        self.tensor = tensor

    def __call__(self, input):
        """Replace the single input of symbolic variable to be the passed argument.

        Parameters
        ----------
        input: TensorVariable
        """
        (oldinput,) = explicit_graph_inputs(self.tensor)
        return pytensor.clone_replace(self.tensor, {oldinput: input}, rebuild_strict=False)


def ix_(*args):
    """
    PyTensor np.ix_ analog.

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


def find_rng_nodes(
    variables: Iterable[Variable],
) -> list[RandomGeneratorSharedVariable]:
    """Return shared RNG variables in a graph."""
    return [
        node for node in graph_inputs(variables) if isinstance(node, RandomGeneratorSharedVariable)
    ]


def replace_rng_nodes(outputs: Sequence[TensorVariable]) -> list[TensorVariable]:
    """Replace any RNG nodes upstream of outputs by new RNGs of the same type.

    This can be used when combining a pre-existing graph with a cloned one, to ensure
    RNGs are unique across the two graphs.
    """
    rng_nodes = find_rng_nodes(outputs)

    # Nothing to do here
    if not rng_nodes:
        return outputs

    graph = FunctionGraph(outputs=outputs, clone=False)
    new_rng_nodes = [pytensor.shared(np.random.Generator(np.random.PCG64())) for _ in rng_nodes]
    graph.replace_all(zip(rng_nodes, new_rng_nodes), import_missing=True)
    return cast(list[TensorVariable], graph.outputs)


SeedSequenceSeed = None | int | Sequence[int] | np.ndarray | np.random.SeedSequence


def reseed_rngs(
    rngs: Sequence[SharedVariable],
    seed: SeedSequenceSeed,
) -> None:
    """Create a new set of RandomState/Generator for each rng based on a seed."""
    bit_generators = [
        np.random.PCG64(sub_seed) for sub_seed in np.random.SeedSequence(seed).spawn(len(rngs))
    ]
    for rng, bit_generator in zip(rngs, bit_generators):
        rng.set_value(np.random.Generator(bit_generator), borrow=True)


def collect_default_updates_inner_fgraph(node: Apply) -> dict[Variable, Variable]:
    """Collect default updates from node with inner fgraph."""
    op = node.op
    inner_updates = collect_default_updates(
        inputs=op.inner_inputs, outputs=op.inner_outputs, must_be_shared=False
    )

    # Map inner updates to outer inputs/outputs
    updates = {}
    for rng, update in inner_updates.items():
        inp_idx = op.inner_inputs.index(rng)
        out_idx = op.inner_outputs.index(update)
        updates[node.inputs[inp_idx]] = node.outputs[out_idx]

    return updates


def collect_default_updates(
    outputs: Variable | Sequence[Variable],
    *,
    inputs: Sequence[Variable] | None = None,
    must_be_shared: bool = True,
) -> dict[Variable, Variable]:
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
    .. code-block:: python

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

    def find_default_update(clients, rng: Variable) -> None | Variable:
        rng_clients = clients.get(rng, None)

        # Root case, RNG is not used elsewhere
        if not rng_clients:
            return None

        if len(rng_clients) > 1:
            # Multiple clients are techincally fine if they are used in identical operations
            # We check if the default_update of each client would be the same
            all_updates = [
                find_default_update(
                    # Pass version of clients that includes only one the RNG clients at a time
                    clients | {rng: [rng_client]},
                    rng,
                )
                for rng_client in rng_clients
            ]
            updates = [update for update in all_updates if update is not None]
            if not updates:
                return None
            if len(updates) == 1:
                return updates[0]
            else:
                update, *other_updates = updates
                if all(
                    equal_computations([update], [other_update]) for other_update in other_updates
                ):
                    return update

                warnings.warn(
                    f"RNG Variable {rng} has multiple distinct clients {rng_clients}, "
                    f"likely due to an inconsistent random graph. "
                    f"No default update will be returned.",
                    UserWarning,
                )
                return None

        [client, _] = rng_clients[0]

        # RNG is an output of the function, this is not a problem
        client_op = client.op

        match client_op:
            case Output():
                return None
            # Otherwise, RNG is used by another operator, which should output an update for the RNG
            case RandomVariable():
                # RandomVariable first output is always the update of the input RNG
                next_rng = client.outputs[0]
            case RNGConsumerOp():
                # RNGConsumerOp have an explicit method that returns an update mapping for their RNG(s)
                # RandomVariable is a subclass of RNGConsumerOp, but we specialize above for speedup
                next_rng = client_op.update(client).get(rng)
                if next_rng is None:
                    raise ValueError(f"No update found for at least one RNG used in {client_op}")
            case Scan():
                # Check if any shared output corresponds to the RNG
                rng_idx = client.inputs.index(rng)
                io_map = client_op.get_oinp_iinp_iout_oout_mappings()["outer_out_from_outer_inp"]
                out_idx = io_map.get(rng_idx, -1)
                if out_idx != -1:
                    next_rng = client.outputs[out_idx]
                else:  # No break
                    raise ValueError(
                        f"No update found for at least one RNG used in Scan Op {client_op}.\n"
                        "You can use `pytensorf.collect_default_updates` inside the Scan function to return updates automatically."
                    )
            case OpFromGraph():
                try:
                    next_rng = collect_default_updates_inner_fgraph(client).get(rng)
                    if next_rng is None:
                        # OFG either does not make use of this RNG or inconsistent use that will have emitted a warning
                        return None
                except ValueError as exc:
                    raise ValueError(
                        f"No update found for at least one RNG used in OpFromGraph Op {client_op}.\n"
                        "You can use `pytensorf.collect_default_updates` and include those updates as outputs."
                    ) from exc
            case _:
                # We don't know how this RNG should be updated. The user should provide an update manually
                return None

        # Recurse until we find final update for RNG
        nested_next_rng = find_default_update(clients, next_rng)
        if nested_next_rng is None:
            # There were no more uses of this next_rng
            return next_rng
        else:
            return nested_next_rng

    if inputs is None:
        inputs = []

    outs = makeiter(outputs)
    fg = FunctionGraph(outputs=outs, clone=False)
    clients = fg.clients

    rng_updates = {}
    # Iterate over input RNGs. Only consider shared RNGs if `must_be_shared==True`
    for input_rng in (
        inp
        for inp in graph_inputs(outs, blockers=inputs)
        if (
            (not must_be_shared or isinstance(inp, SharedVariable))
            and isinstance(inp.type, RandomType)
        )
    ):
        # Even if an explicit default update is provided, we call it to
        # issue any warnings about invalid random graphs.
        default_update = find_default_update(clients, input_rng)

        # Respect default update if provided
        if hasattr(input_rng, "default_update") and input_rng.default_update is not None:
            rng_updates[input_rng] = input_rng.default_update
        else:
            if default_update is not None:
                rng_updates[input_rng] = default_update

    return rng_updates


def compile(
    inputs,
    outputs,
    random_seed: SeedSequenceSeed = None,
    mode=None,
    **kwargs,
) -> Function:
    """Use ``pytensor.function`` with specialized pymc rewrites always enabled.

    This function also ensures shared Generator used by RandomVariables
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
    rng_updates = collect_default_updates(
        inputs=[inp.variable if isinstance(inp, pytensor.In) else inp for inp in inputs],
        outputs=[
            out.variable if isinstance(out, pytensor.Out) else out for out in makeiter(outputs)
        ],
    )

    # We always reseed random variables as this provides RNGs with no chances of collision
    if rng_updates:
        rngs = cast(list[SharedVariable], list(rng_updates))
        reseed_rngs(rngs, random_seed)

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
) -> tuple[np.ndarray | Variable, ...]:
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
    fg = FunctionGraph(outputs=xs, features=[ShapeFeature()], copy_inputs=False, clone=True)

    # The default rewrite_graph includes a constant_folding that is not always applied.
    # We use an unconditional constant_folding as the last pass to ensure a thorough constant folding.
    rewrite_graph(fg)
    topo_unconditional_constant_folding.apply(fg)

    folded_xs = fg.outputs

    if raise_not_constant and not all(isinstance(folded_x, Constant) for folded_x in folded_xs):
        raise NotConstantValueError

    return tuple(
        folded_x.data if isinstance(folded_x, Constant) else folded_x for folded_x in folded_xs
    )


def rewrite_pregrad(graph):
    """Apply simplifying or stabilizing rewrites to graph that are safe to use pre-grad."""
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
    fgraph: FunctionGraph,
    replacements: Sequence[tuple[Variable, Variable]],
    reverse: bool = False,
) -> None:
    """Replace multiple variables in place in topological order."""
    fgraph_toposort = {node: i for i, node in enumerate(fgraph.toposort())}
    fgraph_toposort[None] = -1  # Variables without owner are not in the toposort
    sorted_replacements = sorted(
        replacements,
        key=lambda pair: fgraph_toposort[pair[0].owner],
        reverse=reverse,
    )
    fgraph.replace_all(sorted_replacements, import_missing=True)


def normalize_rng_param(rng: None | Variable) -> Variable:
    """Validate rng is a valid type or create a new one if None."""
    if rng is None:
        rng = pytensor.shared(np.random.default_rng())
    elif not isinstance(rng.type, RandomType):
        raise TypeError(
            "The type of rng should be an instance of either RandomGeneratorType or RandomStateType"
        )
    return rng
