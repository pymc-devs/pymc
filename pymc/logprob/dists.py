import warnings
from copy import copy
from typing import Sequence

import aesara
import aesara.tensor as at
import numpy as np
from aesara.compile.builders import OpFromGraph
from aesara.graph.basic import Apply, Constant
from aesara.graph.op import Op
from aesara.tensor.basic import make_vector
from aesara.tensor.random.utils import broadcast_params, normalize_size_param
from aesara.tensor.var import TensorVariable

from aeppl.abstract import MeasurableVariable, _get_measurable_outputs


class DiracDelta(Op):
    """An `Op` that represents a Dirac-delta distribution."""

    __props__ = ("rtol", "atol")

    def __init__(self, rtol=1e-5, atol=1e-8):
        self.rtol = rtol
        self.atol = atol

    def make_node(self, x):
        x = at.as_tensor(x)
        return Apply(self, [x], [x.type()])

    def do_constant_folding(self, fgraph, node):
        # Without this, the `Op` would be removed from the graph during
        # canonicalization
        return False

    def perform(self, node, inp, out):
        (x,) = inp
        (z,) = out
        warnings.warn(
            "DiracDelta is a dummy Op that shouldn't be used in a compiled graph"
        )
        z[0] = x

    def infer_shape(self, fgraph, node, input_shapes):
        return input_shapes


dirac_delta = DiracDelta()

MeasurableVariable.register(DiracDelta)


def non_constant(x):
    x = at.as_tensor_variable(x)
    if isinstance(x, Constant):
        # XXX: This isn't good for `size` parameters, because it could result
        # in `at.get_vector_length` exceptions.
        res = x.type()
        res.tag = copy(res.tag)
        if aesara.config.compute_test_value != "off":
            res.tag.test_value = x.data
        res.name = x.name
        return res
    else:
        return x


def switching_process(
    comp_rvs: Sequence[TensorVariable],
    states: TensorVariable,
):
    """Construct a switching process over arbitrary univariate mixtures and a state sequence.

    This simply constructs a graph of the following form:

        at.stack(comp_rvs)[states, *idx]

    where ``idx`` makes sure that `states` selects mixture components along all
    the other axes.

    Parameters
    ----------
    comp_rvs
        A list containing `MeasurableVariable` objects for each mixture component.
    states
        The hidden state sequence.  It should have a number of states
        equal to the size of `comp_dists`.

    """

    states = at.as_tensor(states, dtype=np.int64)
    comp_rvs_bcast = at.broadcast_arrays(*[at.as_tensor(rv) for rv in comp_rvs])
    M_rv = at.stack(comp_rvs_bcast)
    indices = (states,) + tuple(at.arange(d) for d in tuple(M_rv.shape)[1:])
    rv_var = M_rv[indices]
    return rv_var


class DiscreteMarkovChainFactory(OpFromGraph):
    """An `Op` constructed from an Aesara graph that represents a discrete Markov chain.

    This "composite" `Op` allows us to mark a sub-graph as measurable and
    assign a `_logprob` dispatch implementation.

    As far as broadcasting is concerned, this `Op` has the following
    `RandomVariable`-like properties:

        ndim_supp = 1
        ndims_params = (3, 1)

    TODO: It would be nice to express this as a `Blockwise` `Op`.
    """

    default_output = 0


MeasurableVariable.register(DiscreteMarkovChainFactory)


@_get_measurable_outputs.register(DiscreteMarkovChainFactory)
def _get_measurable_outputs_DiscreteMarkovChainFactory(op, node):
    return [node.outputs[0]]


def create_discrete_mc_op(srng, size, Gammas, gamma_0):
    """Construct a `DiscreteMarkovChainFactory` `Op`.

    This returns a `Scan` that performs the follow:

        states[0] = categorical(gamma_0)
        for t in range(1, N):
            states[t] = categorical(Gammas[t, state[t-1]])

    The Aesara graph representing the above is wrapped in an `OpFromGraph` so
    that we can easily assign it a specific log-probability.

    TODO: Eventually, AePPL should be capable of parsing more sophisticated
    `Scan`s and producing nearly the same log-likelihoods, and the use of
    `OpFromGraph` will no longer be necessary.

    """

    # Again, we need to preserve the length of this symbolic vector, so we do
    # this.
    size_param = make_vector(
        *[non_constant(size[i]) for i in range(at.get_vector_length(size))]
    )
    size_param.name = "size"

    # We make shallow copies so that unwanted ancestors don't appear in the
    # graph.
    Gammas_param = non_constant(Gammas).clone()
    Gammas_param.name = "Gammas_param"

    gamma_0_param = non_constant(gamma_0).clone()
    gamma_0_param.name = "gamma_0_param"

    bcast_Gammas_param, bcast_gamma_0_param = broadcast_params(
        (Gammas_param, gamma_0_param), (3, 1)
    )

    # Sample state 0 in each state sequence
    state_0 = srng.categorical(
        bcast_gamma_0_param,
        size=tuple(size_param) + tuple(bcast_gamma_0_param.shape[:-1]),
        # size=at.join(0, size_param, bcast_gamma_0_param.shape[:-1]),
    )

    N = bcast_Gammas_param.shape[-3]
    states_shape = tuple(state_0.shape) + (N,)

    bcast_Gammas_param = at.broadcast_to(
        bcast_Gammas_param, states_shape + tuple(bcast_Gammas_param.shape[-2:])
    )

    def loop_fn(n, state_nm1, Gammas_inner):
        gamma_t = Gammas_inner[..., n, :, :]
        idx = tuple(at.ogrid[[slice(None, d) for d in tuple(state_0.shape)]]) + (
            state_nm1.T,
        )
        gamma_t = gamma_t[idx]
        state_n = srng.categorical(gamma_t)
        return state_n.T

    res, updates = aesara.scan(
        loop_fn,
        outputs_info=[{"initial": state_0.T, "taps": [-1]}],
        sequences=[at.arange(N)],
        non_sequences=[bcast_Gammas_param],
        # strict=True,
    )

    update_outputs = [state_0.owner.inputs[0].default_update]
    update_outputs.extend(updates.values())

    return (
        DiscreteMarkovChainFactory(
            [size_param, Gammas_param, gamma_0_param],
            [res.T] + update_outputs,
            inline=True,
            on_unused_input="ignore",
        ),
        updates,
    )


def discrete_markov_chain(
    Gammas: TensorVariable, gamma_0: TensorVariable, size=None, srng=None, **kwargs
):
    """Construct a first-order discrete Markov chain distribution.

    This characterizes vector random variables consisting of state indicator
    values (i.e. ``0`` to ``M - 1``) that are driven by a discrete Markov chain.


    Parameters
    ----------
    Gammas
        An array of transition probability matrices.  `Gammas` takes the
        shape ``... x N x M x M`` for a state sequence of length ``N`` having
        ``M``-many distinct states.  Each row, ``r``, in a transition probability
        matrix gives the probability of transitioning from state ``r`` to each
        other state.
    gamma_0
        The initial state probabilities.  The last dimension should be length ``M``,
        i.e. the number of distinct states.
    """
    gamma_0 = at.as_tensor_variable(gamma_0)

    assert Gammas.ndim >= 3

    Gammas = at.as_tensor_variable(Gammas)

    size = normalize_size_param(size)

    if srng is None:
        srng = at.random.RandomStream()

    dmc_op, updates = create_discrete_mc_op(srng, size, Gammas, gamma_0)
    rv_var = dmc_op(size, Gammas, gamma_0)

    updates = {
        rv_var.owner.inputs[-2]: rv_var.owner.outputs[-2],
        rv_var.owner.inputs[-1]: rv_var.owner.outputs[-1],
    }

    testval = kwargs.pop("testval", None)

    if testval is not None:
        rv_var.tag.test_value = testval

    return rv_var, updates
