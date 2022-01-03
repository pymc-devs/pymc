from __future__ import annotations

import contextvars
import logging
import numbers
import warnings

from collections import UserDict
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Callable, Dict, List, cast, overload

import numpy as np
import theano.graph.basic
import theano.graph.fg
import theano.tensor as tt

from arviz import InferenceData
from typing_extensions import Literal, Protocol
from xarray import Dataset

from pymc3.backends.base import MultiTrace
from pymc3.distributions.distribution import (
    _compile_theano_function,
    _DrawValuesContext,
    _DrawValuesContextBlocker,
    is_fast_drawable,
    vectorized_ppc,
)
from pymc3.distributions.shape_utils import to_tuple
from pymc3.exceptions import IncorrectArgumentsError
from pymc3.model import (
    Model,
    MultiObservedRV,
    ObservedRV,
    get_named_nodes_and_relations,
    modelcontext,
)
from pymc3.util import chains_and_samples, dataset_to_point_list, get_var_name
from pymc3.vartypes import theano_constant

# Failing tests:
#    test_mixture_random_shape::test_mixture_random_shape
#

Point = Dict[str, np.ndarray]


class HasName(Protocol):
    name: str


class _TraceDict(UserDict):
    """This class extends the standard trace-based representation
    of traces by adding some helpful attributes used in posterior predictive
    sampling.

    Attributes
    ~~~~~~~~~~
        varnames: list of strings"""

    varnames: list[str]
    _len: int
    data: Point

    def __init__(
        self,
        point_list: list[Point] | None = None,
        multi_trace: MultiTrace | None = None,
        dict_: Point | None = None,
    ):
        """"""
        if multi_trace:
            assert point_list is None and dict_ is None
            self.data = {}
            self._len = sum(len(multi_trace._straces[chain]) for chain in multi_trace.chains)
            self.varnames = multi_trace.varnames
            for vn in multi_trace.varnames:
                self.data[vn] = multi_trace.get_values(vn)
        if point_list is not None:
            assert multi_trace is None and dict_ is None
            self.varnames = varnames = list(point_list[0].keys())
            rep_values = [point_list[0][varname] for varname in varnames]
            # translate the point list.
            self._len = num_points = len(point_list)

            def arr_for(val):
                if np.isscalar(val):
                    return np.ndarray(shape=(num_points,))
                elif isinstance(val, np.ndarray):
                    shp = (num_points,) + val.shape
                    return np.ndarray(shape=shp)
                else:
                    raise TypeError(
                        "Illegal object %s of type %s as value of variable in point list."
                        % (val, type(val))
                    )

            self.data = {name: arr_for(val) for name, val in zip(varnames, rep_values)}
            for i, point in enumerate(point_list):
                for var, value in point.items():
                    self.data[var][i] = value
        if dict_ is not None:
            assert point_list is None and multi_trace is None
            self.data = dict_
            self.varnames = list(dict_.keys())
            self._len = dict_[self.varnames[0]].shape[0]
        assert self.varnames is not None and self._len is not None and self.data is not None

    def __len__(self) -> int:
        return self._len

    def _extract_slice(self, slc: slice) -> _TraceDict:
        sliced_dict: Point = {}

        def apply_slice(arr: np.ndarray) -> np.ndarray:
            if len(arr.shape) == 1:
                return arr[slc]
            else:
                return arr[slc, :]

        for vn, arr in self.data.items():
            sliced_dict[vn] = apply_slice(arr)
        return _TraceDict(dict_=sliced_dict)

    @overload
    def __getitem__(self, item: str | HasName) -> np.ndarray:
        ...

    @overload
    def __getitem__(self, item: slice | int) -> _TraceDict:
        ...

    def __getitem__(self, item):
        if isinstance(item, str):
            return super().__getitem__(item)
        elif isinstance(item, slice):
            return self._extract_slice(item)
        elif isinstance(item, int):
            return _TraceDict(dict_={k: np.atleast_1d(v[item]) for k, v in self.data.items()})
        elif hasattr(item, "name"):
            return super().__getitem__(item.name)
        else:
            raise IndexError("Illegal index %s for _TraceDict" % str(item))


def fast_sample_posterior_predictive(
    trace: MultiTrace | Dataset | InferenceData | list[dict[str, np.ndarray]],
    samples: int | None = None,
    model: Model | None = None,
    var_names: list[str] | None = None,
    keep_size: bool = False,
    random_seed=None,
) -> dict[str, np.ndarray]:
    """Generate posterior predictive samples from a model given a trace.

    This is a vectorized alternative to the standard ``sample_posterior_predictive`` function.
    It aims to be as compatible as possible with the original API, and is significantly
    faster.  Both posterior predictive sampling functions have some remaining issues, and
    we encourage users to verify agreement across the results of both functions for the time
    being.

    Parameters
    ----------
    trace: MultiTrace, xarray.Dataset, InferenceData, or List of points (dictionary)
        Trace generated from MCMC sampling.
    samples: int, optional
        Number of posterior predictive samples to generate. Defaults to one posterior predictive
        sample per posterior sample, that is, the number of draws times the number of chains. It
        is not recommended to modify this value; when modified, some chains may not be represented
        in the posterior predictive sample.
    model: Model (optional if in `with` context)
        Model used to generate `trace`
    var_names: Iterable[str]
        List of vars to sample.
    keep_size: bool, optional
        Force posterior predictive sample to have the same shape as posterior and sample stats
        data: ``(nchains, ndraws, ...)``.
    random_seed: int
        Seed for the random number generator.

    Returns
    -------
    samples: dict
        Dictionary with the variable names as keys, and values numpy arrays containing
        posterior predictive samples.
    """

    ### Implementation note: primarily this function canonicalizes the arguments:
    ### Establishing the model context, wrangling the number of samples,
    ### Canonicalizing the trace argument into a _TraceDict object and fitting it
    ### to the requested number of samples.  Then it invokes posterior_predictive_draw_values
    ### *repeatedly*.  It does this repeatedly, because the trace argument is set up to be
    ### the same as the number of samples. So if the number of samples requested is
    ### greater than the number of samples in the trace parameter, we sample repeatedly.  This
    ### makes the shape issues just a little easier to deal with.

    if isinstance(trace, InferenceData):
        nchains, ndraws = chains_and_samples(trace)
        trace = dataset_to_point_list(trace.posterior)
    elif isinstance(trace, Dataset):
        nchains, ndraws = chains_and_samples(trace)
        trace = dataset_to_point_list(trace)
    elif isinstance(trace, MultiTrace):
        nchains = trace.nchains
        ndraws = len(trace)
    else:
        if keep_size:
            # arguably this should be just a warning.
            raise IncorrectArgumentsError(
                "For keep_size, cannot identify chains and length from %s.", trace
            )

    model = modelcontext(model)
    assert model is not None

    if model.potentials:
        warnings.warn(
            "The effect of Potentials on other parameters is ignored during posterior predictive sampling. "
            "This is likely to lead to invalid or biased predictive samples.",
            UserWarning,
        )

    with model:

        if keep_size and samples is not None:
            raise IncorrectArgumentsError("Should not specify both keep_size and samples arguments")

        if isinstance(trace, list) and all(isinstance(x, dict) for x in trace):
            _trace = _TraceDict(point_list=trace)
        elif isinstance(trace, MultiTrace):
            _trace = _TraceDict(multi_trace=trace)
        else:
            raise TypeError(
                "Unable to generate posterior predictive samples from argument of type %s"
                % type(trace)
            )

        len_trace = len(_trace)

        assert isinstance(_trace, _TraceDict)

        _samples: list[int] = []
        # temporary replacement for more complicated logic.
        max_samples: int = len_trace
        if samples is None or samples == max_samples:
            _samples = [max_samples]
        elif samples < max_samples:
            warnings.warn(
                "samples parameter is smaller than nchains times ndraws, some draws "
                "and/or chains may not be represented in the returned posterior "
                "predictive sample"
            )
            # if this is less than the number of samples in the trace, take a slice and
            # work with that.
            _trace = _trace[slice(samples)]
            _samples = [samples]
        elif samples > max_samples:
            full, rem = divmod(samples, max_samples)
            _samples = (full * [max_samples]) + ([rem] if rem != 0 else [])
        else:
            raise IncorrectArgumentsError(
                "Unexpected combination of samples (%s) and max_samples (%d)"
                % (samples, max_samples)
            )

        if var_names is None:
            vars = model.observed_RVs
        else:
            vars = [model[x] for x in var_names]

        if random_seed is not None:
            np.random.seed(random_seed)

        if TYPE_CHECKING:
            _ETPParent = UserDict[str, np.ndarray]  # this is only processed by mypy
        else:
            # this is not seen by mypy but will be executed at runtime.
            _ETPParent = UserDict

        class _ExtendableTrace(_ETPParent):
            def extend_trace(self, trace: dict[str, np.ndarray]) -> None:
                for k, v in trace.items():
                    if k in self.data:
                        self.data[k] = np.concatenate((self.data[k], v))
                    else:
                        self.data[k] = v

        ppc_trace = _ExtendableTrace()
        for s in _samples:
            strace = _trace if s == len_trace else _trace[slice(0, s)]
            try:
                values = posterior_predictive_draw_values(cast(List[Any], vars), strace, s)
                new_trace: dict[str, np.ndarray] = {k.name: v for (k, v) in zip(vars, values)}
                ppc_trace.extend_trace(new_trace)
            except KeyboardInterrupt:
                pass

    if keep_size:
        return {k: ary.reshape((nchains, ndraws, *ary.shape[1:])) for k, ary in ppc_trace.items()}
    # this gets us a Dict[str, np.ndarray] instead of my wrapped equiv.
    return ppc_trace.data


def posterior_predictive_draw_values(
    vars: list[Any], trace: _TraceDict, samples: int
) -> list[np.ndarray]:
    with _PosteriorPredictiveSampler(vars, trace, samples, None) as sampler:
        return sampler.draw_values()


class _PosteriorPredictiveSampler(AbstractContextManager):
    """The process of posterior predictive sampling is quite complicated so this provides a central data store."""

    # inputs
    vars: list[Any]
    trace: _TraceDict
    samples: int
    size: int | None  # not supported!

    # other slots
    logger: logging.Logger

    # for the search
    evaluated: dict[int, np.ndarray]
    symbolic_params: list[tuple[int, Any]]

    # set by make_graph...
    leaf_nodes: dict[str, Any]
    named_nodes_parents: dict[str, Any]
    named_nodes_children: dict[str, Any]
    _tok: contextvars.Token

    def __init__(self, vars, trace: _TraceDict, samples, model: Model | None, size=None):
        if size is not None:
            raise NotImplementedError(
                "sample_posterior_predictive does not support the size argument at this time."
            )
        assert vars is not None
        self.vars = vars
        self.trace = trace
        self.samples = samples
        self.size = size
        self.logger = logging.getLogger("posterior_predictive")

    def __enter__(self) -> _PosteriorPredictiveSampler:
        self._tok = vectorized_ppc.set(posterior_predictive_draw_values)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        vectorized_ppc.reset(self._tok)
        return False

    def draw_values(self) -> list[np.ndarray]:
        vars = self.vars
        trace = self.trace
        samples = self.samples
        # size = self.size
        params = dict(enumerate(vars))

        with _DrawValuesContext() as context:
            self.init()
            self.make_graph()

            drawn = context.drawn_vars

            # Init givens and the stack of nodes to try to `_draw_value` from
            givens = {
                p.name: (p, v)
                for (p, samples), v in drawn.items()
                if getattr(p, "name", None) is not None
            }
            stack = list(self.leaf_nodes.values())  # A queue would be more appropriate

            while stack:
                next_ = stack.pop(0)
                if (next_, samples) in drawn:
                    # If the node already has a givens value, skip it
                    continue
                elif isinstance(next_, (theano_constant, tt.sharedvar.SharedVariable)):
                    # If the node is a theano.tensor.TensorConstant or a
                    # theano.tensor.sharedvar.SharedVariable, its value will be
                    # available automatically in _compile_theano_function so
                    # we can skip it. Furthermore, if this node was treated as a
                    # TensorVariable that should be compiled by theano in
                    # _compile_theano_function, it would raise a `TypeError:
                    # ('Constants not allowed in param list', ...)` for
                    # TensorConstant, and a `TypeError: Cannot use a shared
                    # variable (...) as explicit input` for SharedVariable.
                    # ObservedRV and MultiObservedRV instances are ViewOPs
                    # of TensorConstants or SharedVariables, we must add them
                    # to the stack or risk evaluating deterministics with the
                    # wrong values (issue #3354)
                    stack.extend(
                        [
                            node
                            for node in self.named_nodes_parents[next_]
                            if isinstance(node, (ObservedRV, MultiObservedRV))
                            and (node, samples) not in drawn
                        ]
                    )
                    continue
                else:
                    # If the node does not have a givens value, try to draw it.
                    # The named node's children givens values must also be taken
                    # into account.
                    children = self.named_nodes_children[next_]
                    temp_givens = [givens[k] for k in givens if k in children]
                    try:
                        # This may fail for autotransformed RVs, which don't
                        # have the random method
                        value = self.draw_value(next_, trace=trace, givens=temp_givens)
                        assert isinstance(value, np.ndarray)
                        givens[next_.name] = (next_, value)
                        drawn[(next_, samples)] = value
                    except theano.graph.fg.MissingInputError:
                        # The node failed, so we must add the node's parents to
                        # the stack of nodes to try to draw from. We exclude the
                        # nodes in the `params` list.
                        stack.extend(
                            [
                                node
                                for node in self.named_nodes_parents[next_]
                                if node is not None and (node, samples) not in drawn
                            ]
                        )

            # the below makes sure the graph is evaluated in order
            # test_distributions_random::TestDrawValues::test_draw_order fails without it
            # The remaining params that must be drawn are all hashable
            to_eval: set[int] = set()
            missing_inputs: set[int] = {j for j, p in self.symbolic_params}

            while to_eval or missing_inputs:
                if to_eval == missing_inputs:
                    raise ValueError(
                        "Cannot resolve inputs for {}".format(
                            [get_var_name(trace.varnames[j]) for j in to_eval]
                        )
                    )
                to_eval = set(missing_inputs)
                missing_inputs = set()
                for param_idx in to_eval:
                    param = vars[param_idx]
                    drawn = context.drawn_vars
                    if (param, samples) in drawn:
                        self.evaluated[param_idx] = drawn[(param, samples)]
                    else:
                        try:
                            if param in self.named_nodes_children:
                                for node in self.named_nodes_children[param]:
                                    if node.name not in givens and (node, samples) in drawn:
                                        givens[node.name] = (
                                            node,
                                            drawn[(node, samples)],
                                        )
                            value = self.draw_value(param, trace=self.trace, givens=givens.values())
                            assert isinstance(value, np.ndarray)
                            self.evaluated[param_idx] = drawn[(param, samples)] = value
                            givens[param.name] = (param, value)
                        except theano.graph.fg.MissingInputError:
                            missing_inputs.add(param_idx)
        return [self.evaluated[j] for j in params]

    def init(self) -> None:
        """This method carries out the initialization phase of sampling
        from the posterior predictive distribution.  Notably it initializes the
        ``_DrawValuesContext`` bookkeeping object and evaluates the "fast drawable"
        parts of the model."""
        vars: list[Any] = self.vars
        trace: _TraceDict = self.trace
        samples: int = self.samples
        leaf_nodes: dict[str, Any]
        named_nodes_parents: dict[str, Any]
        named_nodes_children: dict[str, Any]

        # initialization phase
        context = _DrawValuesContext.get_context()
        assert isinstance(context, _DrawValuesContext)
        with context:
            drawn = context.drawn_vars
            evaluated: dict[int, Any] = {}
            symbolic_params = []
            for i, var in enumerate(vars):
                if is_fast_drawable(var):
                    evaluated[i] = self.draw_value(var)
                    continue
                name = getattr(var, "name", None)
                if (var, samples) in drawn:
                    evaluated[i] = drawn[(var, samples)]
                    # We filter out Deterministics by checking for `model` attribute
                elif name is not None and hasattr(var, "model") and name in trace.varnames:
                    # param.name is in the trace.  Record it as drawn and evaluated
                    drawn[(var, samples)] = evaluated[i] = trace[cast(str, name)]
                else:
                    # param still needs to be drawn
                    symbolic_params.append((i, var))
        self.evaluated = evaluated
        self.symbolic_params = symbolic_params

    def make_graph(self) -> None:
        # Distribution parameters may be nodes which have named node-inputs
        # specified in the point. Need to find the node-inputs, their
        # parents and children to replace them.
        symbolic_params = self.symbolic_params
        self.leaf_nodes = {}
        self.named_nodes_parents = {}
        self.named_nodes_children = {}
        for _, param in symbolic_params:
            if hasattr(param, "name"):
                # Get the named nodes under the `param` node
                nn, nnp, nnc = get_named_nodes_and_relations(param)
                self.leaf_nodes.update(nn)
                # Update the discovered parental relationships
                for k in nnp.keys():
                    if k not in self.named_nodes_parents.keys():
                        self.named_nodes_parents[k] = nnp[k]
                    else:
                        self.named_nodes_parents[k].update(nnp[k])
                # Update the discovered child relationships
                for k in nnc.keys():
                    if k not in self.named_nodes_children.keys():
                        self.named_nodes_children[k] = nnc[k]
                    else:
                        self.named_nodes_children[k].update(nnc[k])

    def draw_value(self, param, trace: _TraceDict | None = None, givens=None):
        """Draw a set of random values from a distribution or return a constant.

        Parameters
        ----------
        param: number, array like, theano variable or pymc3 random variable
            The value or distribution. Constants or shared variables
            will be converted to an array and returned. Theano variables
            are evaluated. If `param` is a pymc3 random variable, draw
            values from it and return that (as ``np.ndarray``), unless a
            value is specified in the ``trace``.
        trace: pm.MultiTrace, optional
            A dictionary from pymc3 variable names to samples of their values
            used to provide context for evaluating ``param``.
        givens: dict, optional
            A dictionary from theano variables to their values. These values
            are used to evaluate ``param`` if it is a theano variable.
        """
        samples = self.samples

        def random_sample(
            meth: Callable[..., np.ndarray],
            param,
            point: _TraceDict,
            size: int,
            shape: tuple[int, ...],
        ) -> np.ndarray:
            val = meth(point=point, size=size)
            try:
                assert val.shape == to_tuple(size) + to_tuple(shape), (
                    "Sampling from random of %s yields wrong shape" % param
                )
            # error-quashing here is *extremely* ugly, but it seems to be what the logic in DensityDist wants.
            except AssertionError as e:
                if (
                    hasattr(param, "distribution")
                    and hasattr(param.distribution, "wrap_random_with_dist_shape")
                    and not param.distribution.wrap_random_with_dist_shape
                ):
                    pass
                else:
                    raise e

            return val

        if isinstance(param, (numbers.Number, np.ndarray)):
            return param
        elif isinstance(param, theano_constant):
            return param.value
        elif isinstance(param, tt.sharedvar.SharedVariable):
            return param.get_value()
        elif isinstance(param, (tt.TensorVariable, MultiObservedRV)):
            if hasattr(param, "model") and trace and param.name in trace.varnames:
                return trace[param.name]
            elif hasattr(param, "random") and param.random is not None:
                model = modelcontext(None)
                assert isinstance(model, Model)
                shape: tuple[int, ...] = tuple(_param_shape(param, model))
                return random_sample(param.random, param, point=trace, size=samples, shape=shape)
            elif (
                hasattr(param, "distribution")
                and hasattr(param.distribution, "random")
                and param.distribution.random is not None
            ):
                if hasattr(param, "observations"):
                    # shape inspection for ObservedRV
                    dist_tmp = param.distribution
                    try:
                        distshape: tuple[int, ...] = tuple(param.observations.shape.eval())
                    except AttributeError:
                        distshape = tuple(param.observations.shape)

                    dist_tmp.shape = distshape
                    try:
                        return random_sample(
                            dist_tmp.random,
                            param,
                            point=trace,
                            size=samples,
                            shape=distshape,
                        )
                    except (ValueError, TypeError):
                        # reset shape to account for shape changes
                        # with theano.shared inputs
                        dist_tmp.shape = ()
                        # We want to draw values to infer the dist_shape,
                        # we don't want to store these drawn values to the context
                        with _DrawValuesContextBlocker():
                            point = trace[0] if trace else None
                            temp_val = np.atleast_1d(dist_tmp.random(point=point, size=None))
                        # if hasattr(param, 'name') and param.name == 'obs':
                        #     import pdb; pdb.set_trace()
                        # Sometimes point may change the size of val but not the
                        # distribution's shape
                        if point and samples is not None:
                            temp_size = np.atleast_1d(samples)
                            if all(temp_val.shape[: len(temp_size)] == temp_size):
                                dist_tmp.shape = tuple(temp_val.shape[len(temp_size) :])
                            else:
                                dist_tmp.shape = tuple(temp_val.shape)
                        # I am not sure why I need to do this, but I do in order to trim off a
                        # degenerate dimension [2019/09/05:rpg]
                        if dist_tmp.shape[0] == 1 and len(dist_tmp.shape) > 1:
                            dist_tmp.shape = dist_tmp.shape[1:]
                        return random_sample(
                            dist_tmp.random,
                            point=trace,
                            size=samples,
                            param=param,
                            shape=tuple(dist_tmp.shape),
                        )
                else:  # has a distribution, but no observations
                    distshape = tuple(param.distribution.shape)
                    return random_sample(
                        meth=param.distribution.random,
                        param=param,
                        point=trace,
                        size=samples,
                        shape=distshape,
                    )
            # NOTE: I think the following is already vectorized.
            else:
                if givens:
                    variables, values = list(zip(*givens))
                else:
                    variables = values = []
                # We only truly care if the ancestors of param that were given
                # value have the matching dshape and val.shape
                param_ancestors = set(
                    theano.graph.basic.ancestors([param], blockers=list(variables))
                )
                inputs = [
                    (var, val) for var, val in zip(variables, values) if var in param_ancestors
                ]
                if inputs:
                    input_vars, input_vals = list(zip(*inputs))
                else:
                    input_vars = []
                    input_vals = []
                func = _compile_theano_function(param, input_vars)
                if not input_vars:
                    assert input_vals == []  # AFAICT if there are now vars, there can't be vals
                    output = func(*input_vals)
                    if hasattr(output, "shape"):
                        val = np.repeat(np.expand_dims(output, 0), samples, axis=0)
                    else:
                        val = np.full(samples, output)

                else:
                    val = func(*input_vals)
                    # np.ndarray([func(*input_vals) for inp in zip(*input_vals)])
                return val
        raise ValueError("Unexpected type in draw_value: %s" % type(param))


def _param_shape(var_desig, model: Model) -> tuple[int, ...]:
    if isinstance(var_desig, str):
        v = model[var_desig]
    else:
        v = var_desig
    if hasattr(v, "observations"):
        try:
            # To get shape of _observed_ data container `pm.Data`
            # (wrapper for theano.SharedVariable) we evaluate it.
            shape = tuple(v.observations.shape.eval())
        except AttributeError:
            shape = v.observations.shape
    elif hasattr(v, "dshape"):
        shape = v.dshape
    else:
        shape = v.tag.test_value.shape
    if shape == (1,):
        shape = tuple()
    return shape
