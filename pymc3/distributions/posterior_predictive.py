import numbers
import itertools
from typing import List, Dict, Any, Optional, Set, Tuple, Union, cast, TYPE_CHECKING, Callable
import warnings
import logging
from collections import UserDict
import contextvars
from contextlib import AbstractContextManager

import numpy as np
import theano
import theano.tensor as tt

from ..backends.base import MultiTrace #, TraceLike, TraceDict
from ..backends.ndarray import point_list_to_multitrace
from .distribution import _DrawValuesContext, _DrawValuesContextBlocker, is_fast_drawable, _compile_theano_function, vectorized_ppc
from ..model import Model, get_named_nodes_and_relations, ObservedRV, MultiObservedRV, modelcontext
from ..exceptions import IncorrectArgumentsError
# Failing tests:
#    test_mixture_random_shape::test_mixture_random_shape
#

PosteriorPredictiveTrace = Dict[str, np.ndarray]
# TraceIsh = Union[MultiTrace, Dict[str, np.ndarray], List['TraceIsh']]
Point = Dict[str, np.ndarray]

# def _canonicalize_traceish(trace: TraceIsh) -> Union[MultiTrace, Dict[str, np.ndarray]]:
#     '''Transform a traceish thing into either a dictionary of variables to values, or
# leave it alone (if it's a MultiTrace).'''
#     if isinstance(trace, MultiTrace):
#         return trace
#     if isinstance(trace, list) and all((isinstance(x, MultiTrace) for x in trace)):
        

# Simplifications: 
# 1. Don't worry about all the different kinds of things that could be the `trace`
# argument: just accept a MultiTrace.  
# 2. Just lock the number of samples to be the same as the number of samples in the
# MultiTrace.
def sample_posterior_predictive(trace: Union[MultiTrace, List[Dict[str, np.ndarray]]],
                                samples: Optional[int]=None,
                                model: Optional[Model]=None,
                                var_names: Optional[List[str]]=None,
                                keep_size: Optional[bool]=False,
                                random_seed=None) -> Dict[str, np.ndarray]:
    """Generate posterior predictive samples from a model given a trace.

    Parameters
    ----------
    trace : MultiTrace
        Trace generated from MCMC sampling.
    samples : int, optional
        Number of posterior predictive samples to generate. Defaults to one posterior predictive
        sample per posterior sample, that is, the number of draws times the number of chains. It
        is not recommended to modify this value; when modified, some chains may not be represented
        in the posterior predictive sample.
    model : Model (optional if in `with` context)
        Model used to generate `trace`
    var_names : Iterable[str]
        List of vars to sample.
    keep_size : bool, optional
        Force posterior predictive sample to have the same shape as posterior and sample stats
        data: ``(nchains, ndraws, ...)``.
    random_seed : int
        Seed for the random number generator.

    Returns
    -------
    samples : dict
        Dictionary with the variable names as keys, and values numpy arrays containing
        posterior predictive samples.
    """

    model = modelcontext(model)

    if keep_size and samples is not None:
        raise IncorrectArgumentsError("Should not specify both keep_size and samples argukments")

    if isinstance(trace, list) and all((isinstance(x, dict) for x in trace)):
       _trace = point_list_to_multitrace(trace, model)
    elif isinstance(trace, MultiTrace):
        _trace = trace
    else:
        raise TypeError("Unable to generate posterior predictive samples from argument of type %s"%type(trace))

    len_trace = len(_trace)
    try:
        nchain = _trace.nchains
    except AttributeError:
        nchain = 1

    assert isinstance(_trace, MultiTrace)

    _samples = [] # type: List[int]
    # temporary replacement for more complicated logic.
    max_samples: int = len(trace) * nchain
    if samples is None:
        _samples = [max_samples]
    elif samples < max_samples:
        warnings.warn("samples parameter is smaller than nchains times ndraws, some draws "
                      "and/or chains may not be represented in the returned posterior "
                      "predictive sample")
        # if this is less than the number of samples in the trace, take a slice and
        # work with that.  It's too hard for me to deal with uneven-length chains in the
        # MultiTrace, so we just don't let you do that. [2019/08/19:rpg]
        if divmod(samples, len(_trace))[1] != 0:
            raise IncorrectArgumentsError("Number of samples must be a multiple of the number of chains in the trace.")
        _trace = _trace[slice(samples // nchain)]
        _samples = [samples]
    elif samples > max_samples:
        full, rem = divmod(samples, max_samples)
        _samples = (full * [max_samples]) + ([rem] if rem != 0 else [])

    # if keep_size and samples is not None:
    #     raise IncorrectArgumentsError("Should not specify both keep_size and samples arguments")

    # if samples is None:
    #     samples = len(trace) * nchain

    # if samples < len_trace * nchain:
    #     warnings.warn("samples parameter is smaller than nchains times ndraws, some draws "
    #                  "and/or chains may not be represented in the returned posterior "
    #                  "predictive sample")

    if var_names is None:
        vars = model.observed_RVs
    else:
        vars = [model[x] for x in var_names]

    if random_seed is not None:
        np.random.seed(random_seed)

#    indices = np.arange(_samples)

    ppc_trace = _ExtendableTrace()
    for s in _samples:
        try:
            new_trace = posterior_predictive_draw_values(cast(List[Any], vars), _trace, s)
            ppc_trace.extend_trace(new_trace)
        except KeyboardInterrupt:
            pass

    if keep_size:
        return {k: ary.reshape((nchain, len_trace, *ary.shape[1:])) for k, ary in ppc_trace.items() }
    else:
        return ppc_trace.data # this gets us a Dict[str, np.ndarray] instead of my wrapped equiv.

if TYPE_CHECKING:
    _ETPParent = UserDict[str, np.ndarray]  # this is only processed by mypy
else:
    _ETPParent = UserDict  # this is not seen by mypy but will be executed at runtime.

class _ExtendableTrace(_ETPParent):
    def extend_trace(self, trace: Dict[str, np.ndarray]) -> None:
        for k, v in trace.items():
            if k in self.data:
                self.data[k] = np.concatenate((self.data[k], v))
            else:
                self.data[k] = v

def posterior_predictive_draw_values(vars: List[Any], trace: MultiTrace, samples: int) -> PosteriorPredictiveTrace:
    sampler = _PosteriorPredictiveSampler(vars, trace, samples, None)
    sampler.draw_values()
    return sampler.pp_trace
    
class _PosteriorPredictiveSampler(AbstractContextManager):
    '''The process of posterior predictive sampling is quite complicated so this provides a central data store.'''
    pp_trace: PosteriorPredictiveTrace

    # inputs
    vars: List[Any]
    trace: MultiTrace
    samples: int
    size: Optional[int] # not supported!

    # other slots
    logger: logging.Logger

    # for the search
    evaluated: Dict[int, np.ndarray]
    symbolic_params: List[Tuple[int, Any]]

    # set by make_graph...
    leaf_nodes: Dict[str, Any]
    named_nodes_parents: Dict[str, Any]
    named_nodes_children: Dict[str, Any]
    _tok = None                  # type: contextvars.Token
    
    def __init__(self, vars, trace, samples, size=None):
        if size is not None:
            raise NotImplementedError("sample_posterior_predictive does not support the size argument at this time.")
        self.vars = vars
        self.trace = trace
        self.samples = samples
        self.size = size
        self.pp_trace = {}
        self.logger = logging.getLogger('posterior_predictive')

    def __enter__(self):
        self._tok = vectorized_ppc.set(self)
        return self

    def __exit__(self):
        vectorized_ppc.reset(self._tok)
        return False
        
    def draw_values(self) -> None:
        vars = self.vars
        trace = self.trace
        samples = self.samples
        # size = self.size

        with _DrawValuesContext() as context:
            self.init()
            self.make_graph()

            drawn = context.drawn_vars
            # Init givens and the stack of nodes to try to `_draw_value` from
            givens = {p.name: (p, v) for (p, samples), v in drawn.items()
                      if getattr(p, 'name', None) is not None}
            stack = list(self.leaf_nodes.values())  # A queue would be more appropriate
            while stack:
                next_ = stack.pop(0)
                if (next_, samples) in drawn:
                    # If the node already has a givens value, skip it
                    continue
                elif isinstance(next_, (tt.TensorConstant,
                                        tt.sharedvar.SharedVariable)):
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
                    stack.extend([node for node in self.named_nodes_parents[next_]
                                  if isinstance(node, (ObservedRV,
                                                       MultiObservedRV))
                                  and (node, samples) not in drawn])
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
                        value = self.draw_value(next_,
                                                trace=trace,
                                                givens=temp_givens)
                        assert isinstance(value, np.ndarray)
                        self.pp_trace[next_.name] = value
                        givens[next_.name] = (next_, value)
                        drawn[(next_, samples)] = value
                    except theano.gof.fg.MissingInputError:
                        # The node failed, so we must add the node's parents to
                        # the stack of nodes to try to draw from. We exclude the
                        # nodes in the `params` list.
                        stack.extend([node for node in self.named_nodes_parents[next_]
                                      if node is not None and
                                      (node, samples) not in drawn])


            # the below makes sure the graph is evaluated in order
            # test_distributions_random::TestDrawValues::test_draw_order fails without it
            # The remaining params that must be drawn are all hashable
            to_eval = set() # type: Set[int]
            missing_inputs = set([j for j, p in self.symbolic_params]) # type: Set[int]
            while to_eval or missing_inputs:
                if to_eval == missing_inputs:
                    raise ValueError('Cannot resolve inputs for {}'.format([str(trace.varnames[j]) for j in to_eval]))
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
                                    if (
                                        node.name not in givens and
                                        (node, samples) in drawn
                                    ):
                                        givens[node.name] = (
                                            node,
                                            drawn[(node, samples)]
                                        )
                            value = self.draw_value(param,
                                                    trace=self.trace,
                                                    givens=givens.values())
                            assert isinstance(value, np.ndarray)
                            self.pp_trace[param.name] = value
                            self.evaluated[param_idx] = drawn[(param, samples)] = value
                            givens[param.name] = (param, value)
                        except theano.gof.fg.MissingInputError:
                            missing_inputs.add(param_idx)
        assert set(self.pp_trace.keys()) == {var.name for var in vars}

    def init(self) -> None:
        '''This method carries out the initialization phase of sampling 
    from the posterior predictive distribution.  Notably it initializes the
    ``_DrawValuesContext`` bookkeeping object and evaluates the "fast drawable"
    parts of the model.'''
        vars: List[Any] = self.vars
        trace: MultiTrace = self.trace
        samples: int = self.samples

        # initialization phase
        context = _DrawValuesContext.get_context()
        with context:
            drawn = context.drawn_vars
            evaluated = {} # type: Dict[int, Any]
            symbolic_params = []
            for i, var in enumerate(vars):
                if is_fast_drawable(var):
                    evaluated[i] = self.pp_trace[var.name] = self.draw_value(var)
                    continue
                name = getattr(var, 'name', None)
                if (var, samples) in drawn:
                    evaluated[i] = val = drawn[(var, samples)]
                                # We filter out Deterministics by checking for `model` attribute
                elif name is not None and hasattr(var, 'model') and name in trace.varnames:
                    # param.name is in point
                    evaluated[i] = drawn[(var, samples)] = trace[name].reshape(-1)
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
        self.leaf_nodes = {} # type: Dict[str, Any]
        self.named_nodes_parents = {} # type: Dict[str, Any]
        self.named_nodes_children = {} # type: Dict[str, Any]
        for _, param in symbolic_params:
            if hasattr(param, 'name'):
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

    def draw_value(self, param, trace: Optional[MultiTrace]=None, givens = None):
        """Draw a set of random values from a distribution or return a constant.

        Parameters
        ----------
        param : number, array like, theano variable or pymc3 random variable
            The value or distribution. Constants or shared variables
            will be converted to an array and returned. Theano variables
            are evaluated. If `param` is a pymc3 random variable, draw
            values from it and return that (as ``np.ndarray``), unless a 
            value is specified in the ``trace``.
        trace : pm.MultiTrace, optional
            A dictionary from pymc3 variable names to samples of their values
            used to provide context for evaluating ``param``.
        givens : dict, optional
            A dictionary from theano variables to their values. These values
            are used to evaluate ``param`` if it is a theano variable.
        """
        samples = self.samples

        def random_sample(meth: Callable[..., np.ndarray], param, point: MultiTrace, size: int, shape: Tuple[int, ...]) -> np.ndarray:
            val = meth(point=trace, size=size)
            if size == 1:
                val = np.expand_dims(val, axis=0)
            assert val.shape == (size, ) + shape, "Sampling from random of %s yields wrong shape"%param
            return val

        if isinstance(param, (numbers.Number, np.ndarray)):
            return param
        elif isinstance(param, tt.TensorConstant):
            return param.value
        elif isinstance(param, tt.sharedvar.SharedVariable):
            return param.get_value()
        elif isinstance(param, (tt.TensorVariable, MultiObservedRV)):
            if hasattr(param, 'model') and trace and param.name in trace.varnames:
                return trace[param.name]
            elif hasattr(param, 'random') and param.random is not None:
                model = modelcontext(None)                                                                      
                shape = tuple(_param_shape(param, model)) # type: Tuple[int, ...]
                return random_sample(param.random, param, point=trace, size=samples, shape=shape)
            elif (hasattr(param, 'distribution') and
                    hasattr(param.distribution, 'random') and
                    param.distribution.random is not None):
                if hasattr(param, 'observations'):
                    # shape inspection for ObservedRV
                    dist_tmp = param.distribution
                    try:
                        distshape = tuple(param.observations.shape.eval()) # type: Tuple[int, ...]
                    except AttributeError:
                        distshape = tuple(param.observations.shape)

                    dist_tmp.shape = distshape
                    try:
                        return random_sample(dist_tmp.random, param, point=trace, size=samples, shape=distshape)
                    except (ValueError, TypeError):
                        # reset shape to account for shape changes
                        # with theano.shared inputs
                        dist_tmp.shape = ()
                        # We want to draw values to infer the dist_shape,
                        # we don't want to store these drawn values to the context
                        with _DrawValuesContextBlocker():
                            point = next(trace.points()) if trace else None
                            temp_val = np.atleast_1d(dist_tmp.random(point=point,
                                                                    size=None))
                            dist_tmp.shape = tuple(temp_val.shape)
                        return random_sample(dist_tmp.random, point=trace, size=samples, param=param, shape=tuple(dist_tmp.shape))
                else: # has a distribution, but no observations
                    distshape = tuple(param.distribution.shape)
                    return random_sample(meth=param.distribution.random, param=param, point=trace, size=samples, shape=distshape)
            # NOTE: I think the following is already vectorized.
            else: # doesn't have a distribution -- deterministic (?) -- this is probably wrong, but
                # the values here will be a list of *sampled* values -- 1 per sample.
                if givens:
                    variables, values = list(zip(*givens))
                else:
                    variables = values = []
                # We only truly care if the ancestors of param that were given
                # value have the matching dshape and val.shape
                param_ancestors = \
                    set(theano.gof.graph.ancestors([param],
                                                   blockers=list(variables))
                        )
                inputs = [(var, val) for var, val in
                          zip(variables, values)
                          if var in param_ancestors]
                if inputs:
                    input_vars, input_vals = list(zip(*inputs))
                else:
                    input_vars = []
                    input_vals = []
                func = _compile_theano_function(param, input_vars)
                if not input_vars:
                    output = func(*input_vals)
                    if hasattr(output, 'shape', None):
                        val = np.ndarray(output * samples)
                    else:
                        val = np.full(samples, output)

                else:
                    val = np.ndarray([func(*input_vals) for inp in zip(*input_vals)])
                return val
        raise ValueError('Unexpected type in draw_value: %s' % type(param))


def _param_shape(var_desig, model: Model) -> Tuple[int, ...]:
    if isinstance(var_desig, str):
        v = model[var_desig]
    else:
        v = var_desig                                                                          
    if hasattr(v, 'observations'):
        try:
            # To get shape of _observed_ data container `pm.Data`
            # (wrapper for theano.SharedVariable) we evaluate it.
            shape = tuple(v.observations.shape.eval())
        except AttributeError:
            shape = v.observations.shape
    elif hasattr(v, 'dshape'):
        shape = v.dshape
    else:
        shape = v.tag.test_value.shape
    if shape == (1,):
        shape = tuple()
    return shape

# # Posterior predictive sampling takes a "trace-like" argument that is
# # either a `pm.MultiTrace` or a dictionary that acts like a
# # trace. This smooths over that distinction
# def _trace_varnames(trace_like: TraceLike) -> List[str]:
#     if hasattr(trace_like, 'varnames'):
#         trace_like = cast(MultiTrace, trace_like)
#         return trace_like.varnames
#     elif isinstance(trace_like, list):
#         varnames = [] # type: List[str]
#         for tl in trace_like:
#             varnames += _trace_varnames(tl)
#         return varnames
#     else:
#         return list(trace_like.keys())


# class _PointIterator (Iterator[Dict[str, np.ndarray]]):
#     new_dict = None # type: Dict[str, np.ndarray]
#     def __init__(self, trace_dict: Dict[str, np.ndarray]):
#         new_dict = {name : val if len(val.shape) > 1 else val.reshape(val.shape + (1,))
#                     for name, val in trace_dict.items() } # type: Dict[str, np.ndarray]
#     def __iter__(self):
#         return self.iter()
#     def iter(self) --> :
#         i = 0
#         def ifunc():
#             try:
#                 point = {name: trace_dict[name][i,:] for name in self.new_dict.keys()}
#                 yield point
#             except IndexError:
#                 raise StopIteration
#         return ifunc



# # Posterior predictive sampling takes a "trace-like" argument that is
# # either a `pm.MultiTrace` or a dictionary that acts like a
# # trace. This smooths over that distinction
# def _trace_points(trace_like: TraceLike) -> Iterator[Dict[str, Any]]:
#     if isinstance(trace_like, MultiTrace):
#         return trace_like.points()
#     elif isinstance(trace_like, dict):
#         return _PointIterator(trace_like)
#     elif isinstance(trace_like, list):
#         raise ValueError("Cannot make point iterator for a list of traces.")
#     else:
#         raise ValueError("Do not know how to make point iterator for object of type %s"%type(trace_like))
