R"""
Variational inference is a great approach for doing really complex,
often intractable Bayesian inference in approximate form. Common methods
(e.g. ADVI) lack from complexity so that approximate posterior does not
reveal the true nature of underlying problem. In some applications it can
yield unreliable decisions.

Recently on NIPS 2017 `OPVI  <https://arxiv.org/abs/1610.09033/>`_ framework
was presented. It generalizes variational inverence so that the problem is
build with blocks. The first and essential block is Model itself. Second is
Approximation, in some cases :math:`log Q(D)` is not really needed. Necessity
depends on the third and forth part of that black box, Operator and
Test Function respectively.

Operator is like an approach we use, it constructs loss from given Model,
Approximation and Test Function. The last one is not needed if we minimize
KL Divergence from Q to posterior. As a drawback we need to compute :math:`loq Q(D)`.
Sometimes approximation family is intractable and :math:`loq Q(D)` is not available,
here comes LS(Langevin Stein) Operator with a set of test functions.

Test Function has more unintuitive meaning. It is usually used with LS operator
and represents all we want from our approximate distribution. For any given vector
based function of :math:`z` LS operator yields zero mean function under posterior.
:math:`loq Q(D)` is no more needed. That opens a door to rich approximation
families as neural networks.

References
----------
-   Rajesh Ranganath, Jaan Altosaar, Dustin Tran, David M. Blei
    Operator Variational Inference
    https://arxiv.org/abs/1610.09033 (2016)
"""

import warnings
import itertools
import numpy as np
import theano
import theano.tensor as tt
import pymc3 as pm
from .updates import adagrad_window
from ..distributions.dist_math import rho2sd, log_normal
from ..model import modelcontext
from ..blocking import (
    ArrayOrdering, DictToArrayBijection, VarMap
)
from ..util import get_default_varnames
from ..theanof import tt_rng, memoize, change_flags, identity


__all__ = [
    'ObjectiveFunction',
    'Operator',
    'TestFunction',
    'Approximation'
]


def node_property(f):
    """
    A shortcut for wrapping method to accessible tensor
    """
    return property(memoize(change_flags(compute_test_value='off')(f)))


@change_flags(compute_test_value='raise')
def try_to_set_test_value(node_in, node_out, s):
    _s = s
    if s is None:
        s = 1
    s = theano.compile.view_op(tt.as_tensor(s))
    if not isinstance(node_in, (list, tuple)):
        node_in = [node_in]
    if not isinstance(node_out, (list, tuple)):
        node_out = [node_out]
    for i, o in zip(node_in, node_out):
        if hasattr(i.tag, 'test_value'):
            if not hasattr(s.tag, 'test_value'):
                continue
            else:
                tv = i.tag.test_value[None, ...]
                tv = np.repeat(tv, s.tag.test_value, 0)
                if _s is None:
                    tv = tv[0]
                o.tag.test_value = tv


def get_transformed(z):
    if hasattr(z, 'transformed'):
        z = z.transformed
    return z


class ObjectiveUpdates(theano.OrderedUpdates):
    """
    OrderedUpdates extension for storing loss
    """
    loss = None


def _warn_not_used(smth, where):
    warnings.warn('`%s` is not used for %s and ignored' % (smth, where))


class ObjectiveFunction(object):
    """Helper class for construction loss and updates for variational inference

    Parameters
    ----------
    op : :class:`Operator`
        OPVI Functional operator
    tf : :class:`TestFunction`
        OPVI TestFunction
    """

    def __init__(self, op, tf):
        self.op = op
        self.tf = tf

    obj_params = property(lambda self: self.op.approx.params)
    test_params = property(lambda self: self.tf.params)
    approx = property(lambda self: self.op.approx)

    def updates(self, obj_n_mc=None, tf_n_mc=None, obj_optimizer=adagrad_window, test_optimizer=adagrad_window,
                more_obj_params=None, more_tf_params=None, more_updates=None,
                more_replacements=None, total_grad_norm_constraint=None):
        """Calculates gradients for objective function, test function and then
        constructs updates for optimization step

        Parameters
        ----------
        obj_n_mc : `int`
            Number of monte carlo samples used for approximation of objective gradients
        tf_n_mc : `int`
            Number of monte carlo samples used for approximation of test function gradients
        obj_optimizer : function (loss, params) -> updates
            Optimizer that is used for objective params
        test_optimizer : function (loss, params) -> updates
            Optimizer that is used for test function params
        more_obj_params : `list`
            Add custom params for objective optimizer
        more_tf_params : `list`
            Add custom params for test function optimizer
        more_updates : `dict`
            Add custom updates to resulting updates
        more_replacements : `dict`
            Apply custom replacements before calculating gradients
        total_grad_norm_constraint : `float`
            Bounds gradient norm, prevents exploding gradient problem

        Returns
        -------
        :class:`ObjectiveUpdates`
        """
        if more_updates is None:
            more_updates = dict()
        resulting_updates = ObjectiveUpdates()
        if self.test_params:
            self.add_test_updates(
                resulting_updates,
                tf_n_mc=tf_n_mc,
                test_optimizer=test_optimizer,
                more_tf_params=more_tf_params,
                more_replacements=more_replacements,
                total_grad_norm_constraint=total_grad_norm_constraint
            )
        else:
            if tf_n_mc is not None:
                _warn_not_used('tf_n_mc', self.op)
            if more_tf_params:
                _warn_not_used('more_tf_params', self.op)
        self.add_obj_updates(
            resulting_updates,
            obj_n_mc=obj_n_mc,
            obj_optimizer=obj_optimizer,
            more_obj_params=more_obj_params,
            more_replacements=more_replacements,
            total_grad_norm_constraint=total_grad_norm_constraint
        )
        resulting_updates.update(more_updates)
        return resulting_updates

    def add_test_updates(self, updates, tf_n_mc=None, test_optimizer=adagrad_window,
                         more_tf_params=None, more_replacements=None,
                         total_grad_norm_constraint=None):
        if more_tf_params is None:
            more_tf_params = []
        if more_replacements is None:
            more_replacements = dict()
        tf_target = self(tf_n_mc, more_tf_params=more_tf_params)
        tf_target = theano.clone(tf_target, more_replacements, strict=False)
        grads = pm.updates.get_or_compute_grads(tf_target, self.obj_params + more_tf_params)
        if total_grad_norm_constraint is not None:
            grads = pm.total_norm_constraint(grads, total_grad_norm_constraint)
        updates.update(
            test_optimizer(
                grads,
                self.test_params +
                more_tf_params))

    def add_obj_updates(self, updates, obj_n_mc=None, obj_optimizer=adagrad_window,
                        more_obj_params=None, more_replacements=None,
                        total_grad_norm_constraint=None):
        if more_obj_params is None:
            more_obj_params = []
        if more_replacements is None:
            more_replacements = dict()
        obj_target = self(obj_n_mc, more_obj_params=more_obj_params)
        obj_target = theano.clone(obj_target, more_replacements, strict=False)
        grads = pm.updates.get_or_compute_grads(obj_target, self.obj_params + more_obj_params)
        if total_grad_norm_constraint is not None:
            grads = pm.total_norm_constraint(grads, total_grad_norm_constraint)
        updates.update(
            obj_optimizer(
                grads,
                self.obj_params +
                more_obj_params))
        if self.op.RETURNS_LOSS:
            updates.loss = obj_target

    @memoize
    @change_flags(compute_test_value='off')
    def step_function(self, obj_n_mc=None, tf_n_mc=None,
                      obj_optimizer=adagrad_window, test_optimizer=adagrad_window,
                      more_obj_params=None, more_tf_params=None,
                      more_updates=None, more_replacements=None,
                      total_grad_norm_constraint=None,
                      score=False, fn_kwargs=None):
        R"""Step function that should be called on each optimization step.

        Generally it solves the following problem:

        .. math::

                \mathbf{\lambda^{*}} = \inf_{\lambda} \sup_{\theta} t(\mathbb{E}_{\lambda}[(O^{p,q}f_{\theta})(z)])

        Parameters
        ----------
        obj_n_mc : `int`
            Number of monte carlo samples used for approximation of objective gradients
        tf_n_mc : `int`
            Number of monte carlo samples used for approximation of test function gradients
        obj_optimizer : function (loss, params) -> updates
            Optimizer that is used for objective params
        test_optimizer : function (loss, params) -> updates
            Optimizer that is used for test function params
        more_obj_params : `list`
            Add custom params for objective optimizer
        more_tf_params : `list`
            Add custom params for test function optimizer
        more_updates : `dict`
            Add custom updates to resulting updates
        total_grad_norm_constraint : `float`
            Bounds gradient norm, prevents exploding gradient problem
        score : `bool`
            calculate loss on each step? Defaults to False for speed
        fn_kwargs : `dict`
            Add kwargs to theano.function (e.g. `{'profile': True}`)
        more_replacements : `dict`
            Apply custom replacements before calculating gradients

        Returns
        -------
        `theano.function`
        """
        if fn_kwargs is None:
            fn_kwargs = {}
        if score and not self.op.RETURNS_LOSS:
            raise NotImplementedError('%s does not have loss' % self.op)
        updates = self.updates(obj_n_mc=obj_n_mc, tf_n_mc=tf_n_mc,
                               obj_optimizer=obj_optimizer,
                               test_optimizer=test_optimizer,
                               more_obj_params=more_obj_params,
                               more_tf_params=more_tf_params,
                               more_updates=more_updates,
                               more_replacements=more_replacements,
                               total_grad_norm_constraint=total_grad_norm_constraint)
        if score:
            step_fn = theano.function(
                [], updates.loss, updates=updates, **fn_kwargs)
        else:
            step_fn = theano.function([], None, updates=updates, **fn_kwargs)
        return step_fn

    @memoize
    @change_flags(compute_test_value='off')
    def score_function(self, sc_n_mc=None, more_replacements=None, fn_kwargs=None):   # pragma: no cover
        R"""Compiles scoring function that operates which takes no inputs and returns Loss

        Parameters
        ----------
        sc_n_mc : `int`
            number of scoring MC samples
        more_replacements:
            Apply custom replacements before compiling a function
        fn_kwargs: `dict`
            arbitrary kwargs passed to theano.function

        Returns
        -------
        theano.function
        """
        if fn_kwargs is None:
            fn_kwargs = {}
        if not self.op.RETURNS_LOSS:
            raise NotImplementedError('%s does not have loss' % self.op)
        if more_replacements is None:
            more_replacements = {}
        loss = theano.clone(
            self(sc_n_mc),
            more_replacements,
            strict=False)
        return theano.function([], loss, **fn_kwargs)

    def __getstate__(self):
        return self.op, self.tf

    def __setstate__(self, state):
        self.__init__(*state)

    @change_flags(compute_test_value='off')
    def __call__(self, nmc, **kwargs):
        if 'more_tf_params' in kwargs:
            m = -1.
        else:
            m = 1.
        a = self.op.apply(self.tf)
        a = self.approx.set_size_and_deterministic(a, nmc, 0)
        return m * self.op.T(a)


class Operator(object):
    R"""Base class for Operator

    Parameters
    ----------
    approx : :class:`Approximation`
        an approximation instance

    Notes
    -----
    For implementing Custom operator it is needed to define :func:`Operator.apply` method
    """

    HAS_TEST_FUNCTION = False
    RETURNS_LOSS = True
    SUPPORT_AEVB = True
    OBJECTIVE = ObjectiveFunction
    T = identity

    def __init__(self, approx):
        if not self.SUPPORT_AEVB and approx.local_vars:
            raise ValueError('%s does not support AEVB, '
                             'please change inference method' % type(self))
        self.model = approx.model
        self.approx = approx

    flat_view = property(lambda self: self.approx.flat_view)
    input = property(lambda self: self.approx.flat_view.input)

    logp = property(lambda self: self.approx.logp)
    logq = property(lambda self: self.approx.logq)
    logp_norm = property(lambda self: self.approx.logp_norm)
    logq_norm = property(lambda self: self.approx.logq_norm)

    def apply(self, f):   # pragma: no cover
        R"""Operator itself

        .. math::

            (O^{p,q}f_{\theta})(z)

        Parameters
        ----------
        f : :class:`TestFunction` or None
            function that takes `z = self.input` and returns
            same dimensional output

        Returns
        -------
        `TensorVariable`
            symbolically applied operator
        """
        raise NotImplementedError

    def __call__(self, f=None):
        if self.HAS_TEST_FUNCTION:
            if f is None:
                raise ValueError('Operator %s requires TestFunction' % self)
            else:
                if not isinstance(f, TestFunction):
                    f = TestFunction.from_function(f)
        else:
            if f is not None:
                warnings.warn(
                    'TestFunction for %s is redundant and removed' %
                    self)
            else:
                pass
            f = TestFunction()
        f.setup(self.approx.total_size)
        return self.OBJECTIVE(self, f)

    def __getstate__(self):
        # pickle only important parts
        return self.approx

    def __setstate__(self, approx):
        self.__init__(approx)

    def __str__(self):    # pragma: no cover
        return '%(op)s[%(ap)s]' % dict(op=self.__class__.__name__,
                                       ap=self.approx.__class__.__name__)


def collect_shared_to_list(params):
    """Helper function for getting a list from
    usable representation of parameters

    Parameters
    ----------
    params : {dict|None}

    Returns
    -------
    list
    """
    if isinstance(params, dict):
        return list(
            t[1] for t in sorted(params.items(), key=lambda t: t[0])
            if isinstance(t[1], theano.compile.SharedVariable)
        )
    elif params is None:
        return []
    else:
        raise TypeError(
            'Unknown type %s for %r, need dict or None')


class TestFunction(object):
    def __init__(self):
        self._inited = False
        self.shared_params = None

    def create_shared_params(self, dim):
        """Returns
        -------
        {dict|list|theano.shared}
        """
        pass

    @property
    def params(self):
        return collect_shared_to_list(self.shared_params)

    def __call__(self, z):
        raise NotImplementedError

    def setup(self, dim):
        if not self._inited:
            self._setup(dim)
            self.shared_params = self.create_shared_params(dim)
            self._inited = True

    def _setup(self, dim):
        R"""Does some preparation stuff before calling :func:`Approximation.create_shared_params`

        Parameters
        ----------
        dim : int
            dimension of posterior distribution
        """
        pass

    @classmethod
    def from_function(cls, f):
        if not callable(f):
            raise ValueError('Need callable, got %r' % f)
        obj = TestFunction()
        obj.__call__ = f
        return obj


class GroupApproximation(object):
    """
    Grouped Approximation that is used for modelling mutual dependencies
    for a specified group of variables.

    Parameters
    ----------
    group : list or -1
        represents grouped variables for approximation
    shapes : dict
        maps variable to it's shape if it is in local variable set
        with None standing for flexible dimension
    params : dict
        custom params with valid shape they exactly the
        params that are stored in shared_params dict.
    size : scalar
        needed when group has local variables
    model : Model
    """
    SUPPORT_AEVB = True
    shared_params = None
    initial_dist_name = 'normal'
    initial_dist_map = 0.

    def __init__(self, group=None,
                 shapes=None,
                 params=None,
                 size=None,
                 random_seed=None,
                 model=None):
        self._rng = tt_rng(random_seed)
        model = modelcontext(model)
        self.model = model
        if group is None:
            self.group = model.vars
        elif group == -1:
            self.group = -1
        else:
            self.group = group
        if shapes is None:
            shapes = dict()
        if params is None:
            params = dict()
        self.shapes = shapes
        self.user_params = params
        self._mid_size = size
        self.vmap = dict()
        self.ndim = 0
        self._global_ = True
        self._freeze_ = False
        if self.group != -1:
            # init can be delayed
            self.__init_group__(self.group)

    @property
    def params_dict(self):
        if self.user_params is not None:
            return self.user_params
        else:
            return self.shared_params

    @property
    def global_(self):
        return self._global_

    @global_.setter
    def global_(self, v):
        if self._freeze_:
            raise TypeError('Cannot set state as this property is unchangeable')
        self._global_ = v

    @property
    def local_(self):
        return not self.global_

    @local_.setter
    def local_(self, v):
        self.global_ = not v

    @change_flags(compute_test_value='off')
    def __init_group__(self, group):
        if group is None:
            raise ValueError('Got empty group')
        if self.group == -1:
            # delayed init
            self.group = group
        seen_local = False
        seen_global = False
        # unwrap transformed for replacements

        for var in group:
            var = get_transformed(var)
            shape = self.shapes.get(var, var.dshape)
            check = sum(s is None for s in shape)
            if check > 1:
                raise ValueError('More than one flexible dim is not supported')
            if None in shape:
                vshape = tuple(s if s is not None else -1 for s in shape)
                seen_local = True
            else:
                vshape = shape
                seen_global = True
            if seen_global and seen_local:
                raise TypeError('Group can consist only with either '
                                'local or global variables, but got both')
            s_ = np.prod([s for s in shape if s is not None])
            begin = self.ndim
            self.ndim += s_
            end = self.ndim
            self.vmap[var] = VarMap(var.name, slice(begin, end), vshape, var.dtype)
        self.global_ = seen_global
        self._freeze_ = True
        # last dimension always stands for latent ndim
        # first dimension always stands for sample size

        if self.global_:
            self.symbolic_initial_ = tt.matrix(self.__class__.__name__ + '_symbolic_initial_matrix')
            self.input = tt.vector(self.__class__.__name__ + '_symbolic_input')
        else:
            if self._mid_size is None:
                raise TypeError('Got local variables without local size')
            if self.user_params is None:
                raise TypeError('Got local variables without parametrization')
            self.symbolic_initial_ = tt.tensor3(self.__class__.__name__ + '_symbolic_initial_tensor')
            self.input = tt.matrix(self.__class__.__name__ + '_symbolic_input')
        self.replacements = dict()
        for v, (name, slc, shape, dtype) in self.vmap.items():
            # slice is taken only by last dimension
            vr = self.input[..., slc].reshape(shape).astype(dtype)
            vr.name = name + '_vi_replacement'
            self.replacements[v] = vr

    @property
    def params(self):
        return collect_shared_to_list(self.params_dict)

    def to_flat_input(self, node):
        """
        Replaces vars with flattened view stored in self.input
        """
        return theano.clone(node, self.replacements, strict=False)

    def _new_initial_(self, size, deterministic):
        if size is None:
            size = 1
        if not isinstance(deterministic, tt.Variable):
            deterministic = np.int8(deterministic)
        ndim, dist_name, dist_map = (
            self.ndim,
            self.initial_dist_name,
            self.initial_dist_map
        )
        dtype = self.symbolic_initial_.dtype
        ndim = tt.as_tensor(ndim)
        size = tt.as_tensor(size)
        if self.local_:
            shape = tt.stack((size, self._mid_size, ndim))
        else:
            shape = tt.stack((size, ndim))
        # apply optimizations if possible
        if not isinstance(deterministic, tt.Variable):
            if deterministic:
                return tt.ones(shape, dtype) * dist_map
            else:
                return getattr(self._rng, dist_name)(shape)
        else:
            sample = getattr(self._rng, dist_name)(shape)
            initial = tt.switch(
                deterministic,
                tt.ones(shape, dtype) * dist_map,
                sample
            )
            return initial

    @node_property
    def symbolic_random(self):
        raise NotImplementedError

    @change_flags(compute_test_value='off')
    def set_size_and_deterministic(self, node, s, d):
        initial_ = self._new_initial_(s, d)
        # optimizations
        out = theano.clone(node, {
            self.symbolic_initial_: initial_,
        })
        try_to_set_test_value(node, out, None)
        return out

    @node_property
    def symbolic_normalizing_constant(self):
        """
        Constant to divide when we want to scale down loss from minibatches
        """
        t = self.to_flat_input(
            tt.max([v.scaling for v in self.group]))
        t = theano.clone(t, {
            self.input: self.symbolic_random[0]
        })
        t = self.set_size_and_deterministic(t, 1, 1)  # remove random, we do not it here at all
        return pm.floatX(t)

    @node_property
    def symbolic_logq(self):
        raise NotImplementedError  # shape (s,)

    @node_property
    def logq(self):
        return self.symbolic_logq.mean(0)

    @node_property
    def logq_norm(self):
        return self.logq / self.symbolic_normalizing_constant


class GroupedApproximation(object):
    def __init__(self, groups, model=None):
        model = modelcontext(model)
        seen = set()
        rest = None
        for g in groups:
            if g.group is -1:
                if rest is not None:
                    raise TypeError('More that one group is specified for '
                                    'the rest variables')
                else:
                    rest = g
            if set(g.group) & seen:
                raise ValueError('Found duplicates in groups')
            seen.update(g.group)
        if set(model.free_RVs) - seen:
            if rest is None:
                raise ValueError('No approximation is specified for the rest variables')
            else:
                rest.__init_group__(set(model.free_RVs) - seen)
        self.groups = groups
        self.model = model

    def _collect(self, item):
        return [getattr(g, item) for g in self.groups]

    inputs = property(lambda self: self._collect('input'))
    symbolic_randoms = property(lambda self: self._collect('symbolic_random'))

    @node_property
    def symbolic_normalizing_constant(self):
        return tt.max(self._collect('symbolic_normalizing_constant'))

    @node_property
    def symbolic_logq(self):
        return tt.add(*self._collect('symbolic_logq'))

    @node_property
    def logq(self):
        return self.symbolic_logq.mean(0)

    @node_property
    def sized_symbolic_logp(self):
        logp = self.to_flat_input(self.model.logpt)
        free_logp_local = self.sample_over_posterior(logp)
        return free_logp_local  # shape (s,)

    @node_property
    def logp(self):
        return self.sized_symbolic_logp.mean(0)

    @node_property
    def single_symbolic_logp(self):
        logp = self.to_flat_input(self.model.logpt)
        post = [v[0] for v in self.symbolic_randoms]
        inp = self.inputs
        return theano.clone(
            logp, dict(zip(inp, post))
        )

    @property
    def replacements(self):
        return dict(itertools.chain(
            *[g.replacements.items()
              for g in self.groups]
        ))

    def construct_replacements(self, more_replacements=None):
        replacements = self.replacements
        if more_replacements is not None:
            replacements.update(more_replacements)
        return more_replacements

    @change_flags(compute_test_value='off')
    def set_size_and_deterministic(self, node, s, d):
        optimizations = self._get_optimization_replacements(s, d)
        node = theano.clone(node, optimizations)
        for g in self.groups:
            node = g.set_size_and_deterministic(node, s, d)
        return node

    def to_flat_input(self, node):
        """
        Replaces vars with flattened view stored in self.inputs
        """
        return theano.clone(node, self.replacements, strict=False)

    def sample_over_posterior(self, node):
        node = self.to_flat_input(node)

        def sample(*post):
            return theano.clone(node, dict(zip(self.inputs, post)))

        nodes, _ = theano.scan(
            sample, self.symbolic_randoms)
        return nodes

    def _get_optimization_replacements(self, s, d):
        repl = dict()
        if isinstance(s, int) and (s == 1) or s is None:
            repl[self.logp] = self.single_symbolic_logp
        return repl

    @change_flags(compute_test_value='off')
    def sample_node(self, node, size=100,
                    deterministic=False,
                    more_replacements=None):
        """Samples given node or nodes over shared posterior

        Parameters
        ----------
        node : Theano Variables (or Theano expressions)
        size : None or scalar
            number of samples
        more_replacements : `dict`
            add custom replacements to graph, e.g. change input source
        deterministic : bool
            whether to use zeros as initial distribution
            if True - zero initial point will produce constant latent variables

        Returns
        -------
        sampled node(s) with replacements
        """
        node_in = node
        node = theano.clone(node, more_replacements)
        if size is None:
            node_out = self.to_flat_input(node)
            node_out = theano.clone(node_out, self.replacements)
        else:
            node_out = self.sample_over_posterior(node)
        node_out = self.set_size_and_deterministic(node_out, size, deterministic)
        try_to_set_test_value(node_in, node_out, size)
        return node_out

    @property
    @memoize
    @change_flags(compute_test_value='off')
    def sample_dict_fn(self):
        s = tt.iscalar()
        flat_inp_vars = self.to_flat_input(self.model.free_RVs)
        sampled = self.sample_over_posterior(flat_inp_vars)
        sampled = self.set_size_and_deterministic(sampled, s, 0)
        sample_fn = theano.function([s], sampled)

        def inner(draws=100):
            _samples = sample_fn(draws)
            return dict([(v_.name, s_) for v_, s_ in zip(self.model.free_RVs, _samples)])

        return inner

    def sample(self, draws=500, include_transformed=True):
        """Draw samples from variational posterior.

        Parameters
        ----------
        draws : `int`
            Number of random samples.
        include_transformed : `bool`
            If True, transformed variables are also sampled. Default is False.

        Returns
        -------
        trace : :class:`pymc3.backends.base.MultiTrace`
            Samples drawn from variational posterior.
        """
        vars_sampled = get_default_varnames(self.model.unobserved_RVs,
                                            include_transformed=include_transformed)
        samples = self.sample_dict_fn(draws)  # type: dict
        trace = pm.sampling.NDArray(model=self.model, vars=vars_sampled)
        points = ({name: samples[name][i] for name in samples.keys()} for i in range(draws))
        try:
            trace.setup(draws=draws, chain=0)
            for point in points:
                trace.record(point)
        finally:
            trace.close()
        return pm.sampling.MultiTrace([trace])


class Approximation(object):
    R"""Base class for approximations.

    Parameters
    ----------
    local_rv : dict[var->tuple]
        mapping {model_variable -> local_variable (:math:`\mu`, :math:`\rho`)}
        Local Vars are used for Autoencoding Variational Bayes
        See (AEVB; Kingma and Welling, 2014) for details
    model : :class:`Model`
        PyMC3 model for inference
    cost_part_grad_scale : float or scalar tensor
        Scaling score part of gradient can be useful near optimum for
        archiving better convergence properties. Common schedule is
        1 at the start and 0 in the end. So slow decay will be ok.
        See (Sticking the Landing; Geoffrey Roeder,
        Yuhuai Wu, David Duvenaud, 2016) for details
    scale_cost_to_minibatch : bool, default False
        Scale cost to minibatch instead of full dataset
    random_seed : None or int
        leave None to use package global RandomStream or other
        valid value to create instance specific one

    Notes
    -----
    Defining an approximation needs
    custom implementation of the following methods:

        - :code:`.create_shared_params(**kwargs)`
            Returns dict

        - :code:`symbolic_random_global_matrix` node property
            It takes internally `symbolic_initial_global_matrix`
            and performs appropriate transforms. To memoize result
            one should use :code:`@node_property` wrapper instead :code:`@property`.
            Returns TensorVariable

        - :code:`.symbolic_log_q_W_global` node property
            Should use vectorized form if possible and return vector of size `(s,)`
            It is needed only if used with operator that requires :math:`logq`
            of an approximation. Returns vector

    You can also override the following methods:

        -   :code:`.check_model(model, **kwargs)`
            Do some specific check for model having `kwargs`

    `kwargs` mentioned above are supplied as additional arguments
    for :class:`Approximation`

    There are some defaults class attributes for approximation classes that can be
    optionally overridden.

        -   :code:`initial_dist_local_name = 'normal'`
            :code:`initial_dist_global_name = 'normal'`
            string that represents name of the initial distribution.
            In most cases if will be `uniform` or `normal`

        -   :code:`initial_dist_local_map = 0.
            :code:`initial_dist_global_map = 0.`
            point where initial distribution has maximum density

    References
    ----------
    -   Geoffrey Roeder, Yuhuai Wu, David Duvenaud, 2016
        Sticking the Landing: A Simple Reduced-Variance Gradient for ADVI
        approximateinference.org/accepted/RoederEtAl2016.pdf

    -   Kingma, D. P., & Welling, M. (2014).
        Auto-Encoding Variational Bayes. stat, 1050, 1.
    """
    initial_dist_local_name = 'normal'
    initial_dist_global_name = 'normal'
    initial_dist_local_map = 0.
    initial_dist_global_map = 0.
    shared_params = None

    @change_flags(compute_test_value='off')
    def __init__(self, local_rv=None, model=None,
                 cost_part_grad_scale=1,
                 scale_cost_to_minibatch=False,
                 random_seed=None, **kwargs):
        model = modelcontext(model)
        self._scale_cost_to_minibatch = theano.shared(np.int8(0))
        self.scale_cost_to_minibatch = scale_cost_to_minibatch
        if not isinstance(cost_part_grad_scale, theano.Variable):
            self.cost_part_grad_scale = theano.shared(pm.floatX(cost_part_grad_scale))
        else:
            self.cost_part_grad_scale = pm.floatX(cost_part_grad_scale)
        self._seed = random_seed
        self._rng = tt_rng(random_seed)
        self.model = model
        self.check_model(model, **kwargs)
        if local_rv is None:
            local_rv = {}

        def get_transformed(v):
            if hasattr(v, 'transformed'):
                return v.transformed
            return v

        known = {get_transformed(k): v for k, v in local_rv.items()}
        self.known = known
        self.local_vars = self.get_local_vars(**kwargs)
        self.global_vars = self.get_global_vars(**kwargs)
        self._g_order = ArrayOrdering(self.global_vars)
        self._l_order = ArrayOrdering(self.local_vars)
        self.gbij = DictToArrayBijection(self._g_order, {})
        self.lbij = DictToArrayBijection(self._l_order, {})
        self.symbolic_initial_local_matrix = tt.matrix(self.__class__.__name__ + '_symbolic_initial_local_matrix')
        self.symbolic_initial_global_matrix = tt.matrix(self.__class__.__name__ + '_symbolic_initial_global_matrix')

        self.global_flat_view = model.flatten(
            vars=self.global_vars,
            order=self._g_order,
        )
        self.local_flat_view = model.flatten(
            vars=self.local_vars,
            order=self._l_order,
        )
        self.symbolic_n_samples = self.symbolic_initial_global_matrix.shape[0]

    _global_view = property(lambda self: self.global_flat_view.view)
    _local_view = property(lambda self: self.local_flat_view.view)
    local_input = property(lambda self: self.local_flat_view.input)
    global_input = property(lambda self: self.global_flat_view.input)

    local_names = property(lambda self: tuple(v.name for v in self.local_vars))
    global_names = property(lambda self: tuple(v.name for v in self.global_vars))

    @property
    def scale_cost_to_minibatch(self):
        return bool(self._scale_cost_to_minibatch.get_value())

    @scale_cost_to_minibatch.setter
    def scale_cost_to_minibatch(self, value):
        self._scale_cost_to_minibatch.set_value(int(bool(value)))

    @staticmethod
    def _choose_alternative(part, loc, glob):
        if part == 'local':
            return loc
        elif part == 'global':
            return glob
        else:
            raise ValueError("part is restricted to be in {'local', 'global'}, got %r" % part)

    def seed(self, random_seed=None):
        """
        Reinitialize RandomStream used by this approximation

        Parameters
        ----------
        random_seed : `int`
            New random seed
        """
        self._seed = random_seed
        self._rng.seed(random_seed)

    def get_global_vars(self, **kwargs):
        return [v for v in self.model.free_RVs if v not in self.known]

    def get_local_vars(self, **kwargs):
        return [v for v in self.model.free_RVs if v in self.known]

    def check_model(self, model, **kwargs):
        """Checks that model is valid for variational inference
        """
        vars_ = [
            var for var in model.vars
            if not isinstance(var, pm.model.ObservedRV)
        ]
        if any([var.dtype in pm.discrete_types for var in vars_]
               ):  # pragma: no cover
            raise ValueError('Model should not include discrete RVs')

    def construct_replacements(self, include=None, exclude=None,
                               more_replacements=None):
        """Construct replacements with given conditions

        Parameters
        ----------
        include : `list`
            latent variables to be replaced
        exclude : `list`
            latent variables to be excluded for replacements
        more_replacements : `dict`
            add custom replacements to graph, e.g. change input source

        Returns
        -------
        `dict`
            Replacements
        """
        if include is not None and exclude is not None:
            raise ValueError(
                'Only one parameter is supported {include|exclude}, got two')
        _replacements = dict()
        _replacements.update(self.global_flat_view.replacements)
        _replacements.update(self.local_flat_view.replacements)
        if include is not None:    # pragma: no cover
            replacements = {k: v for k, v
                            in _replacements.items() if k in include}
        elif exclude is not None:  # pragma: no cover
            replacements = {k: v for k, v
                            in _replacements.items() if k not in exclude}
        else:
            replacements = _replacements
        if more_replacements is not None:   # pragma: no cover
            replacements.update(more_replacements)
        return replacements

    def to_flat_input(self, node):
        """
        Replaces vars with flattened view stored in self.input
        """
        replacements = self.construct_replacements()
        return theano.clone(node, replacements, strict=False)

    @change_flags(compute_test_value='off')
    def apply_replacements(self, node, deterministic=False,
                           include=None, exclude=None,
                           more_replacements=None):
        """Replace variables in graph with variational approximation. By default, replaces all variables

        Parameters
        ----------
        node : Theano Variables (or Theano expressions)
            node or nodes for replacements
        deterministic : bool
            whether to use zeros as initial distribution
            if True - zero initial point will produce constant latent variables
        include : `list`
            latent variables to be replaced
        exclude : `list`
            latent variables to be excluded for replacements
        more_replacements : `dict`
            add custom replacements to graph, e.g. change input source

        Returns
        -------
        node(s) with replacements
        """
        replacements = self.construct_replacements(
            include, exclude, more_replacements
        )
        node_in = node
        node = theano.clone(node, replacements, strict=False)
        posterior_glob = self.random_global(deterministic=deterministic)
        posterior_loc = self.random_local(deterministic=deterministic)
        out = theano.clone(node, {
            self.global_input: posterior_glob,
            self.local_input: posterior_loc
        }, strict=False)
        try_to_set_test_value(node_in, out, None)
        return out

    @change_flags(compute_test_value='off')
    def sample_node(self, node, size=100,
                    more_replacements=None):
        """Samples given node or nodes over shared posterior

        Parameters
        ----------
        node : Theano Variables (or Theano expressions)
        size : scalar
            number of samples
        more_replacements : `dict`
            add custom replacements to graph, e.g. change input source

        Returns
        -------
        sampled node(s) with replacements
        """
        node_in = node
        if more_replacements is not None:   # pragma: no cover
            node = theano.clone(node, more_replacements, strict=False)
        if size is None:
            size = 1
        nodes = self.sample_over_posterior(node)
        nodes = self.set_size_and_deterministic(nodes, size, 0)
        try_to_set_test_value(node_in, nodes, size)
        return nodes

    def sample_over_posterior(self, node):
        node = self.to_flat_input(node)
        posterior_loc = self.symbolic_random_local_matrix
        posterior_glob = self.symbolic_random_global_matrix

        def sample(zl, zg):
            return theano.clone(node, {
                self.local_input: zl,
                self.global_input: zg
            }, strict=False)

        nodes, _ = theano.scan(
            sample, [posterior_loc, posterior_glob])
        try_to_set_test_value(node, nodes, None)
        return nodes

    def scale_grad(self, inp):
        """Rescale gradient of input

        References
        ----------
        - Geoffrey Roeder, Yuhuai Wu, David Duvenaud, 2016
            Sticking the Landing: A Simple Reduced-Variance Gradient for ADVI
            approximateinference.org/accepted/RoederEtAl2016.pdf
        """
        return theano.gradient.grad_scale(inp, self.cost_part_grad_scale)

    @property
    def params(self):
        return collect_shared_to_list(self.shared_params)

    @change_flags(compute_test_value='off')
    def _random_part(self, part, size=None, deterministic=False):
        r_part = self._choose_alternative(
            part,
            self.symbolic_random_local_matrix,
            self.symbolic_random_global_matrix
        )
        if not isinstance(deterministic, tt.Variable):
            deterministic = np.int8(deterministic)
        if size is None:
            i_size = np.int32(1)
        else:
            i_size = size
        r_part = self.set_size_and_deterministic(r_part, i_size, deterministic)
        if size is None:
            r_part = r_part[0]
        return r_part

    def _initial_part_matrix(self, part, size, deterministic):
        if size is None:
            size = 1
        length, dist_name, dist_map = self._choose_alternative(
            part,
            (self.local_size, self.initial_dist_local_name, self.initial_dist_local_map),
            (self.global_size, self.initial_dist_global_name, self.initial_dist_global_map)
        )
        dtype = self.symbolic_initial_global_matrix.dtype
        if length == 0:  # in this case theano fails to compute sample of correct size
            return tt.ones((size, 0), dtype)
        length = tt.as_tensor(length)
        size = tt.as_tensor(size)
        shape = tt.stack((size, length))
        # apply optimizations if possible
        if not isinstance(deterministic, tt.Variable):
            if deterministic:
                return tt.ones(shape, dtype) * dist_map
            else:
                return getattr(self._rng, dist_name)(shape)
        else:
            sample = getattr(self._rng, dist_name)(shape)
            initial = tt.switch(
                deterministic,
                tt.ones(shape, dtype) * dist_map,
                sample
            )
            return initial

    def random_local(self, size=None, deterministic=False):
        """Implements posterior distribution from initial latent space

        Parameters
        ----------
        size : `scalar`
            number of samples from distribution
        deterministic : `bool`
            whether use deterministic distribution

        Returns
        -------
        local posterior space
        """
        return self._random_part('local', size=size, deterministic=deterministic)

    def random_global(self, size=None, deterministic=False):  # pragma: no cover
        """Implements posterior distribution from initial latent space

        Parameters
        ----------
        size : `scalar`
            number of samples from distribution
        deterministic : `bool`
            whether use deterministic distribution

        Returns
        -------
        global posterior space
        """
        return self._random_part('global', size=size, deterministic=deterministic)

    @change_flags(compute_test_value='off')
    def set_size_and_deterministic(self, node, s, d):
        initial_local = self._initial_part_matrix('local', s, d)
        initial_global = self._initial_part_matrix('global', s, d)
        # optimizations
        if isinstance(s, int) and (s == 1) or s is None:
            node = theano.clone(node, {
                self.logp: self.single_symbolic_logp
            })
        out = theano.clone(node, {
            self.symbolic_initial_local_matrix: initial_local,
            self.symbolic_initial_global_matrix: initial_global,
        })
        try_to_set_test_value(node, out, None)
        return out





    @node_property
    def __local_mu_rho(self):
        if not self.local_vars:
            mu, rho = (
                tt.constant(pm.floatX(np.asarray([]))),
                tt.constant(pm.floatX(np.asarray([])))
            )
        else:
            mu = []
            rho = []
            for var in self.local_vars:
                mu.append(self.known[var][0].ravel())
                rho.append(self.known[var][1].ravel())
            mu = tt.concatenate(mu)
            rho = tt.concatenate(rho)
        mu.name = self.__class__.__name__ + '_local_mu'
        rho.name = self.__class__.__name__ + '_local_rho'
        return mu, rho

    @node_property
    def normalizing_constant(self):
        """
        Constant to divide when we want to scale down loss from minibatches
        """
        t = self.to_flat_input(
            tt.max([v.scaling for v in self.model.basic_RVs]))
        t = theano.clone(t, {
            self.global_input: self.symbolic_random_global_matrix[0],
            self.local_input: self.symbolic_random_local_matrix[0]
        })
        t = self.set_size_and_deterministic(t, 1, 1)  # remove random, we do not it here at all
        # if not scale_cost_to_minibatch: t=1
        t = tt.switch(self._scale_cost_to_minibatch, t,
                      tt.constant(1, dtype=t.dtype))
        return pm.floatX(t)

    @node_property
    def symbolic_random_global_matrix(self):
        raise NotImplementedError

    @node_property
    def symbolic_random_local_matrix(self):
        mu, rho = self.__local_mu_rho
        e = self.symbolic_initial_local_matrix
        return e * rho2sd(rho) + mu

    @node_property
    def symbolic_random_total_matrix(self):
        if self.local_vars and self.global_vars:
            return tt.concatenate([
                self.symbolic_random_local_matrix,
                self.symbolic_random_global_matrix,
            ], axis=-1)
        elif self.local_vars:
            return self.symbolic_random_local_matrix
        elif self.global_vars:
            return self.symbolic_random_global_matrix
        else:
            raise TypeError('No free vars in the Model')

    @node_property
    def symbolic_log_q_W_local(self):
        mu, rho = self.__local_mu_rho
        mu = self.scale_grad(mu)
        rho = self.scale_grad(rho)
        z = self.symbolic_random_local_matrix
        logp = log_normal(z, mu, rho=rho)
        if self.local_size == 0:
            scaling = tt.constant(1, mu.dtype)
        else:
            scaling = []
            for var in self.local_vars:
                scaling.append(tt.repeat(var.scaling, var.dsize))
            scaling = tt.concatenate(scaling)
        # we need only dimensions here
        # from incoming unobserved
        # to get rid of input_view
        # I replace it with the first row
        # of total_random matrix
        # that always exists
        scaling = self.to_flat_input(scaling)
        scaling = theano.clone(scaling, {
            self.local_input: self.symbolic_random_local_matrix[0],
            self.global_input: self.symbolic_random_global_matrix[0]
        })
        logp *= scaling
        logp = logp.sum(1)
        return logp  # shape (s,)

    @node_property
    def symbolic_log_q_W_global(self):
        raise NotImplementedError  # shape (s,)

    @node_property
    def symbolic_log_q_W(self):
        q_w_local = self.symbolic_log_q_W_local
        q_w_global = self.symbolic_log_q_W_global
        return q_w_global + q_w_local  # shape (s,)

    @node_property
    def logq(self):
        """Total logq for approximation
        """
        return self.symbolic_log_q_W.mean(0)

    @node_property
    def logq_norm(self):
        return self.logq / self.normalizing_constant

    @node_property
    def sized_symbolic_logp_local(self):
        free_logp_local = tt.sum([
            var.logpt
            for var in self.model.free_RVs if var.name in self.local_names
        ])
        free_logp_local = self.sample_over_posterior(free_logp_local)
        return free_logp_local  # shape (s,)

    @node_property
    def sized_symbolic_logp_global(self):
        free_logp_global = tt.sum([
            var.logpt
            for var in self.model.free_RVs if var.name in self.global_names
        ])
        free_logp_global = self.sample_over_posterior(free_logp_global)
        return free_logp_global  # shape (s,)

    @node_property
    def sized_symbolic_logp_observed(self):
        observed_logp = tt.sum([
            var.logpt
            for var in self.model.observed_RVs
        ])
        observed_logp = self.sample_over_posterior(observed_logp)
        return observed_logp  # shape (s,)

    @node_property
    def sized_symbolic_logp_potentials(self):
        potentials = tt.sum(self.model.potentials)
        potentials = self.sample_over_posterior(potentials)
        return potentials

    @node_property
    def sized_symbolic_logp(self):
        return (self.sized_symbolic_logp_local +
                self.sized_symbolic_logp_global +
                self.sized_symbolic_logp_observed +
                self.sized_symbolic_logp_potentials)

    @node_property
    def single_symbolic_logp(self):
        logp = self.to_flat_input(self.model.logpt)
        loc = self.symbolic_random_local_matrix[0]
        glob = self.symbolic_random_global_matrix[0]
        iloc = self.local_input
        iglob = self.global_input
        return theano.clone(
            logp, {
                iloc: loc,
                iglob: glob
            }
        )

    @node_property
    def logp(self):
        return self.sized_symbolic_logp.mean(0)

    @node_property
    def logp_norm(self):
        return self.logp / self.normalizing_constant

    def _view_part(self, part, space, name, reshape=True):
        theano_is_here = isinstance(space, tt.Variable)
        if not theano_is_here:
            raise TypeError('View on numpy arrays is not supported')
        if part == 'global':
            _, slc, _shape, dtype = self._global_view[name]
        elif part == 'local':
            _, slc, _shape, dtype = self._local_view[name]
        else:
            raise ValueError("%r part is not supported, you can use only {'local', 'global'}")
        if space.ndim > 2:
            raise ValueError('Space should have <= 2 dimensions, got %r' % space.ndim)
        view = space[..., slc]
        if reshape:
            shape = np.asarray((-1,) + _shape, int)
            view = view.reshape(shape, ndim=len(_shape) + 1)
        if space.ndim == 1:
            view = view[0]
        return view.astype(dtype)

    def view_global(self, space, name, reshape=True):
        """Construct view on a variable from flattened `space`

        Parameters
        ----------
        space : matrix or vector
            space to take view of variable from
        name : `str`
            name of variable
        reshape : `bool`
            whether to reshape variable from vectorized view

        Returns
        -------
        (reshaped) slice of matrix
            variable view
        """
        return self._view_part('global', space, name, reshape)

    def view_local(self, space, name, reshape=True):
        """Construct view on a variable from flattened `space`

        Parameters
        ----------
        space : matrix or vector
            space to take view of variable from
        name : `str`
            name of variable
        reshape : `bool`
            whether to reshape variable from vectorized view

        Returns
        -------
        (reshaped) slice of matrix
            variable view
        """
        return self._view_part('local', space, name, reshape)

    @property
    def total_size(self):
        return self.local_size + self.global_size

    @property
    def local_size(self):
        return self._l_order.size

    @property
    def global_size(self):
        return self._g_order.size
