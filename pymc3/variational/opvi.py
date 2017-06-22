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
import numpy as np
import theano
import theano.tensor as tt

import pymc3 as pm
from .updates import adagrad_window
from ..distributions.dist_math import rho2sd, log_normal
from ..model import modelcontext, ArrayOrdering, DictToArrayBijection
from ..util import get_default_varnames
from ..theanof import tt_rng, memoize, change_flags, identity


__all__ = [
    'ObjectiveFunction',
    'Operator',
    'TestFunction',
    'Approximation'
]


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

    def random(self, size=None):
        """
        Posterior distribution from initial latent space

        Parameters
        ----------
        size : `int`
            number of samples from distribution

        Returns
        -------
        posterior space (theano)
        """
        return self.op.approx.random(size)

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
        tf_z = self.get_input(tf_n_mc)
        tf_target = self(tf_z, more_tf_params=more_tf_params)
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
        obj_z = self.get_input(obj_n_mc)
        obj_target = self(obj_z, more_obj_params=more_obj_params)
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

    def get_input(self, n_mc):
        return self.random(n_mc)

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
            self(self.random(sc_n_mc)),
            more_replacements,
            strict=False)
        return theano.function([], loss, **fn_kwargs)

    def __getstate__(self):
        return self.op, self.tf

    def __setstate__(self, state):
        self.__init__(*state)

    def __call__(self, z, **kwargs):
        if 'more_tf_params' in kwargs:
            m = -1
        else:
            m = 1
        if z.ndim > 1:
            a = theano.scan(
                lambda z_: theano.clone(
                    self.op.apply(self.tf),
                    {self.op.input: z_}, strict=False),
                sequences=z, n_steps=z.shape[0])[0].mean()
        else:
            a = theano.clone(
                self.op.apply(self.tf),
                {self.op.input: z}, strict=False)
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


def cast_to_list(params):
    """Helper function for getting a list from
    usable representation of parameters

    Parameters
    ----------
    params : {list|tuple|dict|theano.shared|None}

    Returns
    -------
    list
    """
    if isinstance(params, list):
        return params
    elif isinstance(params, tuple):
        return list(params)
    elif isinstance(params, dict):
        return list(params.values())
    elif isinstance(params, theano.compile.SharedVariable):
        return [params]
    elif params is None:
        return []
    else:
        raise TypeError(
            'Unknown type %s for %r, need list, dict or shared variable')


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
        return cast_to_list(self.shared_params)

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
            Returns {dict|list|theano.shared}

        - :code:`.random_global(size=None, no_rand=False)`
            Generate samples from posterior. If `no_rand==False`:
            sample from MAP of initial distribution.
            Returns TensorVariable

        - :code:`.log_q_W_global(z)`
            It is needed only if used with operator
            that requires :math:`logq` of an approximation
            Returns Scalar

    You can also override the following methods:

        -   :code:`._setup(**kwargs)`
            Do some specific stuff having `kwargs` before calling :func:`Approximation.create_shared_params`

        -   :code:`.check_model(model, **kwargs)`
            Do some specific check for model having `kwargs`

    `kwargs` mentioned above are supplied as additional arguments
    for :class:`Approximation`

    There are some defaults class attributes for approximation classes that can be
    optionally overridden.

        -   :code:`initial_dist_name`
            string that represents name of the initial distribution.
            In most cases if will be `uniform` or `normal`

        -   :code:`initial_dist_map`
            float where initial distribution has maximum density

    References
    ----------
    -   Geoffrey Roeder, Yuhuai Wu, David Duvenaud, 2016
        Sticking the Landing: A Simple Reduced-Variance Gradient for ADVI
        approximateinference.org/accepted/RoederEtAl2016.pdf

    -   Kingma, D. P., & Welling, M. (2014).
        Auto-Encoding Variational Bayes. stat, 1050, 1.
    """
    initial_dist_name = 'normal'
    initial_dist_map = 0.

    def __init__(self, local_rv=None, model=None,
                 cost_part_grad_scale=1,
                 scale_cost_to_minibatch=False,
                 random_seed=None, **kwargs):
        model = modelcontext(model)
        self.scale_cost_to_minibatch = theano.shared(np.int8(0))
        if scale_cost_to_minibatch:
            self.scale_cost_to_minibatch.set_value(1)
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
        self.order = ArrayOrdering(self.local_vars + self.global_vars)
        self.gbij = DictToArrayBijection(ArrayOrdering(self.global_vars), {})
        self.lbij = DictToArrayBijection(ArrayOrdering(self.local_vars), {})
        self.flat_view = model.flatten(
            vars=self.local_vars + self.global_vars
        )
        self._setup(**kwargs)
        self.shared_params = self.create_shared_params(**kwargs)

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

    @property
    def normalizing_constant(self):
        t = self.to_flat_input(
            tt.max([v.scaling for v in self.model.basic_RVs]))
        t = theano.clone(t, {self.input: tt.zeros(self.total_size)})
        # if not scale_cost_to_minibatch: t=1
        t = tt.switch(self.scale_cost_to_minibatch, t,
                      tt.constant(1, dtype=t.dtype))
        return pm.floatX(t)

    def _setup(self, **kwargs):
        pass

    def get_global_vars(self, **kwargs):
        return [v for v in self.model.free_RVs if v not in self.known]

    def get_local_vars(self, **kwargs):
        return [v for v in self.model.free_RVs if v in self.known]

    _view = property(lambda self: self.flat_view.view)
    input = property(lambda self: self.flat_view.input)

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

    def create_shared_params(self, **kwargs):
        """
        Returns
        -------
        {dict|list|theano.shared}
        """
        pass

    def _local_mu_rho(self):
        mu = []
        rho = []
        for var in self.local_vars:
            mu.append(self.known[var][0].ravel())
            rho.append(self.known[var][1].ravel())
        mu = tt.concatenate(mu)
        rho = tt.concatenate(rho)
        return mu, rho

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
        if include is not None:    # pragma: no cover
            replacements = {k: v for k, v
                            in self.flat_view.replacements.items() if k in include}
        elif exclude is not None:  # pragma: no cover
            replacements = {k: v for k, v
                            in self.flat_view.replacements.items() if k not in exclude}
        else:
            replacements = self.flat_view.replacements.copy()
        if more_replacements is not None:   # pragma: no cover
            replacements.update(more_replacements)
        return replacements

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
        node = theano.clone(node, replacements, strict=False)
        posterior = self.random(no_rand=deterministic)
        return theano.clone(node, {self.input: posterior}, strict=False)

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
        if more_replacements is not None:   # pragma: no cover
            node = theano.clone(node, more_replacements, strict=False)
        posterior = self.random(size)
        node = self.to_flat_input(node)

        def sample(z): return theano.clone(node, {self.input: z}, strict=False)
        nodes, _ = theano.scan(sample, posterior, n_steps=size)
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

    def to_flat_input(self, node):
        """
        Replaces vars with flattened view stored in self.input
        """
        return theano.clone(node, self.flat_view.replacements, strict=False)

    @property
    def params(self):
        return cast_to_list(self.shared_params)

    def initial(self, size, no_rand=False, l=None):
        """Initial distribution for constructing posterior

        Parameters
        ----------
        size : `int`
            number of samples
        no_rand : `bool`
            return zeros if True
        l : `int`
            length of sample, defaults to latent space dim

        Returns
        -------
        `tt.TensorVariable`
            sampled latent space
        """

        theano_condition_is_here = isinstance(no_rand, tt.Variable)
        if l is None:   # pragma: no cover
            l = self.total_size
        if size is None:
            shape = (l, )
        else:
            shape = (size, l)
        shape = tt.stack(*shape)
        if theano_condition_is_here:
            no_rand = tt.as_tensor(no_rand)
            sample = getattr(self._rng, self.initial_dist_name)(shape)
            space = tt.switch(
                no_rand,
                tt.ones_like(sample) * self.initial_dist_map,
                sample
            )
        else:
            if no_rand:
                return tt.ones(shape) * self.initial_dist_map
            else:
                return getattr(self._rng, self.initial_dist_name)(shape)
        return space

    def random_local(self, size=None, no_rand=False):
        """Implements posterior distribution from initial latent space

        Parameters
        ----------
        size : `scalar`
            number of samples from distribution
        no_rand : `bool`
            whether use deterministic distribution

        Returns
        -------
        local posterior space
        """

        mu, rho = self._local_mu_rho()
        e = self.initial(size, no_rand, self.local_size)
        return e * rho2sd(rho) + mu

    def random_global(self, size=None, no_rand=False):  # pragma: no cover
        """Implements posterior distribution from initial latent space

        Parameters
        ----------
        size : `scalar`
            number of samples from distribution
        no_rand : `bool`
            whether use deterministic distribution

        Returns
        -------
        global posterior space
        """
        raise NotImplementedError

    def random(self, size=None, no_rand=False):
        """Implements posterior distribution from initial latent space

        Parameters
        ----------
        size : `scalar`
            number of samples from distribution
        no_rand : `bool`
            whether use deterministic distribution

        Returns
        -------
        posterior space (theano)
        """
        if size is None:
            ax = 0
        else:
            ax = 1
        if self.local_vars and self.global_vars:
            return tt.concatenate([
                self.random_local(size, no_rand),
                self.random_global(size, no_rand)
            ], axis=ax)
        elif self.local_vars:   # pragma: no cover
            return self.random_local(size, no_rand)
        elif self.global_vars:
            return self.random_global(size, no_rand)
        else:   # pragma: no cover
            raise ValueError('No FreeVARs in model')

    @property
    @memoize
    @change_flags(compute_test_value='off')
    def random_fn(self):
        """Implements posterior distribution from initial latent space

        Parameters
        ----------
        size : `int`
            number of samples from distribution
        no_rand : `bool`
            whether use deterministic distribution

        Returns
        -------
        posterior space (numpy)
        """
        In = theano.In
        size = tt.iscalar('size')
        no_rand = tt.bscalar('no_rand')
        posterior = self.random(size, no_rand)
        fn = theano.function([In(size, 'size', 1, allow_downcast=True),
                              In(no_rand, 'no_rand', 0, allow_downcast=True)],
                             posterior)

        def inner(size=None, no_rand=False):
            if size is None:
                return fn(1, int(no_rand))[0]
            else:
                return fn(size, int(no_rand))

        return inner

    def sample(self, draws=1, include_transformed=False):
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
        posterior = self.random_fn(draws)
        names = [var.name for var in self.local_vars + self.global_vars]
        samples = {name: self.view(posterior, name)
                   for name in names}

        def points():
            for i in range(draws):
                yield {name: samples[name][i] for name in names}

        trace = pm.sampling.NDArray(model=self.model, vars=vars_sampled)
        try:
            trace.setup(draws=draws, chain=0)
            for point in points():
                trace.record(point)
        finally:
            trace.close()
        return pm.sampling.MultiTrace([trace])

    def log_q_W_local(self, z):
        """log_q_W samples over q for local vars
        Gradient wrt mu, rho in density parametrization
        can be scaled to lower variance of ELBO
        """
        if not self.local_vars:
            return tt.constant(0)
        mu, rho = self._local_mu_rho()
        mu = self.scale_grad(mu)
        rho = self.scale_grad(rho)
        logp = log_normal(z[self.local_slc], mu, rho=rho)
        scaling = []
        for var in self.local_vars:
            scaling.append(tt.repeat(var.scaling, var.dsize))
        scaling = tt.concatenate(scaling)
        logp *= scaling
        return self.to_flat_input(tt.sum(logp))

    def log_q_W_global(self, z):    # pragma: no cover
        """log_q_W samples over q for global vars
        """
        raise NotImplementedError

    def logq(self, z):
        """Total logq for approximation
        """
        return self.log_q_W_global(z) + self.log_q_W_local(z)

    def logq_norm(self, z):
        return self.logq(z) / self.normalizing_constant

    def logp(self, z):
        factors = ([tt.sum(var.logpt)for var in self.model.basic_RVs] +
                   [tt.sum(var) for var in self.model.potentials])
        p = self.to_flat_input(tt.add(*factors))
        p = theano.clone(p, {self.input: z})
        return p

    def logp_norm(self, z):
        t = self.normalizing_constant
        factors = ([tt.sum(var.logpt) / t for var in self.model.basic_RVs] +
                   [tt.sum(var) / t for var in self.model.potentials])
        logpt = tt.add(*factors)
        p = self.to_flat_input(logpt)
        p = theano.clone(p, {self.input: z})
        return p

    def view(self, space, name, reshape=True):
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
        theano_is_here = isinstance(space, tt.TensorVariable)
        slc = self._view[name].slc
        _, _, _shape, dtype = self._view[name]
        if space.ndim == 2:
            view = space[:, slc]
        elif space.ndim < 2:
            view = space[slc]
        else:   # pragma: no cover
            raise ValueError(
                'Space should have no more than 2 dims, got %d' %
                space.ndim)
        if reshape:
            if len(_shape) > 0:
                if theano_is_here:
                    shape = tt.concatenate([space.shape[:-1],
                                            tt.as_tensor(_shape)])
                else:
                    shape = np.concatenate([space.shape[:-1],
                                            _shape]).astype(int)

            else:
                shape = space.shape[:-1]
            if theano_is_here:
                view = view.reshape(shape, ndim=space.ndim + len(_shape) - 1)
            else:
                view = view.reshape(shape)
        return view.astype(dtype)

    @property
    def total_size(self):
        return self.order.dimensions

    @property
    def local_size(self):
        size = sum([0] + [v.dsize for v in self.local_vars])
        return size

    @property
    def global_size(self):
        return self.total_size - self.local_size

    @property
    def local_slc(self):
        return slice(0, self.local_size)

    @property
    def global_slc(self):
        return slice(self.local_size, self.total_size)
