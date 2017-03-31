import numpy as np
import theano
import theano.tensor as tt

import pymc3 as pm
from .updates import adam
from ..distributions.dist_math import rho2sd, log_normal
from ..model import modelcontext, ArrayOrdering
from ..theanof import tt_rng, memoize, change_flags, GradScale


class ObjectiveUpdates(theano.OrderedUpdates):
    """
    OrderedUpdates extension for storing loss
    """
    loss = None


class Operator(object):
    """
    Base class for Operator

    Parameters
    ----------
    approx : Approximation

    Subclassing
    -----------
    For implementing Custom operator it is needed to define `.apply(f)` method
    """

    NEED_F = False

    def __init__(self, approx):
        self.model = approx.model
        self.approx = approx

    flat_view = property(lambda self: self.approx.flat_view)
    input = property(lambda self: self.approx.flat_view.input)

    def logp(self, z):
        p = self.approx.to_flat_input(self.model.logpt)
        p = theano.clone(p, {self.input: z})
        return p

    def logq(self, z):
        return self.approx.logq(z)

    def apply(self, f):   # pragma: no cover
        """
        Operator itself
        .. math::

            (O^{p,q}f_{\theta})(z)

        Parameters
        ----------
        f : function or None
            function that takes `z = self.input` and returns
            same dimension output

        Returns
        -------
        symbolically applied operator
        """
        raise NotImplementedError

    def __call__(self, f=None):
        if f is None:
            if self.NEED_F:
                raise ValueError('Operator %s requires TestFunction' % self)
            else:
                f = TestFunction()
        elif not isinstance(f, TestFunction):
            f = TestFunction.from_function(f)
        f.setup(self.approx.total_size)
        return ObjectiveFunction(self, f)

    def __getstate__(self):
        # pickle only important parts
        return self.approx

    def __setstate__(self, approx):
        self.__init__(approx)

    def __str__(self):    # pragma: no cover
        return '%(op)s[%(ap)s]' % dict(op=self.__class__.__name__,
                                       ap=self.approx.__class__.__name__)


class ObjectiveFunction(object):
    """
    Helper class for construction loss and updates for variational inference

    Parameters
    ----------
    op : Operator
    tf : TestFunction
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
        size : int
            number of samples from distribution

        Returns
        -------
        posterior space (theano)
        """
        return self.op.approx.random(size)

    def __call__(self, z):
        if z.ndim > 1:
            a = theano.scan(
                lambda z_: theano.clone(self.op.apply(self.tf), {self.op.input: z_}),
                sequences=z, n_steps=z.shape[0])[0].mean()
        else:
            a = theano.clone(self.op.apply(self.tf), {self.op.input: z})
        return tt.abs_(a)

    def updates(self, obj_n_mc=None, tf_n_mc=None, obj_optimizer=adam, test_optimizer=adam,
                more_obj_params=None, more_tf_params=None, more_updates=None):
        """
        Calculates gradients for objective function, test function and then
        constructs updates for optimization step

        Parameters
        ----------
        obj_n_mc : int
            Number of monte carlo samples used for approximation of objective gradients
        tf_n_mc : int
            Number of monte carlo samples used for approximation of test function gradients
        obj_optimizer : function (loss, params) -> updates
            Optimizer that is used for objective params
        test_optimizer : function (loss, params) -> updates
            Optimizer that is used for test function params
        more_obj_params : list
            Add custom params for objective optimizer
        more_tf_params : list
            Add custom params for test function optimizer
        more_updates : dict
            Add custom updates to resulting updates

        Returns
        -------
        ObjectiveUpdates
        """
        if more_obj_params is None:
            more_obj_params = []
        if more_tf_params is None:
            more_tf_params = []
        if more_updates is None:
            more_updates = dict()
        resulting_updates = ObjectiveUpdates()

        if self.test_params:
            tf_z = self.random(tf_n_mc)
            tf_target = -self(tf_z)
            resulting_updates.update(test_optimizer(tf_target, self.test_params + more_tf_params))
        else:
            pass
        obj_z = self.random(obj_n_mc)
        obj_target = self(obj_z)
        resulting_updates.update(obj_optimizer(obj_target, self.obj_params + more_obj_params))
        resulting_updates.update(more_updates)
        resulting_updates.loss = obj_target
        return resulting_updates

    @memoize
    def step_function(self, obj_n_mc=None, tf_n_mc=None,
                      obj_optimizer=adam, test_optimizer=adam,
                      more_obj_params=None, more_tf_params=None,
                      more_updates=None, score=False,
                      fn_kwargs=None):
        """
        Step function that should be called on each optimization step.

        Generally it solves the following problem:
        .. math::

                \textbf{\lambda^{*}} = \inf_{\lambda} \sup_{\theta} t(\mathbb{E}_{\lambda}[(O^{p,q}f_{\theta})(z)])

        Parameters
        ----------
        obj_n_mc : int
            Number of monte carlo samples used for approximation of objective gradients
        tf_n_mc : int
            Number of monte carlo samples used for approximation of test function gradients
        obj_optimizer : function (loss, params) -> updates
            Optimizer that is used for objective params
        test_optimizer : function (loss, params) -> updates
            Optimizer that is used for test function params
        more_obj_params : list
            Add custom params for objective optimizer
        more_tf_params : list
            Add custom params for test function optimizer
        more_updates : dict
            Add custom updates to resulting updates
        score : bool
            calculate loss on each step? Defaults to False for speed
        fn_kwargs : dict
            Add kwargs to theano.function (e.g. `{'profile': True}`)
        Returns
        -------
        theano.function
        """
        if fn_kwargs is None:
            fn_kwargs = {}
        updates = self.updates(obj_n_mc=obj_n_mc, tf_n_mc=tf_n_mc,
                               obj_optimizer=obj_optimizer,
                               test_optimizer=test_optimizer,
                               more_obj_params=more_obj_params,
                               more_tf_params=more_tf_params,
                               more_updates=more_updates)
        if score:
            step_fn = theano.function([], updates.loss, updates=updates, **fn_kwargs)
        else:
            step_fn = theano.function([], None, updates=updates, **fn_kwargs)
        return step_fn

    @memoize
    def score_function(self, sc_n_mc=None, fn_kwargs=None):   # pragma: no cover
        if fn_kwargs is None:
            fn_kwargs = {}
        return theano.function([], self(self.random(sc_n_mc)), **fn_kwargs)

    def __getstate__(self):
        return self.op, self.tf

    def __setstate__(self, state):
        self.__init__(*state)


def cast_to_list(params):
    """
    Helper function for getting a list from
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
        raise TypeError('Unknown type %s for %r, need list, dict or shared variable')


class TestFunction(object):
    def __init__(self):
        self._inited = False
        self.shared_params = None

    def create_shared_params(self, dim):
        """
        Returns
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
        """
        Does some preparation stuff before calling `.create_shared_params()`

        Parameters
        ----------
        dim : int dimension of posterior distribution
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
    """
    Base class for approximations.

    Parameters
    ----------
    local_rv : dict
        mapping {model_variable -> local_variable}
        Local Vars are used for Autoencoding Variational Bayes
        See (AEVB; Kingma and Welling, 2014) for details

    model : PyMC3 model for inference

    cost_part_grad_scale : float or scalar tensor
        Scaling score part of gradient can be useful near optimum for
        archiving better convergence properties. Common schedule is
        1 at the start and 0 in the end. So slow decay will be ok.
        See (Sticking the Landing; Geoffrey Roeder,
        Yuhuai Wu, David Duvenaud, 2016) for details

    Subclassing
    -----------
    Defining an approximation needs
    custom implementation of the following methods:
        - `.create_shared_params()`
            Returns {dict|list|theano.shared}

        - `.random_global(size=None, no_rand=False)`
            Generate samples from posterior. If `no_rand==False`:
            sample from MAP of initial distribution.
            Returns TensorVariable

        - `.log_q_W_global(z)`
            It is needed only if used with operator
            that requires :math:`logq` of an approximation
            Returns Scalar

    Notes
    -----
    There are some defaults for approximation classes that can be
    optionally overriden.
        - `initial_dist_name`
            string that represents name of the initial distribution.
            In most cases if will be `uniform` or `normal`
        - `initial_dist_map`
            float where initial distribution has maximum density

    References
    ----------
    - Geoffrey Roeder, Yuhuai Wu, David Duvenaud, 2016
        Sticking the Landing: A Simple Reduced-Variance Gradient for ADVI
        approximateinference.org/accepted/RoederEtAl2016.pdf

    - Kingma, D. P., & Welling, M. (2014).
      Auto-Encoding Variational Bayes. stat, 1050, 1.
    """
    initial_dist_name = 'normal'
    initial_dist_map = 0.

    def __init__(self, local_rv=None, model=None, cost_part_grad_scale=1):
        model = modelcontext(model)
        self.model = model
        self.check_model(model)
        if local_rv is None:
            local_rv = {}

        def get_transformed(v):
            if hasattr(v, 'transformed'):
                return v.transformed
            return v

        known = {get_transformed(k): v for k, v in local_rv.items()}
        self.known = known
        self.local_vars = self.get_local_vars()
        self.global_vars = self.get_global_vars()
        self.order = ArrayOrdering(self.local_vars + self.global_vars)
        self.flat_view = model.flatten(
            vars=self.local_vars + self.global_vars
        )
        self.grad_scale_op = GradScale(cost_part_grad_scale)
        self._setup()
        self.shared_params = self.create_shared_params()

    def _setup(self):
        pass

    def get_global_vars(self):
        return [v for v in self.model.free_RVs if v not in self.known]

    def get_local_vars(self):
        return [v for v in self.model.free_RVs if v in self.known]

    def __getstate__(self):
        state = self.__dict__.copy()
        # can be inferred from the rest parts
        state.pop('flat_view')
        state.pop('order')
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.order = ArrayOrdering(self.local_vars + self.global_vars)
        self.flat_view = self.model.flatten(
            vars=self.local_vars + self.global_vars
        )

    _view = property(lambda self: self.flat_view.view)
    input = property(lambda self: self.flat_view.input)

    def check_model(self, model):
        """
        Checks that model is valid for variational inference
        """
        vars_ = [var for var in model.vars if not isinstance(var, pm.model.ObservedRV)]
        if any([var.dtype in pm.discrete_types for var in vars_]):  # pragma: no cover
            raise ValueError('Model should not include discrete RVs')

    def create_shared_params(self):
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
        """
        Construct replacements with given conditions

        Parameters
        ----------
        include : list
            latent variables to be replaced
        exclude : list
            latent variables to be excluded for replacements
        more_replacements : dict
            add custom replacements to graph, e.g. change input source

        Returns
        -------
        dict
            Replacements
        """
        if include is not None and exclude is not None:
            raise ValueError('Only one parameter is supported {include|exclude}, got two')
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
        """
        Replace variables in graph with variational approximation. By default, replaces all variables

        Parameters
        ----------
        node : Theano Variables (or Theano expressions)
            node or nodes for replacements
        deterministic : bool
            whether to use zeros as initial distribution
            if True - zero initial point will produce constant latent variables
        include : list
            latent variables to be replaced
        exclude : list
            latent variables to be excluded for replacements
        more_replacements : dict
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
        return theano.clone(node, {self.input: posterior})

    def sample_node(self, node, size=100,
                    more_replacements=None):
        """
        Samples given node or nodes over shared posterior

        Parameters
        ----------
        node : Theano Variables (or Theano expressions)
        size : scalar
            number of samples
        more_replacements : dict
            add custom replacements to graph, e.g. change input source

        Returns
        -------
        sampled node(s) with replacements
        """
        if more_replacements is not None:   # pragma: no cover
            node = theano.clone(node, more_replacements)
        posterior = self.random(size)
        node = self.to_flat_input(node)

        def sample(z): return theano.clone(node, {self.input: z})
        nodes, _ = theano.scan(sample, posterior, n_steps=size)
        return nodes

    def scale_grad(self, inp):
        """
        Rescale gradient of input

        References
        ----------
        - Geoffrey Roeder, Yuhuai Wu, David Duvenaud, 2016
            Sticking the Landing: A Simple Reduced-Variance Gradient for ADVI
            approximateinference.org/accepted/RoederEtAl2016.pdf
        """
        return self.grad_scale_op(inp)

    def to_flat_input(self, node):
        """
        Replaces vars with flattened view stored in self.input
        """
        return theano.clone(node, self.flat_view.replacements, strict=False)

    @property
    def params(self):
        return cast_to_list(self.shared_params)

    def initial(self, size, no_rand=False, l=None):
        """
        Initial distribution for constructing posterior

        Parameters
        ----------
        size : int - number of samples
        no_rand : bool - return zeros if True
        l : length of sample, defaults to latent space dim

        Returns
        -------
        Tensor
            sampled latent space shape == size + latent_dim
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
            sample = getattr(tt_rng(), self.initial_dist_name)(shape)
            space = tt.switch(
                no_rand,
                tt.ones_like(sample) * self.initial_dist_map,
                sample
            )
        else:
            if no_rand:
                return tt.ones(shape) * self.initial_dist_map
            else:
                return getattr(tt_rng(), self.initial_dist_name)(shape)
        return space

    def random_local(self, size=None, no_rand=False):
        """
        Implements posterior distribution from initial latent space

        Parameters
        ----------
        size : number of samples from distribution
        no_rand : whether use deterministic distribution

        Returns
        -------
        local posterior space
        """

        mu, rho = self._local_mu_rho()
        e = self.initial(size, no_rand, self.local_size)
        return e * rho2sd(rho) + mu

    def random_global(self, size=None, no_rand=False):  # pragma: no cover
        """
        Implements posterior distribution from initial latent space

        Parameters
        ----------
        size : number of samples from distribution
        no_rand : whether use deterministic distribution

        Returns
        -------
        global posterior space
        """
        raise NotImplementedError

    def random(self, size=None, no_rand=False):
        """
        Implements posterior distribution from initial latent space

        Parameters
        ----------
        size : number of samples from distribution
        no_rand : whether use deterministic distribution

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
        """
        Implements posterior distribution from initial latent space

        Parameters
        ----------
        size : number of samples from distribution
        no_rand : whether use deterministic distribution

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

    def sample_vp(self, draws=1, hide_transformed=False):
        """
        Draw samples from variational posterior.

        Parameters
        ----------
        draws : int
            Number of random samples.
        hide_transformed : bool
            If False, transformed variables are also sampled. Default is True.

        Returns
        -------
        trace : pymc3.backends.base.MultiTrace
            Samples drawn from the variational posterior.
        """
        if hide_transformed:
            vars_sampled = [v_ for v_ in self.model.unobserved_RVs
                            if not str(v_).endswith('_')]
        else:
            vars_sampled = [v_ for v_ in self.model.unobserved_RVs]
        posterior = self.random_fn(draws)
        names = [var.name for var in self.local_vars + self.global_vars]
        samples = {name: self.view(posterior, name)
                   for name in names}

        def points():
            for i in range(draws):
                yield {name: samples[name][i]
                       for name in names}

        trace = pm.sampling.NDArray(model=self.model, vars=vars_sampled)
        try:
            trace.setup(draws=draws, chain=0)
            for point in points():
                trace.record(point)
        finally:
            trace.close()
        return pm.sampling.MultiTrace([trace])

    def log_q_W_local(self, z):
        """
        log_q_W samples over q for local vars
        Gradient wrt mu, rho in density parametrization
        is set to zero to lower variance of ELBO
        """
        if not self.local_vars:
            return tt.constant(0)
        mu, rho = self._local_mu_rho()
        mu = self.scale_grad(mu)
        rho = self.scale_grad(rho)
        logp = log_normal(z[self.local_slc], mu, rho=rho)
        scaling = []
        for var in self.local_vars:
            scaling.append(tt.ones(var.dsize)*var.scaling)
        scaling = tt.concatenate(scaling)
        if z.ndim > 1:  # pragma: no cover
            # rare case when logq(z) is called directly
            logp *= scaling[None]
        else:
            logp *= scaling
        return self.to_flat_input(tt.sum(logp))

    def log_q_W_global(self, z):    # pragma: no cover
        """
        log_q_W samples over q for global vars
        """
        raise NotImplementedError

    def logq(self, z):
        """
        Total logq for approximation
        """
        return self.log_q_W_global(z) + self.log_q_W_local(z)

    def view(self, space, name, reshape=True):
        """
        Construct view on a variable from flattened `space`

        Parameters
        ----------
        space : space to take view of variable from
        name : str
            name of variable
        reshape : bool
            whether to reshape variable from vectorized view

        Returns
        -------
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
            raise ValueError('Space should have no more than 2 dims, got %d' % space.ndim)
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
