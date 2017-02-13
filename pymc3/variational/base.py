import numpy as np
import theano
import theano.tensor as tt
from theano.ifelse import ifelse

import pymc3 as pm
from .advi import adagrad_optimizer
from ..distributions.dist_math import rho2sd, log_normal
from ..model import modelcontext
from ..theanof import tt_rng, memoize, change_flags, GradScale


class Operator(object):
    NEED_F = False

    def __init__(self, approx):
        self.model = approx.model
        self.approx = approx

    flat_view = property(lambda self: self.approx.flat_view)
    input = property(lambda self: self.approx.flat_view.input)

    def logp(self, z):
        p = theano.clone(self.model.logpt, self.flat_view.replacements, strict=False)
        p = theano.clone(p, {self.input: z})
        return p

    def logq(self, z):
        return self.approx.logq(z)

    def apply(self, f):
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
        return Operator.ObjectiveFunction(self, f)

    class ObjectiveFunction(object):
        def __init__(self, op, tf):
            self.tf = tf
            self.op = op

        obj_params = property(lambda self: self.op.approx.params)
        test_params = property(lambda self: self.tf.params)

        def random(self, size):
            return self.op.approx.random(size)

        def __call__(self, z):
            if z.ndim > 1:
                a = theano.scan(
                    lambda z_: theano.clone(self.op.apply(self.tf), {self.op.input: z_}),
                    sequences=z, n_steps=z.shape[0])[0].mean()
            else:
                a = theano.clone(self.op.apply(self.tf), {self.op.input: z})
            return tt.abs_(a)

        def updates(self, obj_n_mc=None, tf_n_mc=None, obj_optimizer=None, test_optimizer=None,
                    more_params=None, more_updates=None):
            if more_params is None:
                more_params = []
            if more_updates is None:
                more_updates = dict()
            if obj_optimizer is None:
                obj_optimizer = adagrad_optimizer(learning_rate=.01, epsilon=.1)
            if test_optimizer is None:
                test_optimizer = adagrad_optimizer(learning_rate=.01, epsilon=.1)
            resulting_updates = theano.OrderedUpdates()

            if self.test_params:
                tf_z = self.random(tf_n_mc)
                tf_target = -self(tf_z)
                resulting_updates.update(test_optimizer(tf_target, self.test_params))
            else:
                pass
            obj_z = self.random(obj_n_mc)
            obj_target = self(obj_z)
            resulting_updates.update(obj_optimizer(obj_target, self.obj_params + more_params))
            resulting_updates.update(more_updates)
            return resulting_updates

        def step_function(self, obj_n_mc=None, tf_n_mc=None,
                          obj_optimizer=None, test_optimizer=None,
                          more_params=None, more_updates=None,
                          score=True):
            updates = self.updates(obj_n_mc, tf_n_mc,
                                   obj_optimizer, test_optimizer,
                                   more_params, more_updates)
            step_fn = theano.function([], [], updates=updates)
            if score:
                val_fun = theano.function([], self(self.random()))
            else:
                val_fun = None

            if val_fun is not None:
                def step():
                    step_fn()
                    return val_fun()
            else:
                step = step_fn
            return step

    def __str__(self):
        return '%(op)s[%(ap)s]' % dict(op=self.__class__.__name__,
                                       ap=self.approx.__class__.__name__)


def cast_to_list(params):
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
        pass

    @classmethod
    def from_function(cls, f):
        if not callable(f):
            raise ValueError('Need callable, got %r' % f)
        obj = TestFunction()
        obj.__call__ = f
        return obj


class Approximation(object):
    initial_dist_name = 'normal'
    initial_dist_map = 0.

    # TODO: docs
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
        self.local_vars = [v for v in model.free_RVs if v in known]
        self.global_vars = [v for v in model.free_RVs if v not in known]
        self.flat_view, self.order, self._view = model.flatten(
            vars=self.local_vars + self.global_vars
        )
        self.cost_part_grad_scale = cost_part_grad_scale
        self.shared_params = self.create_shared_params()

    input = property(lambda self: self.flat_view.input)

    @staticmethod
    def check_model(model):
        """
        Checks that model is valid for variational inference
        """
        vars_ = [var for var in model.vars if not isinstance(var, pm.model.ObservedRV)]
        if any([var.dtype in pm.discrete_types for var in vars_]):
            raise ValueError('Model should not include discrete RVs')

    def create_shared_params(self):
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

    def apply_replacements(self, node, deterministic=False, include=None, exclude=None):
        """
        Replace variables in graph with variational approximation. By default, replaces all variables

        Parameters
        ----------
        node : Variable
            node for replacements
        deterministic : bool
            whether to use zeros as initial distribution
            if True - zero initial point will produce constant latent variables
        include : list
            latent variables to be replaced
        exclude : list
            latent variables to be excluded for replacements

        Returns
        -------
        node with replacements
        """
        if include is not None and exclude is not None:
            raise ValueError('Only one parameter is supported {include|exclude}, got two')
        if include is not None:
            replacements = {k: v for k, v
                            in self.flat_view.replacements.items() if k in include}
        elif exclude is not None:
            replacements = {k: v for k, v
                            in self.flat_view.replacements.items() if k not in exclude}
        else:
            replacements = self.flat_view.replacements
        node = theano.clone(node, replacements, strict=False)
        posterior = self.random(no_rand=deterministic)
        return theano.clone(node, {self.input: posterior})

    def scale_grad(self, inp):
        """
        Identity with scaling gradient

        References
        ----------
        Sticking the Landing: A Simple Reduced-Variance Gradient for ADVI
        approximateinference.org/accepted/RoederEtAl2016.pdf
        """
        return GradScale(self.cost_part_grad_scale)(inp)

    def to_flat_input(self, node):
        """
        Replaces vars with flattened view stored in self.input
        """
        return theano.clone(node, self.flat_view.replacements, strict=False)

    @property
    def params(self):
        return cast_to_list(self.shared_params)

    def initial(self, samples, no_rand=False, l=None):
        """
        Initial distribution for constructing posterior

        Parameters
        ----------
        samples : int - number of samples
        no_rand : bool - return zeros if True
        l : length of sample, defaults to latent space dim

        Returns
        -------
        Tensor
            sampled latent space shape == size + latent_dim
        """
        if l is None:
            l = self.total_size
        if samples is None:
            shape = tt.as_tensor(l)[None]
        else:
            shape = tt.stack([tt.as_tensor(samples),
                              tt.as_tensor(l)])
        if isinstance(no_rand, bool):
            no_rand = int(no_rand)
        zeros = tt.as_tensor(no_rand)
        sample = getattr(tt_rng(), self.initial_dist_name)(shape)
        space = ifelse(zeros,
                       tt.ones_like(sample) * self.initial_dist_map,
                       sample)
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

    def random_global(self, size=None, no_rand=False):
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
        posterior space
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
        elif self.local_vars:
            return self.random_local(size, no_rand)
        elif self.global_vars:
            return self.random_global(size, no_rand)
        else:
            raise ValueError('No FreeVARs in model')

    @property
    @memoize
    @change_flags(compute_test_value='off')
    def random_fn(self):
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
        trace.setup(draws=draws, chain=0)
        for point in points():
            trace.record(point)
        return pm.sampling.MultiTrace([trace])

    def log_q_W_local(self, z):
        """
        log_q_W samples over q for local vars
        Gradient wrt mu, rho in density parametrization
        is set to zero to lower variance of ELBO
        """
        if not self.local_vars:
            return tt.constant(0)
        logp = []
        for var in self.local_vars:
            mu = self.known[var][0].ravel()
            rho = self.known[var][1].ravel()
            mu = self.scale_grad(mu)
            rho = self.scale_grad(rho)
            x = self.view(z, var.name, reshape=False)
            q = log_normal(x, mu, rho=rho)
            logp.append(q.sum() * var.scaling)
        return tt.sum(logp)

    def log_q_W_global(self, z):
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

        slc = self._view[name].slc
        _, _, _shape, dtype = self._view[name]
        if space.ndim == 2:
            view = space[:, slc]
        elif space.ndim < 2:
            view = space[slc]
        else:
            raise ValueError('Space should have no more than 2 dims, got %d' % space.ndim)
        if reshape:
            if len(_shape) > 0:
                if isinstance(space, tt.TensorVariable):
                    shape = tt.concatenate([space.shape[:-1],
                                            tt.as_tensor(_shape)])
                else:
                    shape = np.concatenate([space.shape[:-1],
                                            _shape])

            else:
                shape = space.shape[:-1]
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
