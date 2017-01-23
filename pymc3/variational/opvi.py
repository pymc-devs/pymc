import collections
import theano.tensor as tt
from theano.ifelse import ifelse
import theano
import pymc3 as pm
from .advi import adagrad_optimizer
from ..theanof import tt_rng, memoize, change_flags, GradScale
from ..blocking import ArrayOrdering
from ..distributions.dist_math import rho2sd, log_normal, log_normal_mv
from ..distributions.distribution import infer_shape
from ..math import flatten_list
from ..model import modelcontext
import numpy as np

# helper class
FlatView = collections.namedtuple('FlatView', 'input, replacements')
# shortcut for zero grad
Z = theano.gradient.zero_grad


def flatten_model(model, vars=None):
    """Helper function for flattening model input"""
    if vars is None:
        vars = model.free_RVs
    order = ArrayOrdering(vars)
    inputvar = tt.vector('flat_view', dtype=theano.config.floatX)
    inputvar.tag.test_value = flatten_list(vars).tag.test_value
    replacements = {model.named_vars[name]: inputvar[slc].reshape(shape).astype(dtype)
                    for name, slc, shape, dtype in order.vmap}
    flat_view = FlatView(inputvar, replacements)
    view = {vm.var: vm for vm in order.vmap}
    return order, flat_view, view


class Operator(object):
    def __init__(self, approx):
        self.model = approx.model
        self.approx = approx

    flat_view = property(lambda self: self.approx.flat_view)
    input = property(lambda self: self.approx.flat_view.input)

    def logp(self, z):
        p = theano.clone(self.model.logpt, self.flat_view.replacements, strict=False)
        p = theano.clone(p, {self.input: z}, strict=False)
        return p

    def logq(self, z):
        return self.approx.logq(z)

    def apply(self, f):
        raise NotImplementedError

    def __call__(self, f):
        if not isinstance(f, TestFunction):
            f = TestFunction.from_function(f)
        return Operator.ObjectiveFunction(self, f)

    class ObjectiveFunction(object):
        def __init__(self, op, tf):
            self.random = lambda: op.approx.random()
            self.obj = lambda z: theano.clone(op.apply(tf), {op.input: z}, strict=False)
            self.test_params = tf.params
            self.obj_params = op.approx.params

        def __call__(self, z):
            return self.obj(z)

        def updates(self, z, obj_optimizer=None, test_optimizer=None, more_params=None):
            if more_params is None:
                more_params = []
            if obj_optimizer is None:
                obj_optimizer = adagrad_optimizer(learning_rate=.001, epsilon=.1)
            if test_optimizer is None:
                obj_optimizer = adagrad_optimizer(learning_rate=.001, epsilon=.1)
            target = self(z)
            updates = theano.OrderedUpdates()
            updates.update(obj_optimizer(target, self.obj_params + more_params))
            if self.test_params:
                updates.update(test_optimizer(-target, self.test_params))
            else:
                pass
            return updates


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
    shared_params = None

    @property
    def params(self):
        return cast_to_list(self.shared_params)

    def __call__(self, z):
        raise NotImplementedError

    @classmethod
    def from_function(cls, f):
        if not callable(f):
            raise ValueError('Need callable, got %r' % f)
        obj = TestFunction()
        obj.__call__ = f
        return obj


class Approximation(object):
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
        self.order, self.flat_view, self._view = flatten_model(
            model=self.model,
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

    def normal(self, samples, no_rand=False, l=None):
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
        space = ifelse(zeros,
                       tt.zeros(shape),
                       tt_rng().normal(shape))
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
        e = self.normal(size, no_rand, self.local_size)
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
            return 0
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


class KL(Operator):
    def apply(self, f):
        """KL divergence between posterior and approximation for input `z`
            :math:`z ~ Approximation`
        """
        z = self.input
        return self.logq(z) - self.logp(z)


class LS(Operator):
    def apply(self, f):
        z = self.input
        jacobian = theano.gradient.jacobian
        return (tt.Rop(self.logp(z), z, f(z)) + jacobian(f(z), z).sum()) ** 2


class MeanField(Approximation):
    def create_shared_params(self):
        return {'mu': theano.shared(
                    self.input.tag.test_value[self.global_slc]),
                'rho': theano.shared(
                    np.zeros((self.global_size,), dtype=theano.config.floatX))
                }

    def log_q_W_global(self, z):
        """
        log_q_W samples over q for global vars
        Gradient wrt mu, rho in density parametrization
        is set to zero to lower variance of ELBO
        """
        mu = self.shared_params['mu']
        rho = self.shared_params['rho']
        mu = self.scale_grad(mu)
        rho = self.scale_grad(rho)
        logq = tt.sum(log_normal(z[self.global_slc], mu, rho=rho))
        return logq

    def random_global(self, samples=None, no_rand=False):
        initial = self.normal(samples, no_rand, l=self.global_size)
        sd = rho2sd(self.shared_params['rho'])
        mu = self.shared_params['mu']
        return sd * initial + mu


class FullRank(Approximation):
    def create_shared_params(self):
        return {'mu': theano.shared(
                    self.input.tag.test_value[self.global_slc]),
                'L': theano.shared(
                    np.eye(self.global_size, dtype=theano.config.floatX).ravel())
                }

    def log_q_W_global(self, z):
        """
        log_q_W samples over q for global vars
        Gradient wrt mu, rho in density parametrization
        is set to zero to lower variance of ELBO
        """
        mu = self.shared_params['mu']
        L = self.shared_params['L'].reshape((self.global_size, self.global_size))
        mu = self.scale_grad(mu)
        L = self.scale_grad(L)
        return log_normal_mv(z, mu, chol=L)

    def random_global(self, samples=None, no_rand=False):
        initial = self.normal(samples, no_rand, l=self.global_size)
        L = self.shared_params['L'].reshape((self.global_size, self.global_size))
        mu = self.shared_params['mu']
        return initial.dot(L) + mu
