import collections
import theano.tensor as tt
from theano.ifelse import ifelse
import theano
import pymc3 as pm
from ..theanof import tt_rng, memoize, change_flags
from ..blocking import ArrayOrdering
from ..distributions.dist_math import rho2sd, log_normal3
from ..math import flatten_list
from ..model import modelcontext
import numpy as np

# helper class
FlatView = collections.namedtuple('FlatView', 'input, replacements')
# shortcut for zero grad
Z = theano.gradient.zero_grad


class BaseApproximation(object):
    """Base class for variational methods
    It uses the idea that we can map standard normal distribution
    to the approximate posterior distribution. Thus there are three blank
    places in the implementation:
        1) creating shared parameters `create_shared_params()`
        2) mapping initial distribution to posterior `posterior{_{global|local}}(initial)`
        3) calculating log_q_W for approximation `log_q_W{_{global|local}}(posterior)`

    Then ELBO can be calculated for given approximation
        log_p_D(posterior) - KL_q_p_W(posterior)
    """
    def __init__(self, model=None, population=None, known=None):
        model = modelcontext(model)
        self.check_model(model)
        if known is None:
            known = dict()
        if population is None:
            population = dict()
        self.population = population
        self.model = model
        self.known = known
        self.local_vars = [v for v in model.vars if v in known]
        self.global_vars = [v for v in model.vars if v not in known]
        self.order = ArrayOrdering(self.local_vars + self.global_vars)
        inputvar = tt.vector('flat_view')
        inputvar.tag.test_value = flatten_list(self.local_vars + self.global_vars).tag.test_value
        replacements = {self.model.named_vars[name]: inputvar[slc].reshape(shape).astype(dtype)
                        for name, slc, shape, dtype in self.order.vmap}
        self.flat_view = FlatView(inputvar, replacements)
        self.view = {vm.var: vm for vm in self.order.vmap}
        self.shared_params = self.create_shared_params()

    def set_params(self, params):
        self.shared_params.update(params)

    @property
    def params(self):
        return list(self.shared_params.values())

    @staticmethod
    def check_model(model):
        """Checks that model is valid for variational inference
        """
        vars_ = [var for var in model.vars if not isinstance(var, pm.model.ObservedRV)]
        if any([var.dtype in pm.discrete_types for var in vars_]):
            raise ValueError('Model should not include discrete RVs')

    @property
    def input(self):
        """Shortcut to flattened input
        """
        return self.flat_view.input

    def create_shared_params(self):
        """Any stuff you need will be here

        Returns
        -------
        dict : shared params
        """
        raise NotImplementedError

    def initial(self, samples, subset, zeros=False):
        """
        Parameters
        ----------
        samples : int - number of samples
        subset : determines what space is used can be {all|local|global}
        zeros : bool - return zeros if True

        Returns
        -------
        Tensor
            sampled latent space shape(samples, size)
        """
        assert subset in {'all', 'local', 'global'}
        if subset == 'all':
            size = self.total_size
        elif subset == 'global':
            size = self.global_size
        else:
            size = self.local_size
        shape = tt.stack([tt.as_tensor(samples), tt.as_tensor(size)])
        if isinstance(zeros, bool):
            zeros = int(zeros)
        zeros = tt.as_tensor(zeros)
        space = ifelse(zeros,
                       tt.zeros(shape),
                       tt_rng().normal(shape))
        return space

    def posterior(self, samples=1, zeros=False):
        """Transforms initial latent space to posterior distribution

        Parameters
        ----------
        samples : number of samples
        zeros : set initial distribution to zeros

        Returns
        -------
        Tensor
            posterior space
        """
        if self.local_vars and self.global_vars:
            return tt.concatenate([
                self.posterior_local(self.initial(samples, 'local', zeros)),
                self.posterior_global(self.initial(samples, 'global', zeros))
                ], axis=1)
        elif self.local_vars:
            return self.posterior_local(self.initial(samples, 'local', zeros))
        elif self.global_vars:
            return self.posterior_global(self.initial(samples, 'global', zeros))
        else:
            raise ValueError('No FreeVARs in model')

    def sample_over_space(self, space, node):
        """
        Parameters
        ----------
        space : space to sample over
        node : node that has flattened input

        Returns
        -------
        samples
        """
        def replace_node(post):
            return theano.clone(node, {self.input: post})
        samples, _ = theano.scan(replace_node, space, n_steps=space.shape[0])
        return samples

    def view_from(self, space, name, subset='all'):
        """
        Parameters
        ----------
        space : space to take view of variable from
        name : name of variable
        subset : determines what space is used can be {all|local|global}

        Returns
        -------
        variable
            shape == (samples,) + variable.shape
        """
        assert subset in {'all', 'local', 'global'}
        if subset in {'all', 'local'}:
            slc = self.view[name].slc
        else:
            s = self.view[name].slc.start
            e = self.view[name].slc.stop
            slc = slice(s - self.local_size, e - self.local_size)
        _, _, shape, dtype = self.view[name]
        return space[:, slc].reshape((space.shape[0],) + shape).astype(dtype)

    def posterior_global(self, initial):
        """Implements posterior distribution from initial latent space

        Parameters
        ----------
        initial : initial latent space shape(samples, size)

        Returns
        -------
        global posterior space
        """
        raise NotImplementedError

    def posterior_local(self, initial):
        """Implements posterior distribution from initial latent space

        Parameters
        ----------
        initial : initial latent space shape(samples, size)

        Returns
        -------
        local posterior space
        """
        if not self.local_vars:
            return initial
        mu, rho = self._local_mu_rho()
        x = initial * rho2sd(rho) + mu
        return x

    def _local_mu_rho(self):
        mu = []
        rho = []
        for var in self.local_vars:
            mu.append(self.known[var][0].ravel())
            rho.append(self.known[var][1].ravel())
        mu = tt.concatenate(mu)
        rho = tt.concatenate(rho)
        return mu, rho

    def to_flat_input(self, node):
        """Replaces vars with flattened view stored in self.input
        """
        return theano.clone(node, self.flat_view.replacements, strict=False)

    def log_q_W_local(self, posterior):
        """log_q_W samples over q for local vars

        Gradient wrt mu, rho in density parametrization
        is set to zero to lower variance of ELBO
        """
        if not self.local_vars:
            return 0
        logp = []
        for var in self.local_vars:
            mu = self.known[var][0].ravel()
            rho = self.known[var][1].ravel()
            q = log_normal3(self.view_from(posterior, var.name, 'local'), Z(mu), Z(rho))
            logp.append(self.weighted_logp(var, q))
        samples = tt.sum(tt.concatenate(logp, axis=1), axis=1)
        return samples

    def log_q_W_global(self, posterior):
        """log_q_W samples over q for global vars
        """
        raise NotImplementedError

    def log_q_W(self, posterior):
        """log_q_W samples over q
        """
        return self.log_q_W_global(posterior) + self.log_q_W_local(posterior)

    def log_p_D(self, posterior):
        """log_p_D samples over q
        """
        _log_p_D_ = tt.sum(
            list(map(self.weighted_logp, self.model.observed_RVs))
        )
        _log_p_D_ = self.to_flat_input(_log_p_D_)
        samples = self.sample_over_space(posterior, _log_p_D_)
        return samples

    def weighted_logp(self, var, logp=None):
        """Weight logp according to given population size
        """
        tot = self.population.get(var)
        if logp is None:
            logp = tt.sum(var.logpt)
        if tot is not None:
            tot = tt.as_tensor(tot)
            logp *= tot / var.shape[0]
        return logp

    def KL_q_p_W(self, posterior):
        """KL(q||p) samples over q
        """
        return self.log_q_W(posterior) - self.log_p_W(posterior)

    def log_p_W(self, posterior):
        """log_p_W samples over q
        """
        _log_p_W_ = self.model.varlogpt + tt.sum(self.model.potentials)
        _log_p_W_ = self.to_flat_input(_log_p_W_)
        samples = self.sample_over_space(posterior, _log_p_W_)
        return samples

    def apply_replacements(self, node, deterministic=False, include=None, exclude=None):
        """Replace variables in graph with variational approximation. By default, replaces all variables

        Parameters
        ----------
        node : node for replacements
        deterministic : whether to use zeros as initial distribution
            if True - zero initial point will produce constant latent variables
        include : list - latent variables to be replaced
        exclude : list - latent variables to be excluded for replacements

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
        posterior = self.posterior(zeros=deterministic)[0]
        return theano.clone(node, {self.input: posterior})

    def elbo(self, samples=1, pi=1):
        """Output of this function should be used for fitting a model

        Parameters
        ----------
        samples : Tensor - number of samples
        pi : Tensor - weight of variational part

        Returns
        -------
        ELBO samples
        """
        samples = tt.as_tensor(samples)
        pi = tt.as_tensor(pi)
        posterior = self.posterior(samples)
        elbo = self.log_p_D(posterior) - pi * self.KL_q_p_W(posterior)
        return elbo

    @property
    @memoize
    @change_flags(compute_test_value='off')
    def posterior_fn(self):
        In = theano.In
        samples = tt.iscalar('n_samples')
        zeros = tt.bscalar('zeros')
        posterior = self.posterior(samples, zeros)
        fn = theano.function([In(samples, 'samples', 1, allow_downcast=True),
                              In(zeros, 'zeros', 0, allow_downcast=True)],
                             posterior)

        def inner(samples=1, zeros=False):
            return fn(samples, int(zeros))
        return inner

    def sample_vp(self, draws=1, zeros=False, hide_transformed=False):
        if hide_transformed:
            vars_sampled = [v_ for v_ in self.model.unobserved_RVs
                            if not str(v_).endswith('_')]
        else:
            vars_sampled = [v_ for v_ in self.model.unobserved_RVs]
        posterior = self.posterior_fn(draws, zeros)
        names = [var.name for var in self.local_vars + self.global_vars]
        samples = {name: self.view_from(posterior, name)
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

    @property
    @memoize
    def posterior_to_point_fn(self):
        names_vars = [var.name for var in self.local_vars + self.global_vars]
        names_point = names_vars + [var.name for var in self.model.deterministics]
        point_fn = self.model.fastfn(self.local_vars + self.global_vars + self.model.deterministics)

        def inner(posterior):
            if posterior.shape[0] > 1:
                raise ValueError('Posterior should have one sample')
            point = {name: self.view_from(posterior, name)[0]
                     for name in names_vars}
            return dict(zip(names_point, point_fn(point)))
        return inner

    @property
    def median(self):
        posterior = self.posterior_fn(samples=1, zeros=True)
        return self.posterior_to_point_fn(posterior)

    def random(self):
        posterior = self.posterior_fn(samples=1, zeros=False)
        return self.posterior_to_point_fn(posterior)

    @property
    def local_vmap(self):
        return self.order.vmap[:len(self.local_vars)]

    @property
    def global_vmap(self):
        return self.order.vmap[len(self.local_vars):]

    @property
    def local_size(self):
        size = sum([0] + [v.dsize for v in self.local_vars])
        return size

    @property
    def global_size(self):
        return self.total_size - self.local_size

    @property
    def total_size(self):
        return self.order.dimensions

    @property
    def local_slc(self):
        return slice(0, self.local_size)

    @property
    def global_slc(self):
        return slice(self.local_size, self.total_size)


class Advi(BaseApproximation):
    def create_shared_params(self):
        return {'mu': theano.shared(self.input.tag.test_value[self.global_slc]),
                'rho': theano.shared(np.zeros((self.global_size,)))}

    def posterior_global(self, initial):
        sd = rho2sd(self.shared_params['rho'])
        mu = self.shared_params['mu']
        return sd * initial + mu

    def log_q_W_global(self, posterior):
        """log_q_W samples over q for global vars

        Gradient wrt mu, rho in density parametrization
        is set to zero to lower variance of ELBO"""
        mu = self.shared_params['mu']
        rho = self.shared_params['rho']
        samples = tt.sum(log_normal3(posterior, Z(mu), Z(rho)), axis=1)
        return samples
