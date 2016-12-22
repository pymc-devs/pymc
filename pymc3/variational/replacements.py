import collections
import theano.tensor as tt
import theano
import pymc3 as pm
from ..theanof import tt_rng
from ..blocking import ArrayOrdering
from ..distributions.dist_math import rho2sd, log_normal3
from ..math import flatten_list
from ..model import modelcontext
import numpy as np


FlatView = collections.namedtuple('FlatView', 'input, replacements')


class BaseReplacement(object):
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

    @property
    def constant_shared_params(self):
        """Constant view on shared params
        """
        return collections.OrderedDict(
                [(name, theano.gradient.zero_grad(shared))
                    for name, shared in self.shared_params.items()])

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
        if not zeros:
            return tt_rng().normal(shape)
        else:
            return tt.zeros(shape)

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
        mu, rho = self.__local_mu_rho()
        x = initial * rho2sd(rho) + mu
        return x

    def __local_mu_rho(self):
        mu = []
        rho = []
        for var in self.local_vars:
            mu.append(self.known[var][0].ravel())
            rho.append(self.known[var][1].ravel())
        mu = tt.concatenate(mu)
        rho = tt.concatenate(rho)
        return mu, rho

    def __constant_local_mu_rho(self):
        mu, rho = self.__local_mu_rho()
        return theano.gradient.zero_grad(mu), theano.gradient.zero_grad(rho)

    def to_flat_input(self, node):
        """Replaces vars with flattened view stored in self.input
        """
        return theano.clone(node, self.flat_view.replacements, strict=False)

    def log_q_W_local(self, posterior):
        """log_q_W samples over q for local vars
        """
        if not self.local_vars:
            return 0
        mu, rho = self.__constant_local_mu_rho()
        samples = tt.sum(log_normal3(posterior, mu, rho), axis=1)
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
            list(map(self.weighted_likelihood, self.model.observed_RVs))
        )
        _log_p_D_ = self.to_flat_input(_log_p_D_)
        samples = self.sample_over_space(posterior, _log_p_D_)
        return samples

    def weighted_likelihood(self, var):
        """Weight likelihood according to given population size
        """
        tot = self.population.get(
            var, self.population.get(var.name))
        logpt = tt.sum(var.logpt)
        if tot is not None:
            tot = tt.as_tensor(tot)
            logpt *= tot
            logpt /= var.size
        return logpt

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

    def apply_replacements(self, node, deterministic=False):
        """Replace variables in graph with variational approximation

        Parameters
        ----------
        node : node for replacements
        deterministic : whether to use zeros as initial distribution
            if True - median point will be sampled

        Returns
        -------
        node with replacements
        """
        posterior = self.posterior(zeros=deterministic)[0]
        node = self.to_flat_input(node)
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


class MeanField(BaseReplacement):
    def create_shared_params(self):
        return {'mu': theano.shared(self.input.tag.test_value[self.global_slc]),
                'rho': theano.shared(np.ones((self.global_size,)))}

    def posterior_global(self, initial):
        sd = rho2sd(self.shared_params['rho'])
        mu = self.shared_params['mu']
        return sd * initial + mu

    def log_q_W_global(self, posterior):
        mu = self.constant_shared_params['mu']
        rho = self.constant_shared_params['rho']
        samples = tt.sum(log_normal3(posterior, mu, rho), axis=1)
        return samples
