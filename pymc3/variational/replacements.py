import collections
import theano.tensor as tt
import theano
from ..theanof import tt_rng
from ..blocking import ArrayOrdering
from ..distributions.dist_math import rho2sd, log_normal3
from ..math import flatten_list
import numpy as np


FlatView = collections.namedtuple('FlatView', 'input, replacements')


class BaseReplacement(object):
    def __init__(self, model, population=None, known=None):
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
    def input(self):
        return self.flat_view.input

    def create_shared_params(self):
        """Any stuff you need will be here
        Returns
        -------
        dict : shared params
        """
        raise NotImplementedError

    def initial(self, samples=1, zeros=False):
        """
        Parameters
        ----------
        samples : int - number of samples
        zeros : bool - return zeros if True

        Returns
        -------
        matrix/vector
            sampled latent space
        """
        if not zeros:
            return tt_rng().normal((samples, self.total_size))
        else:
            return tt.zeros((samples, self.total_size))

    def view_from(self, space, name, subset='all'):
        assert subset in {'all', 'local', 'global'}
        if subset in {'all', 'local'}:
            slc = self.view[name].slc
        else:
            s = self.view[name].slc.start
            e = self.view[name].slc.stop
            slc = slice(s - self.global_size, e - self.global_size)
        return space[:, slc]

    def posterior(self, initial):
        return tt.concatenate([
            self.posterior_local(initial[:, self.local_slc]),
            self.posterior_global(initial[:, self.global_slc])
            ], axis=1)

    def posterior_global(self, initial):
        raise NotImplementedError

    def __local_mu_rho(self):
        mu = []
        rho = []
        for var in self.local_vars:
            mu.append(self.known[var][0].ravel())
            rho.append(self.known[var][1].ravel())
        mu = tt.concatenate(mu)
        rho = tt.concatenate(rho)
        return mu, rho

    def posterior_local(self, initial):
        if not self.known:
            return initial
        mu, rho = self.__local_mu_rho()
        x = initial * rho2sd(rho) + mu
        return x

    def to_flat_input(self, node):
        return theano.clone(node, self.flat_view.replacements, strict=False)

    @property
    def local_vmap(self):
        return self.order.vmap[:len(self.local_vars)]

    @property
    def global_vmap(self):
        return self.order.vmap[len(self.local_vars):]

    @property
    def local_size(self):
        size = sum([0] + [v.size for v in self.local_vars])
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

    def log_q_W_local(self, posterior):
        if not self.known:
            return 0
        mu, rho = self.__local_mu_rho()
        logp = tt.sum(log_normal3(posterior, mu, rho), axis=1).mean(axis=0)
        return logp

    def log_q_W_global(self, posterior):
        raise NotImplementedError

    def log_q_W(self, posterior):
        return self.log_q_W_global(posterior) + self.log_q_W_local(posterior)

    def log_p_D(self, posterior):
        _log_p_D_ = tt.add(
            *map(self.weighted_likelihood, self.model.observed_RVs)
        )
        _log_p_D_ = self.to_flat_input(_log_p_D_)

        def replace_log_p_D(post):
            return theano.clone(_log_p_D_, {self.input: post})

        results, _ = theano.map(replace_log_p_D, posterior)
        return results.mean()

    def weighted_likelihood(self, var):
        tot = self.population.get(
            var, self.population.get(var.name))
        logpt = tt.sum(var.logpt)
        if tot is not None:
            tot = tt.as_tensor(tot)
            logpt *= tot
            logpt /= var.size
        return logpt

    def log_p_W(self, posterior):
        _log_p_W_ = self.model.varlogpt + tt.sum(self.model.potentials)
        _log_p_W_ = self.to_flat_input(_log_p_W_)

        def replace_log_p_W(post):
            return theano.clone(_log_p_W_, {self.input: post})

        results, _ = theano.map(replace_log_p_W, posterior)
        return results.mean()

    def elbo(self, posterior):
        return self.log_p_D(posterior) + self.log_p_W(posterior) - self.log_q_W(posterior)

    def apply_replacements(self, node, deterministic=False):
        node = self.to_flat_input(node)
        initial = self.initial(zeros=deterministic)
        posterior = self.posterior(initial)[0]
        return theano.clone(node, {self.input: posterior})

    def sample_elbo(self, samples=1, pi=1):
        """Output of this function should be used for fitting a model

        Parameters
        ----------
        samples : Tensor - number of samples
        pi : Tensor - weight of variational part


        Notes
        -----
        In case of samples == 1 updates are empty and can be ignored

        Returns
        -------
        elbos, updates
        """
        samples = tt.as_tensor(samples)
        pi = tt.as_tensor(pi)
        posterior = self.posterior(self.initial(samples))
        elbo = (self.log_p_D(posterior) +
                pi * (self.log_p_W(posterior) - self.log_q_W(posterior)))
        return elbo


class MeanField(BaseReplacement):
    def create_shared_params(self):
        return {'mu': theano.shared(self.input.tag.test_value[self.global_slc]),
                'rho': theano.shared(np.ones((self.global_size,)))}

    def posterior_global(self, initial):
        sd = rho2sd(self.shared_params['rho'])
        mu = self.shared_params['mu']
        return sd * initial + mu

    def log_q_W_global(self, posterior):
        mu = self.shared_params['mu']
        rho = self.shared_params['rho']
        logp = tt.sum(log_normal3(posterior, mu, rho), axis=1).mean(axis=0)
        return logp
