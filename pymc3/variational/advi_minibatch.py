from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as tt
from theano.sandbox.rng_mrg import MRG_RandomStreams

from pymc3 import modelcontext, ArrayOrdering, DictToArrayBijection
from pymc3.theanof import reshape_t, inputvars
from .advi import check_discrete_rvs, ADVIFit, adagrad_optimizer, gen_random_state

__all__ = ['advi_minibatch']


class Encoder(object):
    """Encode vector into latent representation."""
    def encode(self):
        """Returns variational mean and std vectors."""
        pass

    def get_params(self):
        """Returns list of parameters (shared variables) of the encoder."""
        pass


def _value_error(cond, str):
    if not cond:
        raise ValueError(str)


def _check_minibatches(minibatch_tensors, minibatches):
    _value_error(isinstance(minibatch_tensors, list),
                 'minibatch_tensors must be a list.')

    _value_error(hasattr(minibatches, "__iter__"),
                 'minibatches must be an iterator.')


def _get_rvss(
        minibatch_RVs, local_RVs, observed_RVs, minibatch_tensors, total_size):
    """Returns local_RVs and observed_RVs.

    This function is used for backward compatibility of how input arguments are
    given.
    """
    if minibatch_RVs is not None:
        _value_error(isinstance(minibatch_RVs, list),
                     'minibatch_RVs must be a list.')

        _value_error((local_RVs is None) and (observed_RVs is None),
                     'When minibatch_RVs is given, local_RVs and ' +
                     'observed_RVs must be None.')

        s = np.float32(total_size) / minibatch_tensors[0].shape[0]
        local_RVs = OrderedDict()
        observed_RVs = OrderedDict([(v, s) for v in minibatch_RVs])

    else:
        _value_error((isinstance(local_RVs, OrderedDict) and
                      isinstance(observed_RVs, OrderedDict)),
                     'local_RVs and observed_RVs must be OrderedDict.')

    return local_RVs, observed_RVs


def _init_uw_global_shared(start, global_RVs, global_order):
    start = {v.name: start[v.name] for v in global_RVs}
    bij = DictToArrayBijection(global_order, start)
    u_start = bij.map(start)
    w_start = np.zeros_like(u_start)
    uw_start = np.concatenate([u_start, w_start])
    uw_global_shared = theano.shared(uw_start, 'uw_global_shared')

    return uw_global_shared, bij


def _join_global_RVs(global_RVs, global_order):
    joined_global = tt.concatenate([v.ravel() for v in global_RVs])
    uw_global = tt.dvector('uw_global')
    uw_global.tag.test_value = np.concatenate([joined_global.tag.test_value,
                                               joined_global.tag.test_value])

    inarray_global = joined_global.type('inarray_global')
    inarray_global.tag.test_value = joined_global.tag.test_value

    get_var = {var.name: var for var in global_RVs}
    replace_global = {
        get_var[var]: reshape_t(inarray_global[slc], shp).astype(dtyp)
        for var, slc, shp, dtyp in global_order.vmap
    }

    return inarray_global, uw_global, replace_global


def _join_local_RVs(local_RVs, local_order):
    if len(local_RVs) == 0:
        inarray_local = None
        uw_local = None
        replace_local = {}
    else:
        joined_local = tt.concatenate([v.ravel() for v in local_RVs])
        uw_local = tt.dvector('uw_local')
        uw_local.tag.test_value = np.concatenate([joined_local.tag.test_value,
                                                  joined_local.tag.test_value])

        inarray_local = joined_local.type('inarray_local')
        inarray_local.tag.test_value = joined_local.tag.test_value

        get_var = {var.name: var for var in local_RVs}
        replace_local = {
            get_var[var]: reshape_t(inarray_local[slc], shp).astype(dtyp)
            for var, slc, shp, dtyp in local_order.vmap
        }

    return inarray_local, uw_local, replace_local


def _make_logpt(global_RVs, local_RVs, observed_RVs, model):
    """Return expression of log probability.
    """
    # Scale log probability for mini-batches
    factors = [s * v.logpt for v, s in observed_RVs.items()] + \
              [v.logpt for v in global_RVs] + model.potentials
    if local_RVs is not None:
        factors += [s * v.logpt for v, (_, s) in local_RVs.items()]
    logpt = tt.add(*map(tt.sum, factors))

    return logpt


def _elbo_t(logp, uw_g, uw_l, inarray_g, inarray_l, n_mcsamples, random_seed):
    """Return expression of approximate ELBO based on Monte Carlo sampling.
    """
    if random_seed is None:
        r = MRG_RandomStreams(gen_random_state())
    else:
        r = MRG_RandomStreams(seed=random_seed)

    if uw_l is not None:
        l_g = (uw_g.size / 2).astype('int64')
        u_g = uw_g[:l_g]
        w_g = uw_g[l_g:]
        l_l = (uw_l.size / 2).astype('int64')
        u_l = uw_l[:l_l]
        w_l = uw_l[l_l:]

        def logp_(z_g, z_l):
            return theano.clone(logp, {inarray_g: z_g, inarray_l: z_l}, strict=False)
        if n_mcsamples == 1:
            n_g = r.normal(size=inarray_g.tag.test_value.shape)
            z_g = n_g * tt.exp(w_g) + u_g
            n_l = r.normal(size=inarray_l.tag.test_value.shape)
            z_l = n_l * tt.exp(w_l) + u_l
            elbo = logp_(z_g, z_l) + \
                tt.sum(w_g) + 0.5 * l_g * (1 + np.log(2.0 * np.pi)) + \
                tt.sum(w_l) + 0.5 * l_l * (1 + np.log(2.0 * np.pi))
        else:
            ns_g = r.normal(size=inarray_g.tag.test_value.shape)
            zs_g = ns_g * tt.exp(w_g) + u_g
            ns_l = r.normal(size=inarray_l.tag.test_value.shape)
            zs_l = ns_l * tt.exp(w_l) + u_l
            logps, _ = theano.scan(fn=lambda z_g, z_l: logp_(z_g, z_l),
                                   outputs_info=None,
                                   sequences=zip(zs_g, zs_l))
            elbo = tt.mean(logps) + \
                tt.sum(w_g) + 0.5 * l_g * (1 + np.log(2.0 * np.pi)) + \
                tt.sum(w_l) + 0.5 * l_l * (1 + np.log(2.0 * np.pi))
    else:
        l_g = (uw_g.size / 2).astype('int64')
        u_g = uw_g[:l_g]
        w_g = uw_g[l_g:]

        def logp_(z_g):
            return theano.clone(logp, {inarray_g: z_g}, strict=False)

        if n_mcsamples == 1:
            n_g = r.normal(size=inarray_g.tag.test_value.shape)
            z_g = n_g * tt.exp(w_g) + u_g
            elbo = logp_(z_g) + \
                tt.sum(w_g) + 0.5 * l_g * (1 + np.log(2.0 * np.pi))
        else:
            n_g = r.normal(size=(n_mcsamples, u_g.tag.test_value.shape[0]))
            zs_g = n_g * tt.exp(w_g) + u_g
            logps, _ = theano.scan(fn=lambda q: logp_(q),
                                   outputs_info=None,
                                   sequences=[zs_g])
            elbo = tt.mean(logps) + \
                tt.sum(w_g) + 0.5 * l_g * (1 + np.log(2.0 * np.pi))

    return elbo


def advi_minibatch(vars=None, start=None, model=None, n=5000, n_mcsamples=1,
                   minibatch_RVs=None, minibatch_tensors=None, minibatches=None,
                   local_RVs=None, observed_RVs=None, encoder_params=[],
                   total_size=None, optimizer=None, learning_rate=.001,
                   epsilon=.1, random_seed=None, verbose=1):
    """Perform mini-batch ADVI.

    This function implements a mini-batch ADVI with the meanfield 
    approximation. Autoencoding variational inference is also supported. 

    The log probability terms for mini-batches, corresponding to RVs in 
    minibatch_RVs, are scaled to (total_size) / (the number of samples in each 
    mini-batch), where total_size is an argument for the total data size. 

    minibatch_tensors is a list of tensors (can be shared variables) to which 
    mini-batch samples are set during the optimization. In most cases, these 
    tensors are observations for RVs in the model. 

    local_RVs and observed_RVs are used for autoencoding variational Bayes. 
    Both of these RVs are associated with each of given samples. 
    The difference is that local_RVs are unkown and their posterior 
    distributions are approximated. 

    local_RVs are Ordered dict, whose keys and values are RVs and a tuple of 
    two objects. The first is the theano expression of variational parameters 
    (mean and log of std) of the approximate posterior, which are encoded from 
    given samples by an arbitrary deterministic function, e.g., MLP. The other 
    one is a scaling constant to be multiplied to the log probability term 
    corresponding to the RV. 

    observed_RVs are also Ordered dict with RVs as the keys, but whose values 
    are only the scaling constant as in local_RVs. In this case, total_size is 
    ignored. 

    If local_RVs is None (thus not using autoencoder), the following two 
    settings are equivalent: 

    - observed_RVs=OrderedDict([(rv, total_size / minibatch_size)])
    - minibatch_RVs=[rv], total_size=total_size

    where minibatch_size is minibatch_tensors[0].shape[0]. 

    The variational parameters and the parameters of the autoencoder are 
    simultaneously optimized with given optimizer, which is a function that 
    returns a dictionary of parameter updates as provided to Theano function. 
    See the docstring of pymc3.variational.advi(). 

    Parameters
    ----------
    vars : object
        List of random variables. If None, variational posteriors (normal 
        distribution) are fit for all RVs in the given model. 
    start : Dict or None
        Initial values of parameters (variational means).
    model : Model
        Probabilistic model.
    n : int
        Number of interations updating parameters.
    n_mcsamples : int
        Number of Monte Carlo samples to approximate ELBO.
    minibatch_RVs : list of ObservedRVs
        Random variables in the model for which mini-batch tensors are set. 
        When this argument is given, both of arguments local_RVs and global_RVs 
        must be None. 
    minibatch_tensors : list of (tensors or shared variables)
        Tensors used to create ObservedRVs in minibatch_RVs.
    minibatches : generator of list
        Generates a set of minibatches when calling next().
        The length of the returned list must be the same with the number of
        random variables in `minibatch_tensors`.
    total_size : int
        Total size of training samples. This is used to appropriately scale the 
        log likelihood terms corresponding to mini-batches in ELBO. 
    local_RVs : Ordered dict
        Include encoded variational parameters and a scaling constant for 
        the corresponding RV. See the above description. 
    observed_RVs : Ordered dict
        Include a scaling constant for the corresponding RV. See the above 
        description
    encoder_params : list of theano shared variables
        Parameters of encoder. 
    optimizer : (loss, tensor) -> dict or OrderedDict
        A function that returns parameter updates given loss and parameter
        tensor. If :code:`None` (default), a default Adagrad optimizer is
        used with parameters :code:`learning_rate` and :code:`epsilon` below.
    learning_rate: float
        Base learning rate for adagrad. This parameter is ignored when
        an optimizer is given.
    epsilon : float
        Offset in denominator of the scale of learning rate in Adagrad.
        This parameter is ignored when an optimizer is given.
    random_seed : int
        Seed to initialize random state.

    Returns
    -------
    ADVIFit
        Named tuple, which includes 'means', 'stds', and 'elbo_vals'.
    """
    theano.config.compute_test_value = 'ignore'

    model = modelcontext(model)
    vars = inputvars(vars if vars is not None else model.vars)
    start = start if start is not None else model.test_point
    check_discrete_rvs(vars)
    _check_minibatches(minibatch_tensors, minibatches)

    # Prepare optimizer
    if optimizer is None:
        optimizer = adagrad_optimizer(learning_rate, epsilon)

    # For backward compatibility in how input arguments are given
    local_RVs, observed_RVs = _get_rvss(minibatch_RVs, local_RVs, observed_RVs,
                                        minibatch_tensors, total_size)

    # Replace local_RVs with transformed variables
    ds = model.deterministics

    def get_transformed(v):
        if v in ds:
            return v.transformed
        return v
    local_RVs = OrderedDict(
        [(get_transformed(v), (uw, s)) for v, (uw, s) in local_RVs.items()]
    )

    # Get global variables
    global_RVs = list(set(vars) - set(list(local_RVs) + list(observed_RVs)))

    # Ordering for concatenation of random variables
    global_order = ArrayOrdering([v for v in global_RVs])
    local_order = ArrayOrdering([v for v in local_RVs])

    # ELBO wrt variational parameters
    inarray_g, uw_g, replace_g = _join_global_RVs(global_RVs, global_order)
    inarray_l, uw_l, replace_l = _join_local_RVs(local_RVs, local_order)
    logpt = _make_logpt(global_RVs, local_RVs, observed_RVs, model)
    replace = replace_g
    if replace_l is not None:
        replace.update(replace_l)
    logp = theano.clone(logpt, replace, strict=False)
    elbo = _elbo_t(logp, uw_g, uw_l, inarray_g, inarray_l,
                   n_mcsamples, random_seed)
    del logpt

    # Variational parameters for global RVs
    uw_global_shared, bij = _init_uw_global_shared(start, global_RVs,
                                                   global_order)

    # Variational parameters for local RVs, encoded from samples in
    # mini-batches
    if 0 < len(local_RVs):
        uws = [uw for _, (uw, _) in local_RVs.items()]
        uw_local_encoded = tt.concatenate([uw[0].ravel() for uw in uws] +
                                          [uw[1].ravel() for uw in uws])

    # Replace tensors in ELBO
    updates = {uw_g: uw_global_shared, uw_l: uw_local_encoded} \
        if 0 < len(local_RVs) else \
              {uw_g: uw_global_shared}
    elbo = theano.clone(elbo, updates, strict=False)

    # Replace input shared variables with tensors
    def is_shared(t):
        return isinstance(t, theano.compile.sharedvalue.SharedVariable)
    tensors = [(t.type() if is_shared(t) else t) for t in minibatch_tensors]
    updates = OrderedDict(
        {t: t_ for t, t_ in zip(minibatch_tensors, tensors) if is_shared(t)}
    )
    elbo = theano.clone(elbo, updates, strict=False)

    # Create parameter update function used in the training loop
    params = [uw_global_shared] + encoder_params
    updates = OrderedDict()
    for param in params:
        # g = tt.grad(elbo, wrt=param)
        # updates.update(adagrad(g, param, learning_rate, epsilon, n=10))
        updates.update(optimizer(loss=-1 * elbo, param=param))
    f = theano.function(tensors, elbo, updates=updates)

    # Optimization loop
    elbos = np.empty(n)
    for i in range(n):
        e = f(*next(minibatches))
        elbos[i] = e
        if verbose and not i % (n // 10):
            if not i:
                print('Iteration {0} [{1}%]: ELBO = {2}'.format(
                    i, 100 * i // n, e.round(2)))
            else:
                avg_elbo = elbos[i - n // 10:i].mean()
                print('Iteration {0} [{1}%]: Average ELBO = {2}'.format(
                    i, 100 * i // n, avg_elbo.round(2)))

    if verbose:
        print('Finished [100%]: ELBO = {}'.format(elbos[-1].round(2)))

    l = int(uw_global_shared.get_value(borrow=True).size / 2)

    u = bij.rmap(uw_global_shared.get_value(borrow=True)[:l])
    w = bij.rmap(uw_global_shared.get_value(borrow=True)[l:])
    # w is in log space
    for var in w.keys():
        w[var] = np.exp(w[var])
    return ADVIFit(u, w, elbos)
