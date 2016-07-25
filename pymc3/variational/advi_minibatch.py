import numpy as np
from ..model import modelcontext
from ..blocking import ArrayOrdering, DictToArrayBijection
import theano
from ..theanof import reshape_t, inputvars
import theano.tensor as tt
from theano.sandbox.rng_mrg import MRG_RandomStreams
from collections import OrderedDict
from .advi import check_discrete_rvs, adagrad, ADVIFit

__all__ = ['advi_minibatch']

# Flatten list
from itertools import chain
flt = lambda l: list(chain.from_iterable(l))

class Encoder(object):
    """Encode vector into latent representation.
    """
    def encode(self):
        """Returns variational mean and std vectors. 
        """
        pass
        
    def get_params(self):
        """Returns list of parameters (shared variables) of the encoder. 
        """
        pass

def _value_error(cond, str):
    if not cond:
        raise ValueError(str)

def _check_minibatches(minibatch_tensors, minibatches):
    _value_error(isinstance(minibatch_tensors, list), 
                 'minibatch_tensors must be a list.')

    _value_error(hasattr(minibatches, "__iter__"), 
                 'minibatches must be an iterator.')

def _replace_shared_minibatch_tensors(minibatch_tensors):
    """Replace shared variables in minibatch tensors with normal tensors. 
    """
    givens = dict()
    tensors = list()

    for t in minibatch_tensors:
        if isinstance(t, theano.compile.sharedvalue.SharedVariable):
            t_ = t.type()
            tensors.append(t_)
            givens.update({t: t_})
        else:
            tensors.append(t)

    return tensors, givens

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

    get_var = {var.name : var for var in global_RVs}
    replace_global = {
        get_var[var] : reshape_t(inarray_global[slc], shp).astype(dtyp)
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

        get_var = {var.name : var for var in local_RVs}
        replace_local = {
            get_var[var] : reshape_t(inarray_local[slc], shp).astype(dtyp)
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
        factors += [s * v.logpt for v, (_ , s) in local_RVs.items()]
    logpt = tt.add(*map(tt.sum, factors))

    return logpt

def _elbo_t_new(logp, uw_g, uw_l, inarray_g, inarray_l, 
                n_mcsamples, random_seed):
    """Return expression of approximate ELBO based on Monte Carlo sampling.
    """
    r = MRG_RandomStreams(seed=random_seed)

    if uw_l is not None:
        l_g = (uw_g.size/2).astype('int64')
        u_g = uw_g[:l_g]
        w_g = uw_g[l_g:]
        l_l = (uw_l.size/2).astype('int64')
        u_l = uw_l[:l_l]
        w_l = uw_l[l_l:]
        logp_ = lambda z_g, z_l: theano.clone(
            logp, {inarray_g: z_g, inarray_l: z_l}, strict=False
        )

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
        l_g = (uw_g.size/2).astype('int64')
        u_g = uw_g[:l_g]
        w_g = uw_g[l_g:]

        logp_ = lambda z_g: theano.clone(logp, {inarray_g: z_g}, strict=False)

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
    total_size=None, scales=None, learning_rate=.001, epsilon=.1, 
    random_seed=20090425, verbose=1):
    """Run mini-batch ADVI. 

    minibatch_tensors and minibatches should be in the same order. 

    Parameters
    ----------
    vars : object
        Random variables. 
    start : Dict or None
        Initial values of parameters (variational means). 
    model : Model
        Probabilistic model. 
    n : int
        Number of interations updating parameters. 
    n_mcsamples : int
        Number of Monte Carlo samples to approximate ELBO. 
    minibatch_RVs : list of ObservedRVs
        Random variables for mini-batch. 
    minibatch_tensors : list of (tensors or shared variables)
        Tensors used to create ObservedRVs in minibatch_RVs. 
    minibatches : generator of list
        Generates a set of minibatches when calling next(). 
        The length of the returned list must be the same with the number of 
        random variables in `minibatch_tensors`. 
    total_size : int
        Total size of training samples. 
    learning_rate: float
        Adagrad base learning rate. 
    epsilon : float
        Offset in denominator of the scale of learning rate in Adagrad.  
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

    # For backward compatibility in how input arguments are given
    local_RVs, observed_RVs = _get_rvss(minibatch_RVs, local_RVs, observed_RVs, 
                                        minibatch_tensors, total_size)

    # Replace local_RVs with transformed variables
    ds = model.deterministics
    get_transformed = lambda v: v if v not in ds else v.transformed
    local_RVs = OrderedDict(
        [(get_transformed(v), (uw, s)) for v, (uw, s) in local_RVs.items()]
    )

    # Get global variables
    rvs = lambda x: [rv for rv in x]
    global_RVs = list(set(vars) - set(rvs(local_RVs) + rvs(observed_RVs)))

    # Ordering for concatenation of random variables
    global_order = ArrayOrdering([v for v in global_RVs])
    local_order = ArrayOrdering([v for v in local_RVs])

    # ELBO wrt variational parameters
    inarray_g, uw_g, replace_g = _join_global_RVs(global_RVs, global_order)
    inarray_l, uw_l, replace_l = _join_local_RVs(local_RVs, local_order)
    logpt = _make_logpt(global_RVs, local_RVs, observed_RVs, model)
    replace = replace_g
    if replace_l is not None: replace.update(replace_l)
    logp = theano.clone(logpt, replace, strict=False)
    elbo = _elbo_t_new(logp, uw_g, uw_l, inarray_g, inarray_l, 
                       n_mcsamples, random_seed)
    del logpt

    # Variational parameters for global RVs
    uw_global_shared, bij = _init_uw_global_shared(start, global_RVs, 
                                                   global_order)

    # Variational parameters for local RVs, encoded from samples in mini-batches
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
    isshared = lambda t: isinstance(t, theano.compile.sharedvalue.SharedVariable)
    tensors = [(t.type() if isshared(t) else t) for t in minibatch_tensors]
    updates = OrderedDict(
        {t: t_ for t, t_ in zip(minibatch_tensors, tensors) if isshared(t)}
    )
    elbo = theano.clone(elbo, updates, strict=False)

    # Create parameter update function used in the training loop
    params = [uw_global_shared] + encoder_params
    updates = OrderedDict()
    for param in params:
        g = tt.grad(elbo, wrt=param)
        updates.update(adagrad(g, param, learning_rate, epsilon, n=10))
    f = theano.function(tensors, elbo, updates=updates)

    # Training loop
    elbos = np.empty(n)
    for i in range(n):
        e = f(*next(minibatches))
        elbos[i] = e
        if verbose and not i % (n//10):
            if not i:
                print('Iteration {0} [{1}%]: ELBO = {2}'.format(i, 100*i//n, e.round(2)))
            else:
                avg_elbo = elbos[i-n//10:i].mean()
                print('Iteration {0} [{1}%]: Average ELBO = {2}'.format(i, 100*i//n, avg_elbo.round(2)))
    
    if verbose:
        print('Finished [100%]: ELBO = {}'.format(elbos[-1].round(2)))

    l = int(uw_global_shared.get_value(borrow=True).size / 2)

    u = bij.rmap(uw_global_shared.get_value(borrow=True)[:l])
    w = bij.rmap(uw_global_shared.get_value(borrow=True)[l:])
    # w is in log space
    for var in w.keys():
        w[var] = np.exp(w[var])
    return ADVIFit(u, w, elbos)
