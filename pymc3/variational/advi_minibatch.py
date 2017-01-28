from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as tt
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.configparser import change_flags
import tqdm

import pymc3 as pm
from pymc3.theanof import reshape_t, inputvars, floatX
from .advi import ADVIFit, adagrad_optimizer, gen_random_state

__all__ = ['advi_minibatch']


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

        s = floatX(total_size / minibatch_tensors[0].shape[0])
        local_RVs = OrderedDict()
        observed_RVs = OrderedDict([(v, s) for v in minibatch_RVs])

    else:
        _value_error((isinstance(local_RVs, OrderedDict) and
                      isinstance(observed_RVs, OrderedDict)),
                     'local_RVs and observed_RVs must be OrderedDict.')

    return local_RVs, observed_RVs


def _init_uw_global_shared(start, global_RVs):
    global_order = pm.ArrayOrdering([v for v in global_RVs])
    start = {v.name: start[v.name] for v in global_RVs}
    bij = pm.DictToArrayBijection(global_order, start)
    u_start = bij.map(start)
    w_start = np.zeros_like(u_start)
    uw_start = floatX(np.concatenate([u_start, w_start]))
    uw_global_shared = theano.shared(uw_start, 'uw_global_shared')

    return uw_global_shared, bij


def _join_global_RVs(global_RVs, global_order):
    if len(global_RVs) == 0:
        inarray_global = None
        uw_global = None
        replace_global = {}
        c_g = 0
    else:
        joined_global = tt.concatenate([v.ravel() for v in global_RVs])
        uw_global = tt.vector('uw_global')
        uw_global.tag.test_value = np.concatenate(
            [joined_global.tag.test_value, joined_global.tag.test_value]
        )

        inarray_global = joined_global.type('inarray_global')
        inarray_global.tag.test_value = joined_global.tag.test_value

        # Replace RVs with reshaped subvectors of the joined vector
        # The order of global_order is the same with that of global_RVs
        subvecs = [reshape_t(inarray_global[slc], shp).astype(dtyp)
                   for _, slc, shp, dtyp in global_order.vmap]
        replace_global = {v: subvec for v, subvec in zip(global_RVs, subvecs)}

        # Weight vector
        cs = [c for _, c in global_RVs.items()]
        oness = [tt.ones(v.ravel().tag.test_value.shape) for v in global_RVs]
        c_g = tt.concatenate([c * ones for c, ones in zip(cs, oness)])

    return inarray_global, uw_global, replace_global, c_g


def _join_local_RVs(local_RVs, local_order):
    if len(local_RVs) == 0:
        inarray_local = None
        uw_local = None
        replace_local = {}
        c_l = 0
    else:
        joined_local = tt.concatenate([v.ravel() for v in local_RVs])
        uw_local = tt.vector('uw_local')
        uw_local.tag.test_value = np.concatenate([joined_local.tag.test_value,
                                                  joined_local.tag.test_value])

        inarray_local = joined_local.type('inarray_local')
        inarray_local.tag.test_value = joined_local.tag.test_value

        get_var = {var.name: var for var in local_RVs}
        replace_local = {
            get_var[var]: reshape_t(inarray_local[slc], shp).astype(dtyp)
            for var, slc, shp, dtyp in local_order.vmap
        }

        # Weight vector
        cs = [c for _, (_, c) in local_RVs.items()]
        oness = [tt.ones(v.ravel().tag.test_value.shape) for v in local_RVs]
        c_l = tt.concatenate([c * ones for c, ones in zip(cs, oness)])

    return inarray_local, uw_local, replace_local, c_l


def _make_logpt(global_RVs, local_RVs, observed_RVs, potentials):
    """Return expression of log probability.
    """
    # Scale log probability for mini-batches
    factors = [c * v.logpt for v, c in observed_RVs.items()] + \
              [c * v.logpt for v, c in global_RVs.items()] + \
              [c * v.logpt for v, (_, c) in local_RVs.items()] + \
              potentials
    logpt = tt.add(*map(tt.sum, factors))

    return logpt


def _elbo_t(
    logp, uw_g, uw_l, inarray_g, inarray_l, c_g, c_l, n_mcsamples,
    random_seed):
    """Return expression of approximate ELBO based on Monte Carlo sampling.
    """
    if random_seed is None:
        r = MRG_RandomStreams(gen_random_state())
    else:
        r = MRG_RandomStreams(seed=random_seed)

    normal_const = floatX(1 + np.log(2.0 * np.pi))

    elbo = 0

    # Sampling local variational parameters
    if uw_l is not None:
        l_l = (uw_l.size / 2).astype('int32')
        u_l = uw_l[:l_l]
        w_l = uw_l[l_l:]
        ns_l = r.normal(size=(n_mcsamples, inarray_l.tag.test_value.shape[0]))
        zs_l = ns_l * tt.exp(w_l) + u_l
        elbo += tt.sum(c_l * (w_l + 0.5 * normal_const))
    else:
        zs_l = None

    # Sampling global variational parameters
    if uw_g is not None:
        l_g = (uw_g.size / 2).astype('int32')
        u_g = uw_g[:l_g]
        w_g = uw_g[l_g:]
        ns_g = r.normal(size=(n_mcsamples, inarray_g.tag.test_value.shape[0]))
        zs_g = ns_g * tt.exp(w_g) + u_g
        elbo += tt.sum(c_g * (w_g + 0.5 * normal_const))
    else:
        zs_g = None

    if (zs_l is not None) and (zs_g is not None):
        def logp_(z_g, z_l):
            return theano.clone(
                logp, OrderedDict({inarray_g: z_g, inarray_l: z_l}),
                strict=False
            )
        sequences = [zs_g, zs_l]

    elif zs_l is not None:
        def logp_(z_l):
            return theano.clone(
                logp, OrderedDict({inarray_l: z_l}),
                strict=False
            )
        sequences = [zs_l]

    else:
        def logp_(z_g):
            return theano.clone(
                logp, OrderedDict({inarray_g: z_g}),
                strict=False
            )
        sequences = [zs_g]

    logps, _ = theano.scan(fn=logp_, outputs_info=None, sequences=sequences)
    elbo += tt.mean(logps)

    return elbo


def _make_elbo_t(
    observed_RVs, global_RVs, local_RVs, potentials, n_mcsamples, random_seed):
    global_order = pm.ArrayOrdering([v for v in global_RVs])
    local_order = pm.ArrayOrdering([v for v in local_RVs])

    inarray_g, uw_g, replace_g, c_g = _join_global_RVs(global_RVs, global_order)
    inarray_l, uw_l, replace_l, c_l = _join_local_RVs(local_RVs, local_order)

    logpt = _make_logpt(global_RVs, local_RVs, observed_RVs, potentials)
    replace = replace_g
    replace.update(replace_l)
    logpt = theano.clone(logpt, replace, strict=False)

    elbo = _elbo_t(logpt, uw_g, uw_l, inarray_g, inarray_l, c_g, c_l,
                   n_mcsamples, random_seed)

    return elbo, uw_l, uw_g


@change_flags(compute_test_value='ignore')
def advi_minibatch(vars=None, start=None, model=None, n=5000, n_mcsamples=1,
                   minibatch_RVs=None, minibatch_tensors=None,
                   minibatches=None, global_RVs=None, local_RVs=None,
                   observed_RVs=None, encoder_params=None, total_size=None,
                   optimizer=None, learning_rate=.001, epsilon=.1,
                   random_seed=None, mode=None):
    """Perform mini-batch ADVI.

    This function implements a mini-batch automatic differentiation variational
    inference (ADVI; Kucukelbir et al., 2015) with the meanfield
    approximation. Autoencoding variational Bayes (AEVB; Kingma and Welling,
    2014) is also supported.

    For explanation, we classify random variables in probabilistic models into
    three types. Observed random variables
    :math:`{\cal Y}=\{\mathbf{y}_{i}\}_{i=1}^{N}` are :math:`N` observations.
    Each :math:`\mathbf{y}_{i}` can be a set of observed random variables,
    i.e., :math:`\mathbf{y}_{i}=\{\mathbf{y}_{i}^{k}\}_{k=1}^{V_{o}}`, where
    :math:`V_{k}` is the number of the types of observed random variables
    in the model.

    The next ones are global random variables
    :math:`\Theta=\{\\theta^{k}\}_{k=1}^{V_{g}}`, which are used to calculate
    the probabilities for all observed samples.

    The last ones are local random variables
    :math:`{\cal Z}=\{\mathbf{z}_{i}\}_{i=1}^{N}`, where
    :math:`\mathbf{z}_{i}=\{\mathbf{z}_{i}^{k}\}_{k=1}^{V_{l}}`.
    These RVs are used only in AEVB.

    The goal of ADVI is to approximate the posterior distribution
    :math:`p(\Theta,{\cal Z}|{\cal Y})` by variational posterior
    :math:`q(\Theta)\prod_{i=1}^{N}q(\mathbf{z}_{i})`. All of these terms
    are normal distributions (mean-field approximation).

    :math:`q(\Theta)` is parametrized with its means and standard deviations.
    These parameters are denoted as :math:`\gamma`. While :math:`\gamma` is
    a constant, the parameters of :math:`q(\mathbf{z}_{i})` are dependent on
    each observation. Therefore these parameters are denoted as
    :math:`\\xi(\mathbf{y}_{i}; \\nu)`, where :math:`\\nu` is the parameters
    of :math:`\\xi(\cdot)`. For example, :math:`\\xi(\cdot)` can be a
    multilayer perceptron or convolutional neural network.

    In addition to :math:`\\xi(\cdot)`, we can also include deterministic
    mappings for the likelihood of observations. We denote the parameters of
    the deterministic mappings as :math:`\eta`. An example of such mappings is
    the deconvolutional neural network used in the convolutional VAE example
    in the PyMC3 notebook directory.

    This function maximizes the evidence lower bound (ELBO)
    :math:`{\cal L}(\gamma, \\nu, \eta)` defined as follows:

    .. math::

        {\cal L}(\gamma,\\nu,\eta) & =
        \mathbf{c}_{o}\mathbb{E}_{q(\Theta)}\left[
        \sum_{i=1}^{N}\mathbb{E}_{q(\mathbf{z}_{i})}\left[
        \log p(\mathbf{y}_{i}|\mathbf{z}_{i},\Theta,\eta)
        \\right]\\right] \\\\ &
        - \mathbf{c}_{g}KL\left[q(\Theta)||p(\Theta)\\right]
        - \mathbf{c}_{l}\sum_{i=1}^{N}
            KL\left[q(\mathbf{z}_{i})||p(\mathbf{z}_{i})\\right],

    where :math:`KL[q(v)||p(v)]` is the Kullback-Leibler divergence

    .. math::

        KL[q(v)||p(v)] = \int q(v)\log\\frac{q(v)}{p(v)}dv,

    :math:`\mathbf{c}_{o/g/l}` are vectors for weighting each term of ELBO.
    More precisely, we can write each of the terms in ELBO as follows:

    .. math::

        \mathbf{c}_{o}\log p(\mathbf{y}_{i}|\mathbf{z}_{i},\Theta,\eta) & = &
        \sum_{k=1}^{V_{o}}c_{o}^{k}
            \log p(\mathbf{y}_{i}^{k}|
                   {\\rm pa}(\mathbf{y}_{i}^{k},\Theta,\eta)) \\\\
        \mathbf{c}_{g}KL\left[q(\Theta)||p(\Theta)\\right] & = &
        \sum_{k=1}^{V_{g}}c_{g}^{k}KL\left[
            q(\\theta^{k})||p(\\theta^{k}|{\\rm pa(\\theta^{k})})\\right] \\\\
        \mathbf{c}_{l}KL\left[q(\mathbf{z}_{i}||p(\mathbf{z}_{i})\\right] & = &
        \sum_{k=1}^{V_{l}}c_{l}^{k}KL\left[
            q(\mathbf{z}_{i}^{k})||
            p(\mathbf{z}_{i}^{k}|{\\rm pa}(\mathbf{z}_{i}^{k}))\\right],

    where :math:`{\\rm pa}(v)` denotes the set of parent variables of :math:`v`
    in the directed acyclic graph of the model.

    When using mini-batches, :math:`c_{o}^{k}` and :math:`c_{l}^{k}` should be
    set to :math:`N/M`, where :math:`M` is the number of observations in each
    mini-batch. Another weighting scheme was proposed in
    (Blundell et al., 2015) for accelarating model fitting.

    For working with ADVI, we need to give the probabilistic model
    (:code:`model`), the three types of RVs (:code:`observed_RVs`,
    :code:`global_RVs` and :code:`local_RVs`), the tensors to which
    mini-bathced samples are supplied (:code:`minibatches`) and
    parameters of deterministic mappings :math:`\\xi` and :math:`\eta`
    (:code:`encoder_params`) as input arguments.

    :code:`observed_RVs` is a :code:`OrderedDict` of the form
    :code:`{y_k: c_k}`, where :code:`y_k` is a random variable defined in the
    PyMC3 model. :code:`c_k` is a scalar (:math:`c_{o}^{k}`) and it can be a
    shared variable.

    :code:`global_RVs` is a :code:`OrderedDict` of the form
    :code:`{t_k: c_k}`, where :code:`t_k` is a random variable defined in the
    PyMC3 model. :code:`c_k` is a scalar (:math:`c_{g}^{k}`) and it can be a
    shared variable.

    :code:`local_RVs` is a :code:`OrderedDict` of the form
    :code:`{z_k: ((m_k, s_k), c_k)}`, where :code:`z_k` is a random variable
    defined in the PyMC3 model. :code:`c_k` is a scalar (:math:`c_{l}^{k}`)
    and it can be a shared variable. :code:`(m_k, s_k)` is a pair of tensors
    of means and log standard deviations of the variational distribution;
    samples drawn from the variational distribution replaces :code:`z_k`.
    It should be noted that if :code:`z_k` has a transformation that changes
    the dimension (e.g., StickBreakingTransform), the variational distribution
    must have the same dimension. For example, if :code:`z_k` is distributed
    with Dirichlet distribution with :code:`p` choices, :math:`m_k` and
    :code:`s_k` has the shape :code:`(n_samples_in_minibatch, p - 1)`.

    :code:`minibatch_tensors` is a list of tensors (can be shared variables)
    to which mini-batch samples are set during the optimization.
    These tensors are observations (:code:`obs=`) in :code:`observed_RVs`.

    :code:`minibatches` is a generator of a list of :code:`numpy.ndarray`.
    Each item of the list will be set to tensors in :code:`minibatch_tensors`.

    :code:`encoder_params` is a list of shared variables of the parameters
    :math:`\\nu` and :math:`\eta`. We do not need to include the variational
    parameters of the global variables, :math:`\gamma`, because these are
    automatically created and updated in this function.

    The following is a list of example notebooks using advi_minibatch:

    - docs/source/notebooks/GLM-hierarchical-advi-minibatch.ipynb
    - docs/source/notebooks/bayesian_neural_network_advi.ipynb
    - docs/source/notebooks/convolutional_vae_keras_advi.ipynb
    - docs/source/notebooks/gaussian-mixture-model-advi.ipynb
    - docs/source/notebooks/lda-advi-aevb.ipynb

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
        Number of iterations updating parameters.
    n_mcsamples : int
        Number of Monte Carlo samples to approximate ELBO.
    minibatch_RVs : list of ObservedRVs
        Random variables in the model for which mini-batch tensors are set.
        When this argument is given, both of arguments local_RVs and
        observed_RVs must be None.
    minibatch_tensors : list of (tensors or shared variables)
        Tensors used to create ObservedRVs in minibatch_RVs.
    minibatches : generator of list
        Generates a set of minibatches when calling next().
        The length of the returned list must be the same with the number of
        random variables in `minibatch_tensors`.
    total_size : int
        Total size of training samples. This is used to appropriately scale the
        log likelihood terms corresponding to mini-batches in ELBO.
    observed_RVs : Ordered dict
        Include a scaling constant for the corresponding RV. See the above
        description.
    global_RVs : Ordered dict or None
        Include a scaling constant for the corresponding RV. See the above
        description. If :code:`None`, it is set to
        :code:`{v: 1 for v in grvs}`, where :code:`grvs` is
        :code:`list(set(vars) - set(list(local_RVs) + list(observed_RVs)))`.
    local_RVs : Ordered dict or None
        Include encoded variational parameters and a scaling constant for
        the corresponding RV. See the above description.
    encoder_params : list of theano shared variables
        Parameters of encoder.
    optimizer : (loss, list of shared variables) -> dict or OrderedDict
        A function that returns parameter updates given loss and shared
        variables of parameters. If :code:`None` (default), a default
        Adagrad optimizer is used with parameters :code:`learning_rate`
        and :code:`epsilon` below.
    learning_rate: float
        Base learning rate for adagrad.
        This parameter is ignored when :code:`optimizer` is set.
    epsilon : float
        Offset in denominator of the scale of learning rate in Adagrad.
        This parameter is ignored when :code:`optimizer` is set.
    random_seed : int
        Seed to initialize random state.

    Returns
    -------
    ADVIFit
        Named tuple, which includes 'means', 'stds', and 'elbo_vals'.

    References
    ----------
    - Kingma, D. P., & Welling, M. (2014).
      Auto-Encoding Variational Bayes. stat, 1050, 1.
    - Kucukelbir, A., Ranganath, R., Gelman, A., & Blei, D. (2015).
      Automatic variational inference in Stan. In Advances in neural
      information processing systems (pp. 568-576).
    - Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015).
      Weight Uncertainty in Neural Network. In Proceedings of the 32nd
      International Conference on Machine Learning (ICML-15) (pp. 1613-1622).
    """
    if encoder_params is None:
        encoder_params = []

    model = pm.modelcontext(model)
    vars = inputvars(vars if vars is not None else model.vars)
    start = start if start is not None else model.test_point

    if not pm.model.all_continuous(vars):
        raise ValueError('Model can not include discrete RVs for ADVI.')

    _check_minibatches(minibatch_tensors, minibatches)

    if encoder_params is None:
        encoder_params = []

    # Prepare optimizer
    if optimizer is None:
        optimizer = adagrad_optimizer(learning_rate, epsilon)

    # For backward compatibility in how input arguments are given
    local_RVs, observed_RVs = _get_rvss(minibatch_RVs, local_RVs, observed_RVs,
                                        minibatch_tensors, total_size)

    # Replace local_RVs with transformed variables
    def get_transformed(v):
        if hasattr(v, 'transformed'):
            return v.transformed
        return v
    local_RVs = OrderedDict(
        [(get_transformed(v), (uw, s)) for v, (uw, s) in local_RVs.items()]
    )

    # Get global variables
    grvs = list(set(vars) - set(list(local_RVs) + list(observed_RVs)))
    if global_RVs is None:
        global_RVs = OrderedDict({v: 1 for v in grvs})
    elif len(grvs) != len(global_RVs):
        _value_error(
            'global_RVs ({}) must have all global RVs: {}'.format(
                [v for v in global_RVs], grvs
            )
        )

    # ELBO wrt variational parameters
    elbo, uw_l, uw_g = _make_elbo_t(observed_RVs, global_RVs, local_RVs,
                                    model.potentials, n_mcsamples, random_seed)

    # Replacements tensors of variational parameters in the graph
    replaces = dict()

    # Variational parameters for global RVs
    if 0 < len(global_RVs):
        uw_global_shared, bij = _init_uw_global_shared(start, global_RVs)
        replaces.update({uw_g: uw_global_shared})

    # Variational parameters for local RVs, encoded from samples in
    # mini-batches
    if 0 < len(local_RVs):
        uws = [uw for _, (uw, _) in local_RVs.items()]
        uw_local_encoded = tt.concatenate([uw[0].ravel() for uw in uws] +
                                          [uw[1].ravel() for uw in uws])
        replaces.update({uw_l: uw_local_encoded})

    # Replace tensors of variational parameters in ELBO
    elbo = theano.clone(elbo, OrderedDict(replaces), strict=False)

    # Replace input shared variables with tensors
    def is_shared(t):
        return isinstance(t, theano.compile.sharedvalue.SharedVariable)
    tensors = [(t.type() if is_shared(t) else t) for t in minibatch_tensors]
    updates = OrderedDict(
        {t: t_ for t, t_ in zip(minibatch_tensors, tensors) if is_shared(t)}
    )
    elbo = theano.clone(elbo, updates, strict=False)

    # Create parameter update function used in the training loop
    params = encoder_params
    if 0 < len(global_RVs):
        params += [uw_global_shared]
    updates = OrderedDict(optimizer(loss=-1 * elbo, param=params))
    f = theano.function(tensors, elbo, updates=updates, mode=mode)

    # Optimization loop
    elbos = np.empty(n)
    progress = tqdm.trange(n)
    for i in progress:
        e = f(*next(minibatches))
        if np.isnan(e):
            raise FloatingPointError('NaN occurred in ADVI optimization.')
        elbos[i] = e
        if n < 10:
            progress.set_description('ELBO = {:,.2f}'.format(elbos[i]))
        elif i % (n // 10) == 0 and i > 0:
            avg_elbo = elbos[i - n // 10:i].mean()
            progress.set_description('Average ELBO = {:,.2f}'.format(avg_elbo))

    pm._log.info('Finished minibatch ADVI: ELBO = {:,.2f}'.format(elbos[-1]))

    # Variational parameters of global RVs
    if 0 < len(global_RVs):
        l = int(uw_global_shared.get_value(borrow=True).size / 2)
        u = bij.rmap(uw_global_shared.get_value(borrow=True)[:l])
        w = bij.rmap(uw_global_shared.get_value(borrow=True)[l:])
        # w is in log space
        for var in w.keys():
            w[var] = np.exp(w[var])
    else:
        u = dict()
        w = dict()

    return ADVIFit(u, w, elbos)
