"""Statistical utility functions for PyMC"""
from collections import namedtuple
import itertools
import pkg_resources
import warnings

import numpy as np
import pandas as pd
from scipy.stats import dirichlet
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from tqdm import tqdm

from .model import modelcontext
from .util import get_default_varnames
import pymc3 as pm
from pymc3.theanof import floatX

if pkg_resources.get_distribution('scipy').version < '1.0.0':
    from scipy.misc import logsumexp
else:
    from scipy.special import logsumexp


__all__ = ['autocorr', 'autocov', 'waic', 'loo', 'hpd', 'quantiles',
           'mc_error', 'summary', 'compare', 'bfmi', 'r2_score']


def statfunc(f):
    """
    Decorator for statistical utility function to automatically
    extract the trace array from whatever object is passed.
    """

    def wrapped_f(pymc3_obj, *args, **kwargs):
        try:
            vars = kwargs.pop('vars',  pymc3_obj.varnames)
            chains = kwargs.pop('chains', pymc3_obj.chains)
        except AttributeError:
            # If fails, assume that raw data was passed.
            return f(pymc3_obj, *args, **kwargs)

        burn = kwargs.pop('burn', 0)
        thin = kwargs.pop('thin', 1)
        combine = kwargs.pop('combine', False)
        # Remove outer level chain keys if only one chain)
        squeeze = kwargs.pop('squeeze', True)

        results = {chain: {} for chain in chains}
        for var in vars:
            samples = pymc3_obj.get_values(var, chains=chains, burn=burn,
                                           thin=thin, combine=combine,
                                           squeeze=False)
            for chain, data in zip(chains, samples):
                results[chain][var] = f(np.squeeze(data), *args, **kwargs)

        if squeeze and (len(chains) == 1 or combine):
            results = results[chains[0]]
        return results

    wrapped_f.__doc__ = f.__doc__
    wrapped_f.__name__ = f.__name__

    return wrapped_f


@statfunc
def autocorr(x, lag=None):
    """
    Compute autocorrelation using FFT for every lag for the input array
    https://en.wikipedia.org/wiki/Autocorrelation#Efficient_computation

    Parameters
    ----------
    x : Numpy array
        An array containing MCMC samples

    Returns
    -------
    acorr: Numpy array same size as the input array
    """
    y = x - x.mean()
    n = len(y)
    result = fftconvolve(y, y[::-1])
    acorr = result[len(result) // 2:]
    acorr /= np.arange(n, 0, -1)
    acorr /= acorr[0]
    if lag is None:
        return acorr
    else:
        warnings.warn(
            "The `lag` argument has been deprecated. If you want to get "
            "the value of a specific lag please call `autocorr(x)[lag]`.",
            DeprecationWarning)
        return acorr[lag]


@statfunc
def autocov(x, lag=None):
    """Compute autocovariance estimates for every lag for the input array

    Parameters
    ----------
    x : Numpy array
        An array containing MCMC samples

    Returns
    -------
    acov: Numpy array same size as the input array
    """
    acorr = autocorr(x)
    varx = np.var(x, ddof=1) * (len(x) - 1) / len(x)
    acov = acorr * varx
    if lag is None:
        return acov
    else:
        warnings.warn(
            "The `lag` argument has been deprecated. If you want to get "
            "the value of a specific lag please call `autocov(x)[lag]`.",
            DeprecationWarning)
        return acov[lag]


def _log_post_trace(trace, model=None, progressbar=False):
    """Calculate the elementwise log-posterior for the sampled trace.

    Parameters
    ----------
    trace : result of MCMC run
    model : PyMC Model
        Optional model. Default None, taken from context.
    progressbar: bool
        Whether or not to display a progress bar in the command line. The
        bar shows the percentage of completion, the evaluation speed, and
        the estimated time to completion

    Returns
    -------
    logp : array of shape (n_samples, n_observations)
        The contribution of the observations to the logp of the whole model.
    """
    model = modelcontext(model)
    cached = [(var, var.logp_elemwise) for var in model.observed_RVs]

    def logp_vals_point(pt):
        if len(model.observed_RVs) == 0:
            return floatX(np.array([], dtype='d'))

        logp_vals = []
        for var, logp in cached:
            logp = logp(pt)
            if var.missing_values:
                logp = logp[~var.observations.mask]
            logp_vals.append(logp.ravel())

        return np.concatenate(logp_vals)

    try:
        points = trace.points()
    except AttributeError:
        points = trace

    points = tqdm(points) if progressbar else points

    try:
        logp = (logp_vals_point(pt) for pt in points)
        return np.stack(logp)
    finally:
        if progressbar:
            points.close()


WAIC_r_pointwise = namedtuple('WAIC_r_pointwise', 'WAIC, WAIC_se, p_WAIC, var_warn, WAIC_i')
WAIC_r = namedtuple('WAIC_r', 'WAIC, WAIC_se, p_WAIC, var_warn')
def waic(trace, model=None, pointwise=False, progressbar=False):
    """Calculate the widely available information criterion, its standard error
    and the effective number of parameters of the samples in trace from model.
    Read more theory here - in a paper by some of the leading authorities on
    model selection - dx.doi.org/10.1111/1467-9868.00353

    Parameters
    ----------
    trace : result of MCMC run
    model : PyMC Model
        Optional model. Default None, taken from context.
    pointwise: bool
        if True the pointwise predictive accuracy will be returned.
        Default False
    progressbar: bool
        Whether or not to display a progress bar in the command line. The
        bar shows the percentage of completion, the evaluation speed, and
        the estimated time to completion

    Returns
    -------
    namedtuple with the following elements:
    waic: widely available information criterion
    waic_se: standard error of waic
    p_waic: effective number parameters
    var_warn: 1 if posterior variance of the log predictive
         densities exceeds 0.4
    waic_i: and array of the pointwise predictive accuracy, only if pointwise True
    """
    model = modelcontext(model)

    log_py = _log_post_trace(trace, model, progressbar=progressbar)
    if log_py.size == 0:
        raise ValueError('The model does not contain observed values.')

    lppd_i = logsumexp(log_py, axis=0, b=1.0 / log_py.shape[0])

    vars_lpd = np.var(log_py, axis=0)
    warn_mg = 0
    if np.any(vars_lpd > 0.4):
        warnings.warn("""For one or more samples the posterior variance of the
        log predictive densities exceeds 0.4. This could be indication of
        WAIC starting to fail see http://arxiv.org/abs/1507.04544 for details
        """)
        warn_mg = 1

    waic_i = - 2 * (lppd_i - vars_lpd)

    waic_se = np.sqrt(len(waic_i) * np.var(waic_i))

    waic = np.sum(waic_i)

    p_waic = np.sum(vars_lpd)

    if pointwise:
        if np.equal(waic, waic_i).all():
            warnings.warn("""The point-wise WAIC is the same with the sum WAIC,
            please double check the Observed RV in your model to make sure it
            returns element-wise logp.
            """)
        return WAIC_r_pointwise(waic, waic_se, p_waic, warn_mg, waic_i)
    else:
        return WAIC_r(waic, waic_se, p_waic, warn_mg)


LOO_r_pointwise = namedtuple('LOO_r_pointwise', 'LOO, LOO_se, p_LOO, shape_warn, LOO_i')
LOO_r = namedtuple('LOO_r', 'LOO, LOO_se, p_LOO, shape_warn')
def loo(trace, model=None, pointwise=False, reff=None, progressbar=False):
    """Calculates leave-one-out (LOO) cross-validation for out of sample
    predictive model fit, following Vehtari et al. (2015). Cross-validation is
    computed using Pareto-smoothed importance sampling (PSIS).

    Parameters
    ----------
    trace : result of MCMC run
    model : PyMC Model
        Optional model. Default None, taken from context.
    pointwise: bool
        if True the pointwise predictive accuracy will be returned.
        Default False
    reff : float
        relative MCMC efficiency, `effective_n / n` i.e. number of effective
        samples divided by the number of actual samples. Computed from trace by
        default.
    progressbar: bool
        Whether or not to display a progress bar in the command line. The
        bar shows the percentage of completion, the evaluation speed, and
        the estimated time to completion

    Returns
    -------
    namedtuple with the following elements:
    loo: approximated Leave-one-out cross-validation
    loo_se: standard error of loo
    p_loo: effective number of parameters
    shape_warn: 1 if the estimated shape parameter of
        Pareto distribution is greater than 0.7 for one or more samples
    loo_i: array of pointwise predictive accuracy, only if pointwise True
    """
    model = modelcontext(model)

    if reff is None:
        if trace.nchains == 1:
            reff = 1.
        else:
            eff = pm.effective_n(trace)
            eff_ave = pm.stats.dict2pd(eff, 'eff').mean()
            samples = len(trace) * trace.nchains
            reff = eff_ave / samples

    log_py = _log_post_trace(trace, model, progressbar=progressbar)
    if log_py.size == 0:
        raise ValueError('The model does not contain observed values.')

    lw, ks = _psislw(-log_py, reff)
    lw += log_py

    warn_mg = 0
    if np.any(ks > 0.7):
        warnings.warn("""Estimated shape parameter of Pareto distribution is
        greater than 0.7 for one or more samples.
        You should consider using a more robust model, this is because
        importance sampling is less likely to work well if the marginal
        posterior and LOO posterior are very different. This is more likely to
        happen with a non-robust model and highly influential observations.""")
        warn_mg = 1

    loo_lppd_i = - 2 * logsumexp(lw, axis=0)
    loo_lppd = loo_lppd_i.sum()
    loo_lppd_se = (len(loo_lppd_i) * np.var(loo_lppd_i)) ** 0.5
    lppd = np.sum(logsumexp(log_py, axis=0, b=1. / log_py.shape[0]))
    p_loo = lppd + (0.5 * loo_lppd)

    if pointwise:
        if np.equal(loo_lppd, loo_lppd_i).all():
            warnings.warn("""The point-wise LOO is the same with the sum LOO,
            please double check the Observed RV in your model to make sure it
            returns element-wise logp.
            """)
        return LOO_r_pointwise(loo_lppd, loo_lppd_se, p_loo, warn_mg, loo_lppd_i)
    else:
        return LOO_r(loo_lppd, loo_lppd_se, p_loo, warn_mg)


def _psislw(lw, reff):
    """Pareto smoothed importance sampling (PSIS).

    Parameters
    ----------
    lw : array
        Array of size (n_samples, n_observations)
    reff : float
        relative MCMC efficiency, `effective_n / n`

    Returns
    -------
    lw_out : array
        Smoothed log weights
    kss : array
        Pareto tail indices
    """
    n, m = lw.shape

    lw_out = np.copy(lw, order='F')
    kss = np.empty(m)

    # precalculate constants
    cutoff_ind = - int(np.ceil(min(n / 5., 3 * (n / reff) ** 0.5))) - 1
    cutoffmin = np.log(np.finfo(float).tiny)
    k_min = 1. / 3

    # loop over sets of log weights
    for i, x in enumerate(lw_out.T):
        # improve numerical accuracy
        x -= np.max(x)
        # sort the array
        x_sort_ind = np.argsort(x)
        # divide log weights into body and right tail
        xcutoff = max(x[x_sort_ind[cutoff_ind]], cutoffmin)

        expxcutoff = np.exp(xcutoff)
        tailinds, = np.where(x > xcutoff)
        x2 = x[tailinds]
        n2 = len(x2)
        if n2 <= 4:
            # not enough tail samples for gpdfit
            k = np.inf
        else:
            # order of tail samples
            x2si = np.argsort(x2)
            # fit generalized Pareto distribution to the right tail samples
            x2 = np.exp(x2) - expxcutoff
            k, sigma = _gpdfit(x2[x2si])

        if k >= k_min and not np.isinf(k):
            # no smoothing if short tail or GPD fit failed
            # compute ordered statistic for the fit
            sti = np.arange(0.5, n2) / n2
            qq = _gpinv(sti, k, sigma)
            qq = np.log(qq + expxcutoff)
            # place the smoothed tail into the output array
            x[tailinds[x2si]] = qq
            # truncate smoothed values to the largest raw weight 0
            x[x > 0] = 0
        # renormalize weights
        x -= logsumexp(x)
        # store tail index k
        kss[i] = k

    return lw_out, kss


def _gpdfit(x):
    """Estimate the parameters for the Generalized Pareto Distribution (GPD)

    Empirical Bayes estimate for the parameters of the generalized Pareto
    distribution given the data.

    Parameters
    ----------
    x : array
        sorted 1D data array

    Returns
    -------
    k : float
        estimated shape parameter
    sigma : float
        estimated scale parameter
    """
    prior_bs = 3
    prior_k = 10
    n = len(x)
    m = 30 + int(n**0.5)

    bs = 1 - np.sqrt(m / (np.arange(1, m + 1, dtype=float) - 0.5))
    bs /= prior_bs * x[int(n/4 + 0.5) - 1]
    bs += 1 / x[-1]

    ks = np.log1p(-bs[:, None] * x).mean(axis=1)
    L = n * (np.log(-(bs / ks)) - ks - 1)
    w = 1 / np.exp(L - L[:, None]).sum(axis=1)

    # remove negligible weights
    dii = w >= 10 * np.finfo(float).eps
    if not np.all(dii):
        w = w[dii]
        bs = bs[dii]
    # normalise w
    w /= w.sum()

    # posterior mean for b
    b = np.sum(bs * w)
    # estimate for k
    k = np.log1p(- b * x).mean()
    # add prior for k
    k = (n * k + prior_k * 0.5) / (n + prior_k)
    sigma = - k / b

    return k, sigma


def _gpinv(p, k, sigma):
    """Inverse Generalized Pareto distribution function"""
    x = np.full_like(p, np.nan)
    if sigma <= 0:
        return x
    ok = (p > 0) & (p < 1)
    if np.all(ok):
        if np.abs(k) < np.finfo(float).eps:
            x = - np.log1p(-p)
        else:
            x = np.expm1(-k * np.log1p(-p)) / k
        x *= sigma
    else:
        if np.abs(k) < np.finfo(float).eps:
            x[ok] = - np.log1p(-p[ok])
        else:
            x[ok] = np.expm1(-k * np.log1p(-p[ok])) / k
        x *= sigma
        x[p == 0] = 0
        if k >= 0:
            x[p == 1] = np.inf
        else:
            x[p == 1] = - sigma / k

    return x


def compare(model_dict, ic='WAIC', method='stacking', b_samples=1000,
            alpha=1, seed=None, round_to=2):
    R"""Compare models based on the widely available information criterion (WAIC)
    or leave-one-out (LOO) cross-validation.
    Read more theory here - in a paper by some of the leading authorities on
    model selection - dx.doi.org/10.1111/1467-9868.00353

    Parameters
    ----------
    model_dict : dictionary of PyMC3 traces indexed by corresponding model
    ic : string
        Information Criterion (WAIC or LOO) used to compare models.
        Default WAIC.
    method : str
        Method used to estimate the weights for each model. Available options
        are:
            - 'stacking' : (default) stacking of predictive distributions.
            - 'BB-pseudo-BMA' : pseudo-Bayesian Model averaging using Akaike-type
               weighting. The weights are stabilized using the Bayesian bootstrap
            - 'pseudo-BMA': pseudo-Bayesian Model averaging using Akaike-type
               weighting, without Bootstrap stabilization (not recommended)

        For more information read https://arxiv.org/abs/1704.02030
    b_samples: int
        Number of samples taken by the Bayesian bootstrap estimation. Only
        useful when method = 'BB-pseudo-BMA'.
    alpha : float
        The shape parameter in the Dirichlet distribution used for the
        Bayesian bootstrap. Only useful when method = 'BB-pseudo-BMA'. When
        alpha=1 (default), the distribution is uniform on the simplex. A
        smaller alpha will keeps the final weights more away from 0 and 1.
    seed : int or np.random.RandomState instance
           If int or RandomState, use it for seeding Bayesian bootstrap. Only
           useful when method = 'BB-pseudo-BMA'. Default None the global
           np.random state is used.
    round_to : int
        Number of decimals used to round results (default 2).

    Returns
    -------
    A DataFrame, ordered from lowest to highest IC. The index reflects
    the order in which the models are passed to this function. The columns are:
    IC : Information Criteria (WAIC or LOO).
        Smaller IC indicates higher out-of-sample predictive fit ("better" model).
        Default WAIC.
    pIC : Estimated effective number of parameters.
    dIC : Relative difference between each IC (WAIC or LOO)
    and the lowest IC (WAIC or LOO).
        It's always 0 for the top-ranked model.
    weight: Relative weight for each model.
        This can be loosely interpreted as the probability of each model
        (among the compared model) given the data. By default the uncertainty
        in the weights estimation is considered using Bayesian bootstrap.
    SE : Standard error of the IC estimate.
        If method = BB-pseudo-BMA these values are estimated using Bayesian
        bootstrap.
    dSE : Standard error of the difference in IC between each model and
    the top-ranked model.
        It's always 0 for the top-ranked model.
    warning : A value of 1 indicates that the computation of the IC may not be
        reliable. Details see the related warning message in pm.waic and pm.loo
    """

    names = [model.name for model in model_dict if model.name]
    if not names:
        names = np.arange(len(model_dict))

    if ic == 'WAIC':
        ic_func = waic
        df_comp = pd.DataFrame(index=names,
                               columns=['WAIC', 'pWAIC', 'dWAIC', 'weight',
                                        'SE', 'dSE', 'var_warn'])

    elif ic == 'LOO':
        ic_func = loo
        df_comp = pd.DataFrame(index=names,
                               columns=['LOO', 'pLOO', 'dLOO', 'weight',
                                        'SE', 'dSE', 'shape_warn'])

    else:
        raise NotImplementedError(
            'The information criterion {} is not supported.'.format(ic))

    if len(set([len(m.observed_RVs) for m in model_dict])) != 1:
        raise ValueError(
            'The number of observed RVs should be the same across all models')

    if method not in ['stacking', 'BB-pseudo-BMA', 'pseudo-BMA']:
        raise ValueError('The method {}, to compute weights,'
                         'is not supported.'.format(method))

    ics = []
    for n, (m, t) in zip(names, model_dict.items()):
        ics.append((n, ic_func(t, m, pointwise=True)))

    ics.sort(key=lambda x: x[1][0])

    if method == 'stacking':
        N, K, ic_i = _ic_matrix(ics)
        exp_ic_i = np.exp(-0.5 * ic_i)
        Km = K - 1

        def w_fuller(w):
            return np.concatenate((w, [max(1. - np.sum(w), 0.)]))

        def log_score(w):
            w_full = w_fuller(w)
            score = 0.
            for i in range(N):
                score += np.log(np.dot(exp_ic_i[i], w_full))
            return -score

        def gradient(w):
            w_full = w_fuller(w)
            grad = np.zeros(Km)
            for k in range(Km):
                for i in range(N):
                    grad[k] += (exp_ic_i[i, k] - exp_ic_i[i, Km]) / \
                        np.dot(exp_ic_i[i], w_full)
            return -grad

        theta = np.full(Km, 1. / K)
        bounds = [(0., 1.) for i in range(Km)]
        constraints = [{'type': 'ineq', 'fun': lambda x: -np.sum(x) + 1.},
                       {'type': 'ineq', 'fun': lambda x: np.sum(x)}]

        w = minimize(fun=log_score,
                     x0=theta,
                     jac=gradient,
                     bounds=bounds,
                     constraints=constraints)

        weights = w_fuller(w['x'])
        ses = [res[1] for _, res in ics]

    elif method == 'BB-pseudo-BMA':
        N, K, ic_i = _ic_matrix(ics)
        ic_i = ic_i * N

        b_weighting = dirichlet.rvs(alpha=[alpha] * N, size=b_samples,
                                    random_state=seed)
        weights = np.zeros((b_samples, K))
        z_bs = np.zeros_like(weights)
        for i in range(b_samples):
            z_b = np.dot(b_weighting[i], ic_i)
            u_weights = np.exp(-0.5 * (z_b - np.min(z_b)))
            z_bs[i] = z_b
            weights[i] = u_weights / np.sum(u_weights)

        weights = weights.mean(0)
        ses = z_bs.std(0)

    elif method == 'pseudo-BMA':
        min_ic = ics[0][1][0]
        Z = np.sum([np.exp(-0.5 * (x[1][0] - min_ic)) for x in ics])
        weights = []
        ses = []
        for _, res in ics:
            weights.append(np.exp(-0.5 * (res[0] - min_ic)) / Z)
            ses.append(res[1])

    if np.any(weights):
        for i, (idx, res) in enumerate(ics):
            diff = res[4] - ics[0][1][4]
            d_ic = np.sum(diff)
            d_se = np.sqrt(len(diff) * np.var(diff))
            se = ses[i]
            weight = weights[i]
            df_comp.at[idx] = (round(res[0], round_to),
                               round(res[2], round_to),
                               round(d_ic, round_to),
                               round(weight, round_to),
                               round(se, round_to),
                               round(d_se, round_to),
                               res[3])

        return df_comp.sort_values(by=ic)


def _ic_matrix(ics):
    """Store the previously computed pointwise predictive accuracy values (ics)
    in a 2D matrix array.
    """
    N = len(ics[0][1][4])
    K = len(ics)
    ic_i = np.zeros((N, K))

    for i in range(K):
        ic = ics[i][1][4]
        if len(ic) != N:
            raise ValueError('The number of observations should be the same '
                             'across all models')
        else:
            ic_i[:, i] = ic

    return N, K, ic_i

def make_indices(dimensions):
    # Generates complete set of indices for given dimensions
    level = len(dimensions)
    if level == 1:
        return list(range(dimensions[0]))
    indices = [[]]
    while level:
        _indices = []
        for j in range(dimensions[level - 1]):
            _indices += [[j] + i for i in indices]
        indices = _indices
        level -= 1
    try:
        return [tuple(i) for i in indices]
    except TypeError:
        return indices


def calc_min_interval(x, alpha):
    """Internal method to determine the minimum interval of
    a given width

    Assumes that x is sorted numpy array.
    """
    n = len(x)
    cred_mass = 1.0 - alpha

    interval_idx_inc = int(np.floor(cred_mass * n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError('Too few elements for interval calculation')

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx + interval_idx_inc]
    return hdi_min, hdi_max


@statfunc
def hpd(x, alpha=0.05, transform=lambda x: x):
    """Calculate highest posterior density (HPD) of array for given alpha. The HPD is the
    minimum width Bayesian credible interval (BCI).

    This function assumes the posterior distribution is unimodal:
    it always returns one interval per variable.

    :Arguments:
      x : Numpy array
          An array containing MCMC samples
      alpha : float
          Desired probability of type I error (defaults to 0.05)
      transform : callable
          Function to transform data (defaults to identity)

    """
    # Make a copy of trace
    x = transform(x.copy())

    # For multivariate node
    if x.ndim > 1:

        # Transpose first, then sort
        tx = np.transpose(x, list(range(x.ndim))[1:] + [0])
        dims = np.shape(tx)

        # Container list for intervals
        intervals = np.resize(0.0, dims[:-1] + (2,))

        for index in make_indices(dims[:-1]):

            try:
                index = tuple(index)
            except TypeError:
                pass

            # Sort trace
            sx = np.sort(tx[index])

            # Append to list
            intervals[index] = calc_min_interval(sx, alpha)

        # Transpose back before returning
        return np.array(intervals)

    else:
        # Sort univariate node
        sx = np.sort(x)

        return np.array(calc_min_interval(sx, alpha))


def _hpd_df(x, alpha):
    cnames = ['hpd_{0:g}'.format(100 * alpha / 2),
              'hpd_{0:g}'.format(100 * (1 - alpha / 2))]
    return pd.DataFrame(hpd(x, alpha), columns=cnames)


@statfunc
def mc_error(x, batches=5):
    R"""Calculates the simulation standard error, accounting for non-independent
        samples. The trace is divided into batches, and the standard deviation of
        the batch means is calculated.

    Parameters
    ----------
    x : Numpy array
              An array containing MCMC samples
    batches : integer
              Number of batches

    Returns
    -------
    `float` representing the error
    """
    if x.ndim > 1:

        dims = np.shape(x)
        #ttrace = np.transpose(np.reshape(trace, (dims[0], sum(dims[1:]))))
        trace = np.transpose([t.ravel() for t in x])

        return np.reshape([mc_error(t, batches) for t in trace], dims[1:])

    else:
        if batches == 1:
            return np.std(x) / np.sqrt(len(x))

        try:
            batched_traces = np.resize(x, (batches, int(len(x) / batches)))
        except ValueError:
            # If batches do not divide evenly, trim excess samples
            resid = len(x) % batches
            new_shape = (batches, (len(x) - resid) / batches)
            batched_traces = np.resize(x[:-resid], new_shape)

        means = np.mean(batched_traces, 1)

        return np.std(means) / np.sqrt(batches)


@statfunc
def quantiles(x, qlist=(2.5, 25, 50, 75, 97.5), transform=lambda x: x):
    R"""Returns a dictionary of requested quantiles from array

    Parameters
    ----------
    x : Numpy array
        An array containing MCMC samples
    qlist : tuple or list
        A list of desired quantiles (defaults to (2.5, 25, 50, 75, 97.5))
    transform : callable
        Function to transform data (defaults to identity)

    Returns
    -------
    `dictionary` with the quantiles {quantile: value}
    """
    # Make a copy of trace
    x = transform(x.copy())

    # For multivariate node
    if x.ndim > 1:
        # Transpose first, then sort, then transpose back
        sx = np.sort(x.T).T
    else:
        # Sort univariate node
        sx = np.sort(x)

    try:
        # Generate specified quantiles
        quants = [sx[int(len(sx) * q / 100.0)] for q in qlist]

        return dict(zip(qlist, quants))

    except IndexError:
        pm._log.warning("Too few elements for quantile calculation")

def dict2pd(statdict, labelname):
    """Small helper function to transform a diagnostics output dict into a
    pandas Series.
    """
    from .backends import tracetab as ttab
    var_dfs = []
    for key, value in statdict.items():
        var_df = pd.Series(value.flatten())
        var_df.index = ttab.create_flat_names(key, value.shape)
        var_dfs.append(var_df)
    statpd = pd.concat(var_dfs, axis=0)
    statpd = statpd.rename(labelname)
    return statpd

def summary(trace, varnames=None, transform=lambda x: x, stat_funcs=None,
               extend=False, include_transformed=False,
               alpha=0.05, start=0, batches=None):
    R"""Create a data frame with summary statistics.

    Parameters
    ----------
    trace : MultiTrace instance
    varnames : list
        Names of variables to include in summary
    transform : callable
        Function to transform data (defaults to identity)
    stat_funcs : None or list
        A list of functions used to calculate statistics. By default,
        the mean, standard deviation, simulation standard error, and
        highest posterior density intervals are included.

        The functions will be given one argument, the samples for a
        variable as a 2 dimensional array, where the first axis
        corresponds to sampling iterations and the second axis
        represents the flattened variable (e.g., x__0, x__1,...). Each
        function should return either

        1) A `pandas.Series` instance containing the result of
           calculating the statistic along the first axis. The name
           attribute will be taken as the name of the statistic.
        2) A `pandas.DataFrame` where each column contains the
           result of calculating the statistic along the first axis.
           The column names will be taken as the names of the
           statistics.
    extend : boolean
        If True, use the statistics returned by `stat_funcs` in
        addition to, rather than in place of, the default statistics.
        This is only meaningful when `stat_funcs` is not None.
    include_transformed : bool
        Flag for reporting automatically transformed variables in addition
        to original variables (defaults to False).
    alpha : float
        The alpha level for generating posterior intervals. Defaults
        to 0.05. This is only meaningful when `stat_funcs` is None.
    start : int
        The starting index from which to summarize (each) chain. Defaults
        to zero.
    batches : None or int
        Batch size for calculating standard deviation for non-independent
        samples. Defaults to the smaller of 100 or the number of samples.
        This is only meaningful when `stat_funcs` is None.

    Returns
    -------
    `pandas.DataFrame` with summary statistics for each variable Defaults one
    are: `mean`, `sd`, `mc_error`, `hpd_2.5`, `hpd_97.5`, `n_eff` and `Rhat`.
    Last two are only computed for traces with 2 or more chains.

    Examples
    --------
    .. code:: ipython

        >>> import pymc3 as pm
        >>> trace.mu.shape
        (1000, 2)
        >>> pm.summary(trace, ['mu'])
                   mean        sd  mc_error     hpd_5    hpd_95
        mu__0  0.106897  0.066473  0.001818 -0.020612  0.231626
        mu__1 -0.046597  0.067513  0.002048 -0.174753  0.081924

                  n_eff      Rhat
        mu__0     487.0   1.00001
        mu__1     379.0   1.00203

    Other statistics can be calculated by passing a list of functions.

    .. code:: ipython

        >>> import pandas as pd
        >>> def trace_sd(x):
        ...     return pd.Series(np.std(x, 0), name='sd')
        ...
        >>> def trace_quantiles(x):
        ...     return pd.DataFrame(pm.quantiles(x, [5, 50, 95]))
        ...
        >>> pm.summary(trace, ['mu'], stat_funcs=[trace_sd, trace_quantiles])
                     sd         5        50        95
        mu__0  0.066473  0.000312  0.105039  0.214242
        mu__1  0.067513 -0.159097 -0.045637  0.062912
    """
    from .backends import tracetab as ttab

    if varnames is None:
        varnames = get_default_varnames(trace.varnames,
                       include_transformed=include_transformed)

    if batches is None:
        batches = min([100, len(trace)])

    funcs = [lambda x: pd.Series(np.mean(x, 0), name='mean'),
             lambda x: pd.Series(np.std(x, 0), name='sd'),
             lambda x: pd.Series(mc_error(x, batches), name='mc_error'),
             lambda x: _hpd_df(x, alpha)]

    if stat_funcs is not None:
        if extend:
            funcs = funcs + stat_funcs
        else:
            funcs = stat_funcs

    var_dfs = []
    for var in varnames:
        vals = transform(trace.get_values(var, burn=start, combine=True))
        flat_vals = vals.reshape(vals.shape[0], -1)
        var_df = pd.concat([f(flat_vals) for f in funcs], axis=1)
        var_df.index = ttab.create_flat_names(var, vals.shape[1:])
        var_dfs.append(var_df)
    dforg = pd.concat(var_dfs, axis=0)

    if (stat_funcs is not None) and (not extend):
        return dforg
    elif trace.nchains < 2:
        return dforg
    else:
        n_eff = pm.effective_n(trace,
                               varnames=varnames,
                               include_transformed=include_transformed)
        n_eff_pd = dict2pd(n_eff, 'n_eff')
        rhat = pm.gelman_rubin(trace,
                               varnames=varnames,
                               include_transformed=include_transformed)
        rhat_pd = dict2pd(rhat, 'Rhat')
        return pd.concat([dforg, n_eff_pd, rhat_pd],
                         axis=1, join_axes=[dforg.index])


def _calculate_stats(sample, batches, alpha):
    means = sample.mean(0)
    sds = sample.std(0)
    mces = mc_error(sample, batches)
    intervals = hpd(sample, alpha)
    for key, idxs in _groupby_leading_idxs(sample.shape[1:]):
        yield key
        for idx in idxs:
            mean, sd, mce = [stat[idx] for stat in (means, sds, mces)]
            interval = intervals[idx].squeeze().tolist()
            yield {'mean': mean, 'sd': sd, 'mce': mce, 'hpd': interval}


def _calculate_posterior_quantiles(sample, qlist):
    var_quantiles = quantiles(sample, qlist=qlist)
    # Replace ends of qlist with 'lo' and 'hi'
    qends = {qlist[0]: 'lo', qlist[-1]: 'hi'}
    qkeys = {q: qends[q] if q in qends else 'q{}'.format(q) for q in qlist}
    for key, idxs in _groupby_leading_idxs(sample.shape[1:]):
        yield key
        for idx in idxs:
            yield {qkeys[q]: var_quantiles[q][idx] for q in qlist}


def _groupby_leading_idxs(shape):
    """Group the indices for `shape` by the leading indices of `shape`.

    All dimensions except for the rightmost dimension are used to create
    groups.

    A 3d shape will be grouped by the indices for the two leading
    dimensions.

        >>> for key, idxs in _groupby_leading_idxs((3, 2, 2)):
        ...     print('key: {}'.format(key))
        ...     print(list(idxs))
        key: (0, 0)
        [(0, 0, 0), (0, 0, 1)]
        key: (0, 1)
        [(0, 1, 0), (0, 1, 1)]
        key: (1, 0)
        [(1, 0, 0), (1, 0, 1)]
        key: (1, 1)
        [(1, 1, 0), (1, 1, 1)]
        key: (2, 0)
        [(2, 0, 0), (2, 0, 1)]
        key: (2, 1)
        [(2, 1, 0), (2, 1, 1)]

    A 1d shape will only have one group.

        >>> for key, idxs in _groupby_leading_idxs((2,)):
        ...     print('key: {}'.format(key))
        ...     print(list(idxs))
        key: ()
        [(0,), (1,)]
    """
    idxs = itertools.product(*[range(s) for s in shape])
    return itertools.groupby(idxs, lambda x: x[:-1])


def bfmi(trace):
    R"""Calculate the estimated Bayesian fraction of missing information (BFMI).

    BFMI quantifies how well momentum resampling matches the marginal energy
    distribution.  For more information on BFMI, see
    https://arxiv.org/pdf/1604.00695.pdf.  The current advice is that values
    smaller than 0.2 indicate poor sampling.  However, this threshold is
    provisional and may change.  See
    http://mc-stan.org/users/documentation/case-studies/pystan_workflow.html
    for more information.

    Parameters
    ----------
    trace : result of an HMC/NUTS run, must contain energy information

    Returns
    -------
    z : float
        The Bayesian fraction of missing information of the model and trace.
    """
    energy = trace['energy']

    return np.square(np.diff(energy)).mean() / np.var(energy)


r2_r = namedtuple('r2_r', 'r2_median, r2_mean, r2_std')
def r2_score(y_true, y_pred, round_to=2):
    R"""R-squared for Bayesian regression models. Only valid for linear models.
    http://www.stat.columbia.edu/%7Egelman/research/unpublished/bayes_R2.pdf

    Parameters
    ----------
    y_true: : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.
    round_to : int
        Number of decimals used to round results (default 2).

    Returns
    -------
    `namedtuple` with the following elements:
    R2_median: median of the Bayesian R2
    R2_mean: mean of the Bayesian R2
    R2_std: standard deviation of the Bayesian R2
    """
    dimension = None
    if y_true.ndim > 1:
        dimension = 1

    var_y_est = np.var(y_pred, axis=dimension)
    var_e = np.var(y_true - y_pred, axis=dimension)

    r2 = var_y_est / (var_y_est + var_e)
    r2_median = np.around(np.median(r2), round_to)
    r2_mean = np.around(np.mean(r2), round_to)
    r2_std = np.around(np.std(r2), round_to)
    return r2_r(r2_median, r2_mean, r2_std)

