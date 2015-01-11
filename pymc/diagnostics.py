# Convergence diagnostics and model validation
import numpy as np
from . import Matplot, flib
from .utils import autocorr, autocov
from copy import copy
import pdb

try:
    from statsmodels.regression.linear_model import yule_walker
    has_sm = True
except ImportError:
    has_sm = False

from . import six
from .six import print_
xrange = six.moves.xrange

__all__ = [
    'geweke',
    'gelman_rubin',
    'raftery_lewis',
    'validate',
    'discrepancy',
    'iat',
    'ppp_value',
    'effective_n']


def open01(x, limit=1.e-6):
    """Constrain numbers to (0,1) interval"""
    try:
        return np.array([min(max(y, limit), 1. - limit) for y in x])
    except TypeError:
        return min(max(x, limit), 1. - limit)


class diagnostic(object):

    """
    This decorator allows for PyMC arguments of various types to be passed to
    the diagnostic functions. It identifies the type of object and locates its
    trace(s), then passes the data to the wrapped diagnostic function.

    """

    def __init__(self, all_chains=False):
        """ Initialize wrapper """

        self.all_chains = all_chains

    def __call__(self, f):

        def wrapped_f(pymc_obj, *args, **kwargs):

            # Figure out what type of object it is
            try:
                values = {}
                # First try Model type
                for variable in pymc_obj._variables_to_tally:
                    if self.all_chains:
                        k = pymc_obj.db.chains
                        data = [variable.trace(chain=i) for i in range(k)]
                    else:
                        data = variable.trace()
                    name = variable.__name__
                    if kwargs.get('verbose'):
                        print_("\nDiagnostic for %s ..." % name)
                    values[name] = f(data, *args, **kwargs)
                return values
            except AttributeError:
                pass

            try:
                # Then try Node type
                if self.all_chains:
                    k = pymc_obj.trace.db.chains
                    data = [pymc_obj.trace(chain=i) for i in range(k)]
                else:
                    data = pymc_obj.trace()
                name = pymc_obj.__name__
                return f(data, *args, **kwargs)
            except (AttributeError, ValueError):
                pass

            # If others fail, assume that raw data is passed
            return f(pymc_obj, *args, **kwargs)

        wrapped_f.__doc__ = f.__doc__
        wrapped_f.__name__ = f.__name__

        return wrapped_f


def validate(sampler, replicates=20, iterations=10000, burn=5000,
             thin=1, deterministic=False, db='ram', plot=True, verbose=0):
    """
    Model validation method, following Cook et al. (Journal of Computational and
    Graphical Statistics, 2006, DOI: 10.1198/106186006X136976).

    Generates posterior samples based on 'true' parameter values and data simulated
    from the priors. The quantiles of the parameter values are calculated, based on
    the samples. If the model is valid, the quantiles should be uniformly distributed
    over [0,1].

    Since this relies on the generation of simulated data, all data stochastics
    must have a valid random() method for validation to proceed.

    Parameters
    ----------
    sampler : Sampler
      An MCMC sampler object.
    replicates (optional) : int
      The number of validation replicates (i.e. number of quantiles to be simulated).
      Defaults to 100.
    iterations (optional) : int
      The number of MCMC iterations to be run per replicate. Defaults to 2000.
    burn (optional) : int
      The number of burn-in iterations to be run per replicate. Defaults to 1000.
    thin (optional) : int
      The thinning factor to be applied to posterior sample. Defaults to 1 (no thinning)
    deterministic (optional) : bool
      Flag for inclusion of deterministic nodes in validation procedure. Defaults
      to False.
    db (optional) : string
      The database backend to use for the validation runs. Defaults to 'ram'.
    plot (optional) : bool
      Flag for validation plots. Defaults to True.

    Returns
    -------
    stats : dict
      Return a dictionary containing tuples with the chi-square statistic and
      associated p-value for each data stochastic.

    Notes
    -----
    This function requires SciPy.
    """
    import scipy as sp
    # Set verbosity for models to zero
    sampler.verbose = 0

    # Specify parameters to be evaluated
    parameters = sampler.stochastics
    if deterministic:
        # Add deterministics to the mix, if requested
        parameters = parameters | sampler.deterministics

    # Assign database backend
    original_backend = sampler.db.__name__
    sampler._assign_database_backend(db)

    # Empty lists for quantiles
    quantiles = {}

    if verbose:
        print_("\nExecuting Cook et al. (2006) validation procedure ...\n")

    # Loop over replicates
    for i in range(replicates):

        # Sample from priors
        for p in sampler.stochastics:
            if not p.extended_parents:
                p.random()

        # Sample "true" data values
        for o in sampler.observed_stochastics:
            # Generate simuated data for data stochastic
            o.set_value(o.random(), force=True)
            if verbose:
                print_("Data for %s is %s" % (o.__name__, o.value))

        param_values = {}
        # Record data-generating parameter values
        for s in parameters:
            param_values[s] = s.value

        try:
            # Fit models given parameter values
            sampler.sample(iterations, burn=burn, thin=thin)

            for s in param_values:

                if not i:
                    # Initialize dict
                    quantiles[s.__name__] = []
                trace = s.trace()
                q = sum(trace < param_values[s], 0) / float(len(trace))
                quantiles[s.__name__].append(open01(q))

            # Replace data values
            for o in sampler.observed_stochastics:
                o.revert()

        finally:
            # Replace data values
            for o in sampler.observed_stochastics:
                o.revert()

            # Replace backend
            sampler._assign_database_backend(original_backend)

        if not i % 10 and i and verbose:
            print_("\tCompleted validation replicate", i)

    # Replace backend
    sampler._assign_database_backend(original_backend)

    stats = {}
    # Calculate chi-square statistics
    for param in quantiles:
        q = quantiles[param]
        # Calculate chi-square statistics
        X2 = sum(sp.special.ndtri(q) ** 2)
        # Calculate p-value
        p = sp.special.chdtrc(replicates, X2)

        stats[param] = (X2, p)

    if plot:
        # Convert p-values to z-scores
        p = copy(stats)
        for i in p:
            p[i] = p[i][1]
        Matplot.zplot(p, verbose=verbose)

    return stats

def spec(x, order=2):
    
    beta, sigma = yule_walker(x, order)
    return sigma**2 / (1. - np.sum(beta))**2

@diagnostic()
def geweke(x, first=.1, last=.5, intervals=20, maxlag=20):
    """Return z-scores for convergence diagnostics.

    Compare the mean of the first % of series with the mean of the last % of
    series. x is divided into a number of segments for which this difference is
    computed. If the series is converged, this score should oscillate between
    -1 and 1.

    Parameters
    ----------
    x : array-like
      The trace of some stochastic parameter.
    first : float
      The fraction of series at the beginning of the trace.
    last : float
      The fraction of series at the end to be compared with the section
      at the beginning.
    intervals : int
      The number of segments.
    maxlag : int
      Maximum autocorrelation lag for estimation of spectral variance

    Returns
    -------
    scores : list [[]]
      Return a list of [i, score], where i is the starting index for each
      interval and score the Geweke score on the interval.

    Notes
    -----

    The Geweke score on some series x is computed by:

      .. math:: \frac{E[x_s] - E[x_e]}{\sqrt{V[x_s] + V[x_e]}}

    where :math:`E` stands for the mean, :math:`V` the variance,
    :math:`x_s` a section at the start of the series and
    :math:`x_e` a section at the end of the series.

    References
    ----------
    Geweke (1992)
    """
    
    if not has_sm:
        print("statsmodels not available. Geweke diagnostic cannot be calculated.")
        return

    if np.ndim(x) > 1:
        return [geweke(y, first, last, intervals) for y in np.transpose(x)]

    # Filter out invalid intervals
    if first + last >= 1:
        raise ValueError(
            "Invalid intervals for Geweke convergence analysis",
            (first, last))

    # Initialize list of z-scores
    zscores = [None] * intervals

    # Starting points for calculations
    starts = np.linspace(0, int(len(x)*(1.-last)), intervals).astype(int)

    # Loop over start indices
    for i,s in enumerate(starts):

        # Size of remaining array
        x_trunc = x[s:]
        n = len(x_trunc)

        # Calculate slices
        first_slice = x_trunc[:int(first * n)]
        last_slice = x_trunc[int(last * n):]

        z = (first_slice.mean() - last_slice.mean())
        z /= np.sqrt(spec(first_slice)/len(first_slice) +
                     spec(last_slice)/len(last_slice))
        zscores[i] = len(x) - n, z

    return zscores

# From StatLib -- gibbsit.f


@diagnostic()
def raftery_lewis(x, q, r, s=.95, epsilon=.001, verbose=1):
    """
    Return the number of iterations needed to achieve a given
    precision.

    :Parameters:
        x : sequence
            Sampled series.
        q : float
            Quantile.
        r : float
            Accuracy requested for quantile.
        s (optional) : float
            Probability of attaining the requested accuracy (defaults to 0.95).
        epsilon (optional) : float
             Half width of the tolerance interval required for the q-quantile (defaults to 0.001).
        verbose (optional) : int
            Verbosity level for output (defaults to 1).

    :Return:
        nmin : int
            Minimum number of independent iterates required to achieve
            the specified accuracy for the q-quantile.
        kthin : int
            Skip parameter sufficient to produce a first-order Markov
            chain.
        nburn : int
            Number of iterations to be discarded at the beginning of the
            simulation, i.e. the number of burn-in iterations.
        nprec : int
            Number of iterations not including the burn-in iterations which
            need to be obtained in order to attain the precision specified
            by the values of the q, r and s input parameters.
        kmind : int
            Minimum skip parameter sufficient to produce an independence
            chain.

    :Example:
        >>> raftery_lewis(x, q=.025, r=.005)

    :Reference:
        Raftery, A.E. and Lewis, S.M. (1995).  The number of iterations,
        convergence diagnostics and generic Metropolis algorithms.  In
        Practical Markov Chain Monte Carlo (W.R. Gilks, D.J. Spiegelhalter
        and S. Richardson, eds.). London, U.K.: Chapman and Hall.

        See the fortran source file `gibbsit.f` for more details and references.
    """
    if np.ndim(x) > 1:
        return [raftery_lewis(y, q, r, s, epsilon, verbose)
                for y in np.transpose(x)]

    output = nmin, kthin, nburn, nprec, kmind = flib.gibbmain(
        x, q, r, s, epsilon)

    if verbose:

        print_("\n========================")
        print_("Raftery-Lewis Diagnostic")
        print_("========================")
        print_()
        print_(
            "%s iterations required (assuming independence) to achieve %s accuracy with %i percent probability." %
            (nmin, r, 100 * s))
        print_()
        print_(
            "Thinning factor of %i required to produce a first-order Markov chain." %
            kthin)
        print_()
        print_(
            "%i iterations to be discarded at the beginning of the simulation (burn-in)." %
            nburn)
        print_()
        print_("%s subsequent iterations required." % nprec)
        print_()
        print_(
            "Thinning factor of %i required to produce an independence chain." %
            kmind)

    return output


def batch_means(x, f=lambda y: y, theta=.5, q=.95, burn=0):
    """
    TODO: Use Bayesian CI.

    Returns the half-width of the frequentist confidence interval
    (q'th quantile) of the Monte Carlo estimate of E[f(x)].

    :Parameters:
        x : sequence
            Sampled series. Must be a one-dimensional array.
        f : function
            The MCSE of E[f(x)] will be computed.
        theta : float between 0 and 1
            The batch length will be set to len(x) ** theta.
        q : float between 0 and 1
            The desired quantile.

    :Example:
        >>>batch_means(x, f=lambda x: x**2, theta=.5, q=.95)

    :Reference:
        Flegal, James M. and Haran, Murali and Jones, Galin L. (2007).
        Markov chain Monte Carlo: Can we trust the third significant figure?
        <Publication>

    :Note:
        Requires SciPy
    """

    try:
        import scipy
        from scipy import stats
    except ImportError:
        raise ImportError('SciPy must be installed to use batch_means.')

    x = x[burn:]

    n = len(x)

    b = np.int(n ** theta)
    a = n / b

    t_quant = stats.t.isf(1 - q, a - 1)

    Y = np.array([np.mean(f(x[i * b:(i + 1) * b])) for i in xrange(a)])
    sig = b / (a - 1.) * sum((Y - np.mean(f(x))) ** 2)

    return t_quant * sig / np.sqrt(n)


def discrepancy(observed, simulated, expected):
    """Calculates Freeman-Tukey statistics (Freeman and Tukey 1950) as
    a measure of discrepancy between observed and r replicates of simulated data. This
    is a convenient method for assessing goodness-of-fit (see Brooks et al. 2000).

    D(x|\theta) = \sum_j (\sqrt{x_j} - \sqrt{e_j})^2

    :Parameters:
      observed : Iterable of observed values (size=(n,))
      simulated : Iterable of simulated values (size=(r,n))
      expected : Iterable of expected values (size=(r,) or (r,n))

    :Returns:
      D_obs : Discrepancy of observed values
      D_sim : Discrepancy of simulated values

    """
    try:
        simulated = simulated.astype(float)
    except AttributeError:
        simulated = simulated.trace().astype(float)
    try:
        expected = expected.astype(float)
    except AttributeError:
        expected = expected.trace().astype(float)
    # Ensure expected values are rxn
    expected = np.resize(expected, simulated.shape)

    D_obs = np.sum([(np.sqrt(observed) - np.sqrt(
        e)) ** 2 for e in expected], 1)
    D_sim = np.sum(
        [(np.sqrt(s) - np.sqrt(e)) ** 2 for s,
         e in zip(simulated,
                  expected)],
        1)

    # Print p-value
    count = sum(s > o for o, s in zip(D_obs, D_sim))
    print_('Bayesian p-value: p=%.3f' % (1. * count / len(D_obs)))

    return D_obs, D_sim
    

@diagnostic(all_chains=True)
def effective_n(x):
    """ Returns estimate of the effective sample size of a set of traces.

    Parameters
    ----------
    x : array-like
      An array containing the 2 or more traces of a stochastic parameter. That is, an array of dimension m x n x k, where m is the number of traces, n the number of samples, and k the dimension of the stochastic.
    
    Returns
    -------
    n_eff : float
      Return the effective sample size, :math:`\hat{n}_{eff}`

    Notes
    -----

    The diagnostic is computed by:

      .. math:: \hat{n}_{eff} = \frac{mn}}{1 + 2 \sum_{t=1}^T \hat{\rho}_t}

    where :math:`\hat{\rho}_t` is the estimated autocorrelation at lag t, and T
    is the first odd positive integer for which the sum :math:`\hat{\rho}_{T+1} + \hat{\rho}_{T+1}` 
    is negative.

    References
    ----------
    Gelman et al. (2014)"""
    
    if np.shape(x) < (2,):
        raise ValueError(
            'Calculation of effective sample size requires multiple chains of the same length.')

    try:
        m, n = np.shape(x)
    except ValueError:
        return [effective_n(np.transpose(y)) for y in np.transpose(x)]
        
    s2 = gelman_rubin(x, return_var=True)
    
    negative_autocorr = False
    t = 1
    
    variogram = lambda t: (sum(sum((x[j][i] - x[j][i-t])**2 for i in range(t,n)) for j in range(m)) 
                                / (m*(n - t)))
    rho = np.ones(n)
    # Iterate until the sum of consecutive estimates of autocorrelation is negative
    while not negative_autocorr and (t < n):
        
        rho[t] = 1. - variogram(t)/(2.*s2)
        
        if not t % 2:
            negative_autocorr = sum(rho[t-1:t+1]) < 0
        
        t += 1
        
    return int(m*n / (1 + 2*rho[1:t].sum()))
    

@diagnostic(all_chains=True)
def gelman_rubin(x, return_var=False):
    """ Returns estimate of R for a set of traces.

    The Gelman-Rubin diagnostic tests for lack of convergence by comparing
    the variance between multiple chains to the variance within each chain.
    If convergence has been achieved, the between-chain and within-chain
    variances should be identical. To be most effective in detecting evidence
    for nonconvergence, each chain should have been initialized to starting
    values that are dispersed relative to the target distribution.

    Parameters
    ----------
    x : array-like
      An array containing the 2 or more traces of a stochastic parameter. That is, an array of dimension m x n x k, where m is the number of traces, n the number of samples, and k the dimension of the stochastic.
      
    return_var : bool
      Flag for returning the marginal posterior variance instead of R-hat (defaults of False).

    Returns
    -------
    Rhat : float
      Return the potential scale reduction factor, :math:`\hat{R}`

    Notes
    -----

    The diagnostic is computed by:

      .. math:: \hat{R} = \frac{\hat{V}}{W}

    where :math:`W` is the within-chain variance and :math:`\hat{V}` is
    the posterior variance estimate for the pooled traces. This is the
    potential scale reduction factor, which converges to unity when each
    of the traces is a sample from the target posterior. Values greater
    than one indicate that one or more chains have not yet converged.

    References
    ----------
    Brooks and Gelman (1998)
    Gelman and Rubin (1992)"""

    if np.shape(x) < (2,):
        raise ValueError(
            'Gelman-Rubin diagnostic requires multiple chains of the same length.')

    try:
        m, n = np.shape(x)
    except ValueError:
        return [gelman_rubin(np.transpose(y)) for y in np.transpose(x)]

    # Calculate between-chain variance
    B_over_n = np.sum((np.mean(x, 1) - np.mean(x)) ** 2) / (m - 1)

    # Calculate within-chain variances
    W = np.sum(
        [(x[i] - xbar) ** 2 for i,
         xbar in enumerate(np.mean(x,
                                   1))]) / (m * (n - 1))

    # (over) estimate of variance
    s2 = W * (n - 1) / n + B_over_n
    
    if return_var:
        return s2

    # Pooled posterior variance estimate
    V = s2 + B_over_n / m

    # Calculate PSRF
    R = V / W

    return R


def _find_max_lag(x, rho_limit=0.05, maxmaxlag=20000, verbose=0):
    """Automatically find an appropriate maximum lag to calculate IAT"""

    # Fetch autocovariance matrix
    acv = autocov(x)
    # Calculate rho
    rho = acv[0, 1] / acv[0, 0]

    lam = -1. / np.log(abs(rho))

    # Initial guess at 1.5 times lambda (i.e. 3 times mean life)
    maxlag = int(np.floor(3. * lam)) + 1

    # Jump forward 1% of lambda to look for rholimit threshold
    jump = int(np.ceil(0.01 * lam)) + 1

    T = len(x)

    while ((abs(rho) > rho_limit) & (maxlag < min(T / 2, maxmaxlag))):

        acv = autocov(x, maxlag)
        rho = acv[0, 1] / acv[0, 0]
        maxlag += jump

    # Add 30% for good measure
    maxlag = int(np.floor(1.3 * maxlag))

    if maxlag >= min(T / 2, maxmaxlag):
        maxlag = min(min(T / 2, maxlag), maxmaxlag)
        "maxlag fixed to %d" % maxlag
        return maxlag

    if maxlag <= 1:
        print_("maxlag = %d, fixing value to 10" % maxlag)
        return 10

    if verbose:
        print_("maxlag = %d" % maxlag)
    return maxlag


def _cut_time(gammas):
    """Support function for iat().
    Find cutting time, when gammas become negative."""

    for i in range(len(gammas) - 1):

        if not ((gammas[i + 1] > 0.0) & (gammas[i + 1] < gammas[i])):
            return i

    return i


@diagnostic()
def iat(x, maxlag=None):
    """Calculate the integrated autocorrelation time (IAT), given the trace from a Stochastic."""

    if not maxlag:
        # Calculate maximum lag to which autocorrelation is calculated
        maxlag = _find_max_lag(x)

    acr = [autocorr(x, lag) for lag in range(1, maxlag + 1)]

    # Calculate gamma values
    gammas = [(acr[2 * i] + acr[2 * i + 1]) for i in range(maxlag // 2)]

    cut = _cut_time(gammas)

    if cut + 1 == len(gammas):
        print_("Not enough lag to calculate IAT")

    return np.sum(2 * gammas[:cut + 1]) - 1.0


@diagnostic()
def ppp_value(simdata, trueval, round=3):
    """
    Calculates posterior predictive p-values on data simulated from the posterior
     predictive distribution, returning the quantile of the observed data relative to
     simulated.

     The posterior predictive p-value is computed by:

       .. math:: Pr(T(y^{\text{sim}} > T(y) | y)

     where T is a test statistic of interest and :math:`y^{\text{sim}}` is the simulated
     data.

    :Arguments:
        simdata: array or PyMC object
            Trace of simulated data or the PyMC stochastic object containing trace.

        trueval: numeric
            True (observed) value of the data

        round: int
            Rounding of returned quantile (defaults to 3)

    """

    if ndim(trueval) == 1 and ndim(simdata == 2):
        # Iterate over more than one set of data
        return [post_pred_checks(simdata[:, i], trueval[i])
                for i in range(len(trueval))]

    return (simdata > trueval).mean()
