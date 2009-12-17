# Convergence diagnostics and model validation
# Heidelberger and Welch (1983) ?

__all__ = ['geweke', 'gelman_rubin', 'raftery_lewis', 'validate', 'discrepancy']

import numpy as np
import scipy as sp
import pymc
from copy import copy
import pdb

def open01(x, limit=1.e-6):
    """Constrain numbers to (0,1) interval"""
    try:
        return np.array([min(max(y, limit), 1.-limit) for y in x])
    except TypeError:
        return min(max(x, limit), 1.-limit)

def diagnostic(f):
    """
    This decorator allows for PyMC arguments of various types to be passed to
    the diagnostic functions. It identifies the type of object and locates its
    trace(s), then passes the data to the wrapped diagnostic function.

    """

    def wrapper(pymc_obj, *args, **kwargs):

        # Figure out what type of object it is
        try:
            values = {}
            # First try Model type
            for variable in pymc_obj._variables_to_tally:
                data = variable.trace()
                name = variable.__name__
                print "\nDiagnostic for %s ..." % name
                values[name] = f(data, *args, **kwargs)
            return values
        except AttributeError:
            pass

        try:
            # Then try Node type
            data = pymc_obj.trace()
            name = pymc_obj.__name__
            return f(data, *args, **kwargs)
        except (AttributeError,ValueError):
            pass

        # If others fail, assume that raw data is passed
        return f(pymc_obj, *args, **kwargs)

    return wrapper


def validate(sampler, replicates=20, iterations=10000, burn=5000, thin=1, deterministic=False, db='ram', plot=True, verbose=0):
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
    """

    # Set verbosity for models to zero
    sampler.verbose = 0

    # Specify parameters to be evaluated
    parameters = sampler.stochastics
    if deterministic:
        # Add deterministics to the mix, if requested
        parameters = parameters.union(sampler.deterministics)

    # Assign database backend
    original_backend = sampler.db.__name__
    sampler._assign_database_backend(db)

    # Empty lists for quantiles
    quantiles = {}

    if verbose:
        print "\nExecuting Cook et al. (2006) validation procedure ...\n"

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
                print "Data for %s is %s" % (o.__name__, o.value)

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
                q = sum(trace<param_values[s], 0)/float(len(trace))
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
            print "\tCompleted validation replicate", i


    # Replace backend
    sampler._assign_database_backend(original_backend)

    stats = {}
    # Calculate chi-square statistics
    for param in quantiles:
        q = quantiles[param]
        # Calculate chi-square statistics
        X2 = sum(sp.special.ndtri(q)**2)
        # Calculate p-value
        p = sp.special.chdtrc(replicates, X2)

        stats[param] = (X2, p)

    if plot:
        # Convert p-values to z-scores
        p = copy(stats)
        for i in p:
            p[i] = p[i][1]
        pymc.Matplot.zplot(p, verbose=verbose)

    return stats


@diagnostic
def geweke(x, first=.1, last=.5, intervals=20):
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
    # Filter out invalid intervals
    if first + last >= 1:
        raise "Invalid intervals for Geweke convergence analysis",(first,last)

    # Initialize list of z-scores
    zscores = []

    # Last index value
    end = len(x) - 1

    # Calculate starting indices
    sindices = np.arange(0, end/2, step = int((end / 2) / (intervals-1)))

    # Loop over start indices
    for start in sindices:

        # Calculate slices
        first_slice = x[start : start + int(first * (end - start))]
        last_slice = x[int(end - last * (end - start)):]

        z = (first_slice.mean() - last_slice.mean())
        z /= np.sqrt(first_slice.std()**2 + last_slice.std()**2)

        zscores.append([start, z])

    if intervals == None:
        return zscores[0]
    else:
        return zscores

# From StatLib -- gibbsit.f
@diagnostic
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
        s (optional): float
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
    output = nmin, kthin, nburn, nprec, kmind = pymc.flib.gibbmain(x, q, r, s, epsilon)

    if verbose:

        print "\n========================"
        print "Raftery-Lewis Diagnostic"
        print "========================"
        print
        print "%s iterations required (assuming independence) to achieve %s accuracy with %i percent probability." % (nmin, r, 100*s)
        print
        print "Thinning factor of %i required to produce a first-order Markov chain." % kthin
        print
        print "%i iterations to be discarded at the beginning of the simulation (burn-in)." % nburn
        print
        print "%s subsequent iterations required." % nprec
        print
        print "Thinning factor of %i required to produce an independence chain." % kmind

    return output

def batch_means(x, f=lambda y:y, theta=.5, q=.95, burn=0):
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
        raise ImportError, 'SciPy must be installed to use batch_means.'

    x=x[burn:]

    n = len(x)

    b = np.int(n**theta)
    a = n/b

    t_quant = stats.t.isf(1-q,a-1)

    Y = np.array([np.mean(f(x[i*b:(i+1)*b])) for i in xrange(a)])
    sig = b / (a-1.) * sum((Y - np.mean(f(x))) ** 2)

    return t_quant * sig / np.sqrt(n)

def discrepancy(observed, simulated, expected):
    """Calculates Freeman-Tukey statistics (Freeman and Tukey 1950) as
    a measure of discrepancy between observed and simulated data. This
    is a convenient method for assessing goodness-of-fit (see Brooks et al. 2000).

    D(x|\theta) = \sum_j (\sqrt{x_j} - \sqrt{e_j})^2

    :Parameters:
      observed : Iterable of observed values
      simulated : Iterable of simulated values
      expected : Iterable of expected values

    :Returns:
      D_obs : Discrepancy of observed values
      D_sim : Discrepancy of simulated values

    """
    try:
        simulated = simulated.trace()
    except:
        pass    
    try:
        expected = expected.trace()
    except:
        pass
    
    D_obs = np.sum([(np.sqrt(observed)-np.sqrt(e))**2 for e in expected], 1)
    D_sim = np.sum([(np.sqrt(s)-np.sqrt(e))**2 for s,e in zip(simulated, expected)], 1)
    
    # Print p-value
    count = sum(s>o for o,s in zip(D_obs,D_sim))
    print 'Bayesian p-value: p=%.3f' % (1.*count/len(D_obs))

    return D_obs, D_sim


def gelman_rubin(x):
    raise NotImplementedError
# x contains multiple chains
# Transform positive or [0,1] variables using a logarithmic/logittranformation.
#
