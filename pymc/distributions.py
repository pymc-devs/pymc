#-------------------------------------------------------------------
# Decorate fortran functions from pymc.flib to ease argument passing
#-------------------------------------------------------------------
# TODO: Deal with functions that take correlation matrices as arguments.wishart, normal,?
# TODO: test and finalize vectorized multivariate normal like.
# TODO: Add exponweib_expval
# TODO: Complete docstrings with LaTeX formulas from the tutorial.
# TODO: categorical, mvhypergeometric

__docformat__='reStructuredText'

import flib
import pymc
import numpy as np
from Node import ZeroProbability
from PyMCObjects import Stochastic
from numpy import Inf, random, sqrt, log, size, tan, pi, shape, ravel, prod, any, sum
import pdb

if hasattr(flib, 'cov_mvnorm'):
    flib_blas_OK = True
else:
    flib_blas_OK = False


# Import utility functions
import inspect, types
from copy import copy
random_number = random.random
inverse = np.linalg.pinv

class ArgumentError(AttributeError):
    """Incorrect class argument"""
    pass

univ_distributions = ['bernoulli', 'beta', 'binomial', 'cauchy', 'chi2',
'exponential', 'exponweib', 'gamma', 'geometric', 'half_normal', 'hypergeometric',
'inverse_gamma', 'lognormal', 'negative_binomial', 'normal', 'poisson', 'uniform',
'weibull']

if flib_blas_OK:
    mv_distributions = ['dirichlet','multivariate_hypergeometric','mv_normal','mv_normal_cov','mv_normal_chol','multinomial', 'wishart','wishart_cov']
else:
    mv_distributions = ['dirichlet','multivariate_hypergeometric','multinomial']

discrete_distributions = ['binomial', 'poisson', 'multinomial']
continuous_distributions = univ_distributions + mv_distributions
for non_cont in discrete_distributions + ['bernoulli']:
    continuous_distributions.remove(non_cont)

availabledistributions = univ_distributions+mv_distributions

# Changes lower case, underscore-separated names into "Class style" capitalized names
# For example, 'negative_binomial' becomes 'NegativeBinomial'
capitalize = lambda name: ''.join([s.capitalize() for s in name.split('_')])


# ============================================================================================
# = User-accessible function to convert a logp and random function to a Stochastic subclass. =
# ============================================================================================


def new_dist_class(*new_class_args):
    """
    Returns a new class from a distribution.
    """
    (dtype, name, parent_names, parents_default, docstr, logp, random) = new_class_args
    class new_class(Stochastic):
        __doc__ = docstr
        def __init__(self, *args, **kwds):
            (dtype, name, parent_names, parents_default, docstr, logp, random) = new_class_args
            parents=parents_default
                        
            # Figure out what argument names are needed.
            arg_keys = ['name', 'parents', 'value', 'trace', 'rseed', 'doc', 'isdata', 'debug']
            arg_vals = [None, parents, None, True, True, None, False, False]
            arg_dict_out = dict(zip(arg_keys, arg_vals))
            args_needed = ['name'] + parent_names + arg_keys[2:]
            
            # Sort positional arguments
            for i in xrange(len(args)):
                try:
                    k = args_needed.pop(0)
                    if k in parent_names:
                        parents[k] = args[i]
                    else:
                        arg_dict_out[k] = args[i]
                except:
                    raise ValueError, 'Too many positional arguments provided. Arguments for class ' + self.__class__.__name__ + ' are: ' + str(all_args_needed)
                        

            # Sort keyword arguments
            for k in args_needed:
                if k in parent_names:
                    try:
                        parents[k] = kwds.pop(k)
                    except:
                        if k in parents_default:
                            parents[k] = parents_default[k]
                        else:
                            raise ValueError, 'No value given for parent ' + k
                elif k in arg_dict_out.keys():
                    try:
                        arg_dict_out[k] = kwds.pop(k)
                    except:
                        pass
                else:
                    raise ValueError, 'Keyword '+ k + ' not recognized. Arguments recognized are ' + str(args_needed)
            
            # Determine size and shape of value
            # size = arg_dict_out['size']
            # if size is not None:
            #     def random_wrapper_with_size(random=random,size=size,**parents):
            #         return random(size=size,**parents)
            #     random_wrapper_with_size.__name__ = random.__name__
            #     random = random_wrapper_with_size
            # 
            # arg_dict_out.pop('size')        

            # Call base class initialization method
            if arg_dict_out.pop('debug'):
                logp = debug_wrapper(logp)
                random = debug_wrapper(random)
            else:
                Stochastic.__init__(self, logp=logp, random=random, dtype=dtype, **arg_dict_out)

    new_class.__name__ = name
    new_class.parent_names = parent_names
    
    return new_class


def stochastic_from_dist(name, logp, random=None, dtype=np.float):
    """
    Return a Stochastic subclass made from a particular distribution.

      :Example:
        >>> Exponential = stochastic_from_dist('exponential')
        >>> A = Exponential(self_name, value, beta)
        
    Arguments can be passed in positionally; in this case, argument order is: self_name, parents.
    value is set to None by default, and can only be provided as a keyword argument.
    """
    
    (args, varargs, varkw, defaults) = inspect.getargspec(logp)
    parent_names = args[1:]
    try:
        parents_default = dict(zip(args[-len(defaults):], defaults))
    except TypeError: # No parents at all.
        parents_default = {}
    
    name = capitalize(name)
    
    # Build docstring from distribution
    docstr = name[0]+' = '+name + '(name, '+', '.join(parent_names)+', value=None, trace=True, rseed=True, doc=None)\n\n'
    docstr += 'Stochastic variable with '+name+' distribution.\nParents are: '+', '.join(parent_names) + '.\n\n'
    docstr += 'Docstring of log-probability function:\n'
    docstr += logp.__doc__

    logp=valuewrapper(logp)

    return new_dist_class(dtype, name, parent_names, parents_default, docstr, logp, random)


#-------------------------------------------------------------
# Light decorators
#-------------------------------------------------------------

def Vectorize(f):
    """
    Wrapper to vectorize a scalar function.
    """

    return np.vectorize(f)

def randomwrap(func):
    """
    Decorator for random value generators

    Allows passing of sequence of parameters, as well as a size argument.


    Convention:

      - If size=1 and the parameters are all scalars, return a scalar.
      - If size=1, the random variates are 1D.
      - If the parameters are scalars and size > 1, the random variates are 1D.
      - If size > 1 and the parameters are sequences, the random variates are
        aligned as (size, max(length)), where length is the parameters size.


    :Example:
      >>> rbernoulli(.1)
      0
      >>> rbernoulli([.1,.9])
      asarray([0, 1])
      >>> rbernoulli(.9, size=2)
      asarray([1, 1])
      >>> rbernoulli([.1,.9], 2)
      asarray([[0, 1],
             [0, 1]])
    """


    # Find the order of the arguments.
    refargs, varargs, varkw, defaults = inspect.getargspec(func)
    #vfunc = np.vectorize(self.func)
    npos = len(refargs)-len(defaults) # Number of pos. arg.
    nkwds = len(defaults) # Number of kwds args.
    mv = func.__name__[1:] in mv_distributions

    def wrapper(*args, **kwds):
        
        # First transform keyword arguments into positional arguments.
        n = len(args)
        if nkwds > 0:
            args = list(args)
            for i,k in enumerate(refargs[n:]):
                if k in kwds.keys():
                    args.append(kwds[k])
                else:
                    args.append(defaults[n-npos+i])

        r = [];s=[];largs=[];nr = args[-1]
        length = [np.atleast_1d(a).shape[0] for a in args]
        dimension = [np.atleast_1d(a).ndim for a in args]
        N = max(length)
        if len(set(dimension))>2:
            raise 'Dimensions do not agree.'
        # Make sure all elements are iterable and have consistent lengths, ie
        # 1 or n, but not m and n.

        for arg, s in zip(args, length):
            t = type(arg)
            arr = np.empty(N, type)
            if s == 1:
                arr.fill(arg)
            elif s == N:
                arr = np.asarray(arg)
            else:
                raise 'Arguments size not allowed.', s
            largs.append(arr)

        if mv and N >1 and max(dimension)>1 and nr>1:
            raise 'Multivariate distributions cannot take s>1 and multiple values.'

        if mv:
            for i, arg in enumerate(largs[:-1]):
                largs[0] = np.atleast_2d(arg)

        for arg in zip(*largs):
            r.append(func(*arg))

        size = arg[-1]
        vec_stochastics = len(r)>1
        if mv:
            if nr == 1 and N==1:
                return r[0]
            else:
                return np.vstack(r)
        else:
            if size > 1 and vec_stochastics:
                return np.atleast_2d(r).transpose()
            elif vec_stochastics or size > 1:
                return np.concatenate(r)
            else: # Scalar case
                return r[0][0]

    wrapper.__doc__ = func.__doc__
    return wrapper
    
def debugwrapper(func, name):
    # Wrapper to debug distributions
    
    import pdb
    
    def wrapper(*args, **kwargs):
        
        print 'Debugging inside %s:' % name
        print '\tPress \'s\' to step into function for debugging' 
        print '\tCall \'args\' to list function arguments'
        
        # Set debugging trace
        pdb.set_trace()
                
        # Call function
        return func(*args, **kwargs)
    
    return wrapper


#-------------------------------------------------------------
# Utility functions
#-------------------------------------------------------------

def constrain(value, lower=-Inf, upper=Inf, allow_equal=False):
    """
    Apply interval constraint on stochastic value.
    """

    ok = flib.constrain(value, lower, upper, allow_equal)
    if ok == 0:
        raise ZeroProbability

def standardize(x, loc=0, scale=1):
    """
    Standardize x

    Return (x-loc)/scale
    """

    return flib.standardize(x,loc,scale)

# ==================================
# = vectorize causes memory leaks. =
# ==================================
# @Vectorize
def gammaln(x):
    """
    Logarithm of the Gamma function
    """

    return flib.gamfun(x)

def expand_triangular(X,k):
    """
    Expand flattened triangular matrix.
    """

    X = X.tolist()
    # Unflatten matrix
    Y = np.asarray([[0] * i + X[i * k - (i * (i - 1)) / 2 : i * k + (k - i)] for i in range(k)])
    # Loop over rows
    for i in range(k):
        # Loop over columns
        for j in range(k):
            Y[j, i] = Y[i, j]
    return Y


# Loss functions

absolute_loss = lambda o,e: absolute(o - e)

squared_loss = lambda o,e: (o - e)**2

chi_square_loss = lambda o,e: (1.*(o - e)**2)/e

def GOFpoints(x,y,expval,loss):
    return sum(np.transpose([loss(x, expval), loss(y, expval)]), 0)

#--------------------------------------------------------
# Statistical distributions
# random generator, expval, log-likelihood
#--------------------------------------------------------

# Autoregressive lognormal 
def rarlognormal(a, sigma, rho, size=1):
    """rarnormal(a, sigma, rho)
    
    Autoregressive normal random variates.
    
    If a is a scalar, generates one series of length size. 
    If a is a sequence, generates size series of the same length
    as a. 
    """
    f = pymc.utils.ar1
    if np.isscalar(a):
        r = f(rho, 0, sigma, size)
    else:
        n = len(a)
        r = [f(rho, 0, sigma, n) for i in range(size)]
        if size == 1:
            r = r[0]
    return a*np.exp(r)
            

def arlognormal_like(x, a, sigma, rho):
    r"""arnormal(x, a, sigma, rho, beta=1)
    
    Autoregressive lognormal log-likelihood.
    
    .. math::
        x_i & = a_i \exp(e_i) \\
        e_i & = \rho e_{i-1} + \epsilon_i
    
    where :math:'\epsilon_i \sim N(0,\sigma)'.
    """
    return flib.arlognormal(x, np.log(a), sigma, rho, beta=1)
    

# Bernoulli----------------------------------------------
@randomwrap
def rbernoulli(p,size=1):
    """
    rbernoulli(p,size=1)

    Random Bernoulli variates.
    """

    return random.random(size)<p

def bernoulli_expval(p):
    """
    bernoulli_expval(p)

    Expected value of bernoulli distribution.
    """

    return p


def bernoulli_like(x, p):
    r"""
    bernoulli_like(x, p)

    Bernoulli log-likelihood

    The Bernoulli distribution describes the probability of successes (x=1) and
    failures (x=0).

    .. math::
        f(x \mid p) = p^{x- 1} (1-p)^{1-x}

    :Parameters:
      - 'x': Series of successes (1) and failures (0). :math:'x=0,1'
      - 'p': Probability of success. :math:'0 < p < 1'

    :Example:
      >>> bernoulli_like([0,1,0,1], .4)
      -2.8542325496673584

    :Note:
      - :math:'E(x)= p'
      - :math:'Var(x)= p(1-p)'

    """
    
    return flib.bernoulli(x, p)


# Beta----------------------------------------------
@randomwrap
def rbeta(alpha, beta, size=1):
    """
    rbeta(alpha, beta, size=1)

    Random beta variates.
    """

    return random.beta(alpha, beta,size)

def beta_expval(alpha, beta):
    """
    beta_expval(alpha, beta)

    Expected value of beta distribution.
    """

    return 1.0 * alpha / (alpha + beta)


def beta_like(x, alpha, beta):
    r"""
    beta_like(x, alpha, beta)

    Beta log-likelihood.

    .. math::
        f(x \mid \alpha, \beta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha) \Gamma(\beta)} x^{\alpha - 1} (1 - x)^{\beta - 1}

    :Parameters:
      - 'x': 0 < x < 1
      - 'alpha': > 0
      - 'beta': > 0

    :Example:
      >>> beta_like(.4,1,2)
      0.18232160806655884

    :Note:
      - :math:'E(X)=\frac{\alpha}{\alpha+\beta}'
      - :math:'Var(X)=\frac{\alpha \beta}{(\alpha+\beta)^2(\alpha+\beta+1)}'

    """
    # try:
    #     constrain(alpha, lower=0, allow_equal=True)
    #     constrain(beta, lower=0, allow_equal=True)
    #     constrain(x, 0, 1, allow_equal=True)
    # except ZeroProbability:
    #     return -Inf
    return flib.beta_like(x, alpha, beta)

# Binomial----------------------------------------------
@randomwrap
def rbinomial(n, p, size=1):
    """
    rbinomial(n,p,size=1)

    Random binomial variates.
    """

    return random.binomial(n,p,size)

def binomial_expval(n, p):
    """
    binomial_expval(n, p)

    Expected value of binomial distribution.
    """

    return p*n

def binomial_like(x, n, p):
    r"""
    binomial_like(x, n, p)

    Binomial log-likelihood.  The discrete probability distribution of the
    number of successes in a sequence of n independent yes/no experiments,
    each of which yields success with probability p.

    .. math::
        f(x \mid n, p) = \frac{n!}{x!(n-x)!} p^x (1-p)^{1-x}

    :Parameters:
      x : float
        Number of successes, > 0.
      n : int
        Number of Bernoulli trials, > x.
      p : float
        Probability of success in each trial, :math:'p \in [0,1]'.

    :Note:
     - :math:'E(X)=np'
     - :math:'Var(X)=np(1-p)'
    """
    
    return flib.binomial(x,n,p)

# Categorical----------------------------------------------
# GOF not working yet, because expval not conform to wrapper spec.
@randomwrap
def rcategorical(probs, minval=0, step=1):
    return flib.rcat(probs, minval, step)

def categorical_expval(probs, minval=0, step=1):
    """
    categorical_expval(probs, minval=0, step=1)

    Expected value of categorical distribution.
    """
    return sum([p*(minval + i*step) for i, p in enumerate(probs)])

def categorical_like( x, probs, minval=0, step=1):
    """
    Categorical log-likelihood.
    Accepts an array of probabilities associated with the histogram,
    the minimum value of the histogram (defaults to zero),
    and a step size (defaults to 1).
    """

    # Normalize, if not already
    if sum(probs) != 1.0: probs = probs/sum(probs)
    
    return flib.categorical(x, probs, minval, step)


# Cauchy----------------------------------------------
@randomwrap
def rcauchy(alpha, beta, size=1):
    """
    rcauchy(alpha, beta, size=1)

    Returns Cauchy random variates.
    """

    return alpha + beta*tan(pi*random_number(size) - pi/2.0)

def cauchy_expval(alpha, beta):
    """
    cauchy_expval(alpha, beta)

    Expected value of cauchy distribution.
    """

    return alpha

# In wikipedia, the arguments name are k, x0.
def cauchy_like(x, alpha, beta):
    r"""
    cauchy_like(x, alpha, beta)

    Cauchy log-likelihood. The Cauchy distribution is also known as the
    Lorentz or the Breit-Wigner distribution.

    .. math::
        f(x \mid \alpha, \beta) = \frac{1}{\pi \beta [1 + (\frac{x-\alpha}{\beta})^2]}

    :Parameters:
      - 'alpha' : Location parameter.
      - 'beta': Scale parameter > 0.

    :Note:
      - Mode and median are at alpha.
    """
    
    return flib.cauchy(x,alpha,beta)

# Chi square----------------------------------------------
@randomwrap
def rchi2(nu, size=1):
    """
    rchi2(nu, size=1)

    Random :math:'\chi^2' variates.
    """

    return random.chisquare(nu, size)

def chi2_expval(nu):
    """
    chi2_expval(nu)

    Expected value of Chi-squared distribution.
    """

    return nu

def chi2_like(x, nu):
    r"""
    chi2_like(x, nu)

    Chi-squared :math:'\chi^2' log-likelihood.

    .. math::
        f(x \mid \nu) = \frac{x^{(\nu-2)/2}e^{-x/2}}{2^{\nu/2}\Gamma(\nu/2)}

    :Parameters:
      x : float
        :math:'\ge 0'
      :math:'\nu' : int
        Degrees of freedom > 0

    :Note:
      - :math:'E(X)=\nu'
      - :math:'Var(X)=2\nu'

    """
    
    return flib.gamma(x, 0.5*nu, 1./2)

# Dirichlet----------------------------------------------
@randomwrap
def rdirichlet(theta, size=1):
    """
    rdirichlet(theta, size=1)

    Dirichlet random variates.
    """

    gammas = rgamma(theta,1,size)
    if size > 1 and np.size(theta) > 1:
        return (gammas.transpose()/gammas.sum(1)).transpose()
    elif np.size(theta)>1:
        return gammas/gammas.sum()
    else:
        return gammas

def dirichlet_expval(theta):
    """
    dirichlet_expval(theta)

    Expected value of Dirichlet distribution.
    """
    return theta/sum(theta)

def dirichlet_like(x, theta):
    r"""
    dirichlet_like(x, theta)

    Dirichlet log-likelihood.

    This is a multivariate continuous distribution.

    .. math::
        f(\mathbf{x}) = \frac{\Gamma(\sum_{i=1}^k \theta_i)}{\prod \Gamma(\theta_i)} \prod_{i=1}^k x_i^{\theta_i - 1}

    :Parameters:
      x : (n,k) array
        Where 'n' is the number of samples and 'k' the dimension.
        :math:'0 < x_i < 1',  :math:'\sum_{i=1}^k x_i = 1'
      theta : (n,k) or (1,k) float
        :math:'\theta > 0'
    """

    x = np.atleast_2d(x)
    
    if np.any(np.around(x.sum(1), 6)!=1):
        return -np.Inf
    return flib.dirichlet(x,np.atleast_2d(theta))

# Exponential----------------------------------------------
@randomwrap
def rexponential(beta, size=1):
    """
    rexponential(beta)

    Exponential random variates.
    """

    return random.exponential(1./beta,size)

def exponential_expval(beta):
    """
    exponential_expval(beta)

    Expected value of exponential distribution.
    """
    return beta


def exponential_like(x, beta):
    r"""
    exponential_like(x, beta)

    Exponential log-likelihood.

    The exponential distribution is a special case of the gamma distribution
    with alpha=1. It often describes the duration of an event.

    .. math::
        f(x \mid \beta) = \frac{1}{\beta}e^{-x/\beta}

    :Parameters:
      x : float
        :math:'x \ge 0'
      beta : float
        Survival parameter :math:'\beta > 0'

    :Note:
      - :math:'E(X) = \beta'
      - :math:'Var(X) = \beta^2'
    """
    
    return flib.gamma(x, 1, beta)

# Exponentiated Weibull-----------------------------------
@randomwrap
def rexponweib(alpha, k, loc=0, scale=1, size=1):
    """
    rexponweib(alpha, k, loc=0, scale=1, size=1)

    Random exponentiated Weibull variates.
    """

    q = random.uniform(size=size)
    r = flib.exponweib_ppf(q,alpha,k)
    return loc + r*scale

def exponweib_expval(alpha, k, loc, scale):
    # Not sure how we can do this, since the first moment is only
    # tractable at particular values of k
    return 'Not implemented yet.'

def exponweib_like(x, alpha, k, loc=0, scale=1):
    r"""
    exponweib_like(x,alpha,k,loc=0,scale=1)

    Exponentiated Weibull log-likelihood.

    .. math::
        f(x \mid \alpha,k,loc,scale)  & = \frac{\alpha k}{scale} (1-e^{-z^c})^{\alpha-1} e^{-z^c} z^{k-1} \\
        z & = \frac{x-loc}{scale}

    :Parameters:
      - 'x' : > 0
      - 'alpha' : Shape parameter
      - 'k' : > 0
      - 'loc' : Location parameter
      - 'scale' : Scale parameter > 0.

    """
    return flib.exponweib(x,alpha,k,loc,scale)

# Gamma----------------------------------------------
@randomwrap
def rgamma(alpha, beta, size=1):
    """
    rgamma(alpha, beta,size=1)

    Random gamma variates.
    """

    return random.gamma(shape=alpha,scale=1./beta,size=size)

def gamma_expval(alpha, beta):
    """
    gamma_expval(alpha, beta)

    Expected value of gamma distribution.
    """
    return asarray(alpha) * beta

def gamma_like(x, alpha, beta):
    r"""
    gamma_like(x, alpha, beta)

    Gamma log-likelihood.

    Represents the sum of alpha exponentially distributed random variables, each
    of which has mean beta.

    .. math::
        f(x \mid \alpha, \beta) = \frac{\beta^{\alpha}x^{\alpha-1}e^{-\beta x}}{\Gamma(\alpha)}

    :Parameters:
      x : float
        :math:'x \ge 0'
      alpha : float
        Shape parameter :math:'\alpha > 0'.
      beta : float
        Scale parameter :math:'\beta > 0'.

    """
    
    return flib.gamma(x, alpha, beta)


# GEV Generalized Extreme Value ------------------------
# Modify parameterization -> Hosking (kappa, xi, alpha)
@randomwrap
def rgev(xi, mu=0, sigma=1, size=1):
    """
    rgev(xi, mu=0, sigma=0, size=1)

    Random generalized extreme value (GEV) variates.
    """

    q = random.uniform(size=size)
    z = flib.gev_ppf(q,xi)
    return z*sigma + mu

def gev_expval(xi, mu=0, sigma=1):
    """
    gev_expval(xi, mu=0, sigma=1)

    Expected value of generalized extreme value distribution.
    """
    return mu - (sigma / xi) + (sigma / xi) * flib.gamfun(1 - xi)

def gev_like(x, xi, mu=0, sigma=1):
    r"""
    gev_like(x, xi, mu=0, sigma=1)

    Generalized Extreme Value log-likelihood

    .. math::
        pdf(x \mid \xi,\mu,\sigma) = \frac{1}{\sigma}(1 + \xi \left[\frac{x-\mu}{\sigma}\right])^{-1/\xi-1}\exp{-(1+\xi \left[\frac{x-\mu}{\sigma}\right])^{-1/\xi}}

    .. math::
        \sigma & > 0,\\
        x & > \mu-\sigma/\xi \text{ if } \xi > 0,\\
        x & < \mu-\sigma/\xi \text{ if } \xi < 0\\
        x & \in [-\infty,\infty] \text{ if } \xi = 0

    """

    return flib.gev(x, xi, mu, sigma)

# Geometric----------------------------------------------
# Changed the return value
@randomwrap
def rgeometric(p, size=1):
    """
    rgeometric(p, size=1)

    Random geometric variates.
    """

    return random.geometric(p, size)

def geometric_expval(p):
    """
    geometric_expval(p)

    Expected value of geometric distribution.
    """
    return 1. / p

def geometric_like(x, p):
    r"""
    geometric_like(x, p)

    Geometric log-likelihood. The probability that the first success in a
    sequence of Bernoulli trials occurs after x trials.

    .. math::
        f(x \mid p) = p(1-p)^{x-1}

    :Parameters:
      x : int
        Number of trials before first success, > 0.
      p : float
        Probability of success on an individual trial, :math:'p \in [0,1]'

    :Note:
      - :math:'E(X)=1/p'
      - :math:'Var(X)=\frac{1-p}{p^2}'

    """
    
    return flib.geometric(x, p)

# Half-normal----------------------------------------------
@randomwrap
def rhalf_normal(tau, size=1):
    """
    rhalf_normal(tau, size=1)

    Random half-normal variates.
    """

    return abs(random.normal(0, sqrt(1/tau), size))

def half_normal_expval(tau):
    """
    half_normal_expval(tau)

    Expected value of half normal distribution.
    """

    return sqrt(2. * pi / asarray(tau))

def half_normal_like(x, tau):
    r"""
    half_normal_like(x, tau)

    Half-normal log-likelihood, a normal distribution with mean 0 and limited
    to the domain :math:'x \in \[0, \infty\)'.

    .. math::
        f(x \mid \tau) = \sqrt{\frac{2\tau}{\pi}}\exp\left\{ {\frac{-x^2 \tau}{2}}\right\}

    :Parameters:
      x : float
        :math:'x \ge 0'
      tau : float
        :math:'\tau > 0'

    """
    
    return flib.hnormal(x, tau)

# Hypergeometric----------------------------------------------
def rhypergeometric(n, m, N, size=1):
    """
    rhypergeometric(n, m, N, size=1)

    Returns hypergeometric random variates.
    """
    return random.hypergeometric(m, N-m, n, size)

def hypergeometric_expval(n, m, N):
    """
    hypergeometric_expval(n, m, N)

    Expected value of hypergeometric distribution.
    """
    return 1. * n * m / N

def hypergeometric_like(x, n, m, N):
    r"""
    hypergeometric_like(x, n, m, N)

    Hypergeometric log-likelihood. Discrete probability distribution that
    describes the number of successes in a sequence of draws from a finite
    population without replacement.

    .. math::
        f(x \mid n, m, N) = &=& \frac{\left(\begin{array}{c}m \\x\end{array}\right) \left(\begin{array}{c}N-m \\n-x\end{array}\right)}{\left(\begin{array}{c}N \\n\end{array}\right)}

    :Parameters:
      x : int
        Number of successes in a sample drawn from a population.
        :math:'\max(0, draws-failures) \leq x \leq \min(draws, success)'
      n : int
        Size of sample drawn from the population.
      m : int
        Number of successes in the population.
      N : int
        Total number of units in the population.

    :Note:
      :math:'E(X) = \frac{n n}{N}'
    """
    
    return flib.hyperg(x, n, m, N)

# Inverse gamma----------------------------------------------
@randomwrap
def rinverse_gamma(alpha, beta,size=1):
    """
    rinverse_gamma(alpha, beta,size=1)

    Random inverse gamma variates.
    """

    return 1. / random.gamma(shape=alpha, scale=1./beta, size=size)

def inverse_gamma_expval(alpha, beta):
    """
    inverse_gamma_expval(alpha, beta)

    Expected value of inverse gamma distribution.
    """
    return 1. / (asarray(beta) * (alpha-1.))

def inverse_gamma_like(x, alpha, beta):
    r"""
    inverse_gamma_like(x, alpha, beta)

    Inverse gamma log-likelihood, the reciprocal of the gamma distribution.

    .. math::
        f(x \mid \alpha, \beta) = \frac{\beta^{\alpha}}{\Gamma(\alpha)} x^{-\alpha - 1} \exp\left(\frac{-\beta}{x}}\right)

    :Parameters:
      x : float
        x > 0
      alpha : float
        Shape parameter, :math:'\alpha > 0'.
      beta : float
        Scale parameter, :math:'\beta > 0'.

    :Note:
      :math:'E(X)=\frac{1}{\beta(\alpha-1)}' for :math:'\alpha > 1'.
    """
    
    return flib.igamma(x, alpha, beta)

# Lognormal----------------------------------------------
@randomwrap
def rlognormal(mu, tau,size=1):
    """
    rlognormal(mu, tau,size=1)

    Return random lognormal variates.
    """

    return random.lognormal(mu, sqrt(1./tau),size)

def lognormal_expval(mu, tau):
    """
    lognormal_expval(mu, tau)

    Expected value of log-normal distribution.
    """
    return np.exp(mu + 1./2/tau)

def lognormal_like(x, mu, tau):
    r"""
    lognormal_like(x, mu, tau)

    Log-normal log-likelihood. Distribution of any random variable whose
    logarithm is normally distributed. A variable might be modeled as
    log-normal if it can be thought of as the multiplicative product of many
    small independent factors.

    .. math::
        f(x \mid \mu, \tau) = \sqrt{\frac{\tau}{2\pi}}\frac{
        \exp\left\{ -\frac{\tau}{2} (\ln(x)-\mu)^2 \right\}}{x}

    :Parameters:
      x : float
        x > 0
      mu : float
        Location parameter.
      tau : float
        Scale parameter, > 0.

    :Note:
      :math:'E(X)=e^{\mu+\frac{1}{2\tau}}'
    """
    
    
    return flib.lognormal(x,mu,tau)

# Multinomial----------------------------------------------
#@randomwrap
def rmultinomial(n,p,size=None): # Leaving size=None as the default means return value is 1d array if not specified-- nicer.
    """
    rmultinomial(n,p,size=1)

    Random multinomial variates.
    """
    
    # Single value for p:
    if len(np.shape(p))==1:
        return random.multinomial(n,p,size)

    # Multiple values for p:
    if np.isscalar(n):
        n = n * np.ones(p.shape[0],dtype=np.int)
    out = np.empty(p.shape)
    for i in xrange(p.shape[0]):
        out[i,:] = random.multinomial(n[i],p[i,:],size)
    return out

def multinomial_expval(n,p):
    """
    multinomial_expval(n,p)

    Expected value of multinomial distribution.
    """
    return asarray([pr * n for pr in p])

def multinomial_like(x, n, p):
    r"""
    multinomial_like(x, n, p)

    Multinomial log-likelihood with k-1 bins. Generalization of the binomial
    distribution, but instead of each trial resulting in "success" or
    "failure", each one results in exactly one of some fixed finite number k
    of possible outcomes over n independent trials. 'x[i]' indicates the number 
    of times outcome number i was observed over the n trials.

    .. math::
        f(x \mid n, p) = \frac{n!}{\prod_{i=1}^k x_i!} \prod_{i=1}^k p_i^{x_i}

    :Parameters:
      x : (ns, k) int
        Random variable indicating the number of time outcome i is observed,
        :math:'\sum_{i=1}^k x_i=n', :math:'x_i \ge 0'.
      n : int
        Number of trials.
      p : (k,) float
        Probability of each one of the different outcomes,
        :math:'\sum_{i=1}^k p_i = 1)', :math:'p_i \ge 0'.

    :Note:
      - :math:'E(X_i)=n p_i'
      - :math:'var(X_i)=n p_i(1-p_i)'
      - :math:'cov(X_i,X_j) = -n p_i p_j'

    """

    x = np.atleast_2d(x) #flib expects 2d arguments. Do we still want to support multiple p values along realizations ?
    p = np.atleast_2d(p)
    return flib.multinomial(x, n, p)

# Multivariate hypergeometric------------------------------
def rmultivariate_hypergeometric(n, m, size=None):
    """
    Random multivariate hypergeometric variates.
    
    n : Number of draws.
    m : Number of items in each category.
    """

    N = len(m)
    urn = np.repeat(np.arange(N), m)
    
    if size:
        draw = np.array([[urn[i] for i in np.random.permutation(len(urn))[:n]] for j in range(size)])

        r = [[sum(draw[j]==i) for i in range(len(m))] for j in range(size)]
    else:
        draw = np.array([urn[i] for i in np.random.permutation(len(urn))[:n]])

        r = [sum(draw==i) for i in range(len(m))]
    return np.asarray(r)

def multivariate_hypergeometric_expval(n, m):
    """
    multivariate_hypergeometric_expval(n, m)

    Expected value of multivariate hypergeometric distribution.
    
    n : number of items drawn.
    m : number of items in each category.
    """
    m= np.asarray(m, float)
    return n * (m / m.sum())


def multivariate_hypergeometric_like(x, m):
    r"""
    multivariate_hypergeometric_like(x, m)

    The multivariate hypergeometric describes the probability of drawing x[i] 
    elements of the ith category, when the number of items in each category is
    given by m. 
    

    .. math::
        \frac{\prod_i \binom{m_i}{c_i}}{\binom{N}{n]}

    where :math:'N = \sum_i m_i' and :math:'n = \sum_i x_i'.
    
    :Parameters:
        x : int sequence
            Number of draws from each category, :math:'< m'
        m : int sequence
            Number of items in each categoy. 
    """
    
    
    return flib.mvhyperg(x, m)


if flib_blas_OK:
    # Multivariate normal--------------------------------------
    def rmv_normal(mu, tau, size=1):
        """
        rmv_normal(mu, tau, size=1)

        Random multivariate normal variates.
        """
    
        sig = np.linalg.cholesky(tau)
        mu_size = mu.shape

        if size==1:
            out = random.normal(size=mu_size)
            try:
                flib.dtrsm_wrap(sig , out, 'L', 'T', 'L')
            except:
                out = np.linalg.solve(sig, out)
            out+=mu    
            return out
        else:
            if not hasattr(size,'__iter__'):
                size = (size,)
            tot_size = prod(size)
            out = random.normal(size = (tot_size,) + mu_size)
            for i in xrange(tot_size):
                try:
                    flib.dtrsm_wrap(sig , out[i,:], 'L', 'T', 'L')
                except:
                    out[i,:] = np.linalg.solve(sig, out[i,:])
                out[i,:] += mu
            return out.reshape(size+mu_size)

    def mv_normal_expval(mu, tau):
        """
        mv_normal_expval(mu, tau)

        Expected value of multivariate normal distribution.
        """
        return mu

    def mv_normal_like(x, mu, tau):
        r"""
        mv_normal_like(x, mu, tau)

        Multivariate normal log-likelihood

        .. math::
            f(x \mid \pi, T) = \frac{T^{n/2}}{(2\pi)^{1/2}} \exp\left\{ -\frac{1}{2} (x-\mu)^{\prime}T(x-\mu) \right\}

        x: (n,k)
        mu: (k)
        tau: (k,k)
        tau positive definite
        """
        # TODO: Vectorize in Fortran
        if len(x.shape)>1:
            return sum([flib.prec_mvnorm(r,mu,tau) for r in x])
        else:
            return flib.prec_mvnorm(x,mu,tau)
        
    # Multivariate normal, parametrized with covariance---------------------------
    def rmv_normal_cov(mu, C, size=1):
        """
        rmv_normal_cov(mu, C)

        Random multivariate normal variates.
        """
        mu_size = mu.shape
        if size==1:
            return random.multivariate_normal(mu, C, size).reshape(mu_size)
        else:
            return random.multivariate_normal(mu, C, size).reshape((size,)+mu_size)

    def mv_normal_cov_expval(mu, C):
        """
        mv_normal_cov_expval(mu, C)

        Expected value of multivariate normal distribution.
        """
        return mu

    def mv_normal_cov_like(x, mu, C):
        r"""
        mv_normal_cov_like(x, mu, C)

        Multivariate normal log-likelihood

        .. math::
            f(x \mid \pi, C) = \frac{T^{n/2}}{(2\pi)^{1/2}} \exp\left\{ -\frac{1}{2} (x-\mu)^{\prime}C^{-1}(x-\mu) \right\}

        x: (n,k)
        mu: (k)
        C: (k,k)
        C positive definite
        """
        # TODO: Vectorize in Fortran
        if len(x.shape)>1:
            return sum([flib.cov_mvnorm(r,mu,C) for r in x])
        else:
            return flib.cov_mvnorm(x,mu,C)
     

    # Multivariate normal, parametrized with Cholesky factorization.----------
    def rmv_normal_chol(mu, sig, size=1):
        """
        rmv_normal(mu, sig)

        Random multivariate normal variates.
        """
        mu_size = mu.shape

        if size==1:
            out = random.normal(size=mu_size)
            try:
                flib.dtrmm_wrap(sig , out, 'L', 'N', 'L')
            except:
                out = np.dot(sig, out)
            out+=mu    
            return out
        else:
            if not hasattr(size,'__iter__'):
                size = (size,)
            tot_size = prod(size)
            out = random.normal(size = (tot_size,) + mu_size)
            for i in xrange(tot_size):
                try:
                    flib.dtrmm_wrap(sig , out[i,:], 'L', 'N', 'L')
                except:
                    out[i,:] = np.dot(sig, out[i,:])
                out[i,:] += mu
            return out.reshape(size+mu_size)


    def mv_normal_chol_expval(mu, sig):
        """
        mv_normal_expval(mu, sig)

        Expected value of multivariate normal distribution.
        """
        return mu

    def mv_normal_chol_like(x, mu, sig):
        r"""
        mv_normal_like(x, mu, tau)

        Multivariate normal log-likelihood

        .. math::
            f(x \mid \pi, \sigma) = \frac{T^{n/2}}{(2\pi)^{1/2}} \exp\left\{ -\frac{1}{2} (x-\mu)^{\prime}\sigma \sigma^{\prime}(x-\mu) \right\}

        x: (n,k)
        mu: (k)
        sigma: (k,k)
        sigma lower triangular
        """
        # TODO: Vectorize in Fortran
        if len(x.shape)>1:
            return sum([flib.chol_mvnorm(r,mu,sig) for r in x])
        else:
            return flib.chol_mvnorm(x,mu,sig)



# Negative binomial----------------------------------------
@randomwrap
def rnegative_binomial(mu, alpha, size=1):
    """
    rnegative_binomial(mu, alpha, size=1)

    Random negative binomial variates.
    """

    return random.negative_binomial(alpha, alpha / (mu + alpha), size)

def negative_binomial_expval(mu, alpha):
    """
    negative_binomial_expval(mu, alpha)

    Expected value of negative binomial distribution.
    """
    return mu


def negative_binomial_like(x, mu, alpha):
    r"""
    negative_binomial_like(x, mu, alpha)

    Negative binomial log-likelihood

    .. math::
        f(x \mid r, p) = \frac{(x+r-1)!}{x! (r-1)!} p^r (1-p)^x

    x > 0, mu > 0, alpha > 0
    """
    
    
    return flib.negbin2(x, mu, alpha)

# Normal---------------------------------------------------
@randomwrap
def rnormal(mu, tau,size=1):
    """
    rnormal(mu, tau, size=1)

    Random normal variates.
    """
    return random.normal(mu, 1./sqrt(tau), size)

def normal_expval(mu, tau):
    """
    normal_expval(mu, tau)

    Expected value of normal distribution.
    """
    return mu

def normal_like(x, mu, tau):
    r"""
    normal_like(x, mu, tau)

    Normal log-likelihood.

    .. math::
        f(x \mid \mu, \tau) = \sqrt{\frac{\tau}{2\pi}} \exp\left\{ -\frac{\tau}{2} (x-\mu)^2 \right\}


    :Parameters:
      x : float
        Input data.
      mu : float
        Mean of the distribution.
      tau : float
        Precision of the distribution, > 0.

    :Note:
      - :math:'E(X) = \mu'
      - :math:'Var(X) = 1/\tau'

    """
    # try:
    #     constrain(tau, lower=0)
    # except ZeroProbability:
    #     return -Inf
    return flib.normal(x, mu, tau)


# Poisson--------------------------------------------------
@randomwrap
def rpoisson(mu, size=1):
    """
    rpoisson(mu, size=1)

    Random poisson variates.
    """

    return random.poisson(mu,size)


def poisson_expval(mu):
    """
    poisson_expval(mu)

    Expected value of Poisson distribution.
    """
    return mu


def poisson_like(x,mu):
    r"""
    poisson_like(x,mu)

    Poisson log-likelihood. The Poisson is a discrete probability distribution.
    It expresses the probability of a number of events occurring in a fixed
    period of time if these events occur with a known average rate, and are
    independent of the time since the last event. The Poisson distribution can
    be derived as a limiting case of the binomial distribution.

    .. math::
        f(x \mid \mu) = \frac{e^{-\mu}\mu^x}{x!}

    :Parameters:
      x : int
        :math:'x \in {0,1,2,...}'
      mu : float
        Expected number of occurrences that occur during the given interval,
        :math:'\mu \geq 0'.

    :Note:
      - :math:'E(x)=\mu'
      - :math:'Var(x)=\mu'
    """
    # try:
    #     constrain(x, lower=0,allow_equal=True)
    #     constrain(mu, lower=0,allow_equal=True)
    # except ZeroProbability:
    #     return -Inf
    return flib.poisson(x,mu)

# Truncated normal distribution--------------------------
@randomwrap
def rtruncnorm(mu, sigma, a, b, size=1):
    """rtruncnorm(mu, sigma, a, b, size=1)
    
    Random truncated normal variates.
    """
    
    na = pymc.utils.normcdf((a-mu)/sigma)
    nb = pymc.utils.normcdf((b-mu)/sigma)
    
    # Use the inverse CDF generation method.
    U = np.random.mtrand.uniform(size=size)
    q = U * nb + (1-U)*na
    R = pymc.utils.invcdf(q)
    
    # Unnormalize
    return R*sigma + mu

# TODO: code this.
def truncnorm_expval(mu, sigma, a, b):
    """E(X)=\mu + \frac{\sigma(\varphi_1-\varphi_2)}{T}, where T=\Phi\left(\frac{B-\mu}{\sigma}\right)-\Phi\left(\frac{A-\mu}{\sigma}\right) and \varphi_1 = \varphi\left(\frac{A-\mu}{\sigma}\right) and \varphi_2 = \varphi\left(\frac{B-\mu}{\sigma}\right), where \varphi is the probability density function of a standard normal random variable."""
    pass

def truncnorm_like(x, mu, sigma, a, b):
    r"""truncnorm_like(x, mu, sigma, a, b)
    
    Truncated normal log-likelihood.
    
    .. math::
        f(x \mid \mu, \sigma, a, b) = \frac{\phi(\frac{x-\mu}{\sigma})} {\Phi(\frac{b-\mu}{\sigma}) - \Phi(\frac{a-\mu}{\sigma})}, 
    
    """
    x = np.atleast_1d(x)
    a = np.atleast_1d(a)
    b = np.atleast_1d(b)
    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)
    if (x < a).any() or (x>b).any():
        return -np.inf
    else:
        n = len(x)
        phi = normal_like(x, mu, 1./sigma**2)
        Phia = pymc.utils.normcdf((a-mu)/sigma)
        if b == np.inf:
            Phib = 1.0
        else:
            Phib = pymc.utils.normcdf((b-mu)/sigma)
        d = log(Phib-Phia)
        if len(d) == n:
            Phi = d.sum()
        else:
            Phi = n*d
        return phi - Phi

# Uniform--------------------------------------------------
@randomwrap
def runiform(lower, upper, size=1):
    """
    runiform(lower, upper, size=1)

    Random uniform variates.
    """
    return random.uniform(lower, upper, size)

def uniform_expval(lower, upper):
    """
    uniform_expval(lower, upper)

    Expected value of uniform distribution.
    """
    return (upper - lower) / 2.

def uniform_like(x,lower, upper):
    r"""
    uniform_like(x, lower, upper)

    Uniform log-likelihood.

    .. math::
        f(x \mid lower, upper) = \frac{1}{upper-lower}

    :Parameters:
      x : float
       :math:'lower \geq x \geq upper'
      lower : float
        Lower limit.
      upper : float
        Upper limit.
    """

    return flib.uniform_like(x, lower, upper)

# Weibull--------------------------------------------------
@randomwrap
def rweibull(alpha, beta,size=1):
    tmp = -log(runiform(0, 1, size))
    return beta * (tmp ** (1. / alpha))

def weibull_expval(alpha,beta):
    """
    weibull_expval(alpha,beta)

    Expected value of weibull distribution.
    """
    return beta * gammaln((alpha + 1.) / alpha)

def weibull_like(x, alpha, beta):
    r"""
    weibull_like(x, alpha, beta)

    Weibull log-likelihood

    .. math::
        f(x \mid \alpha, \beta) = \frac{\alpha x^{\alpha - 1}
        \exp(-(\frac{x}{\beta})^{\alpha})}{\beta^\alpha}

    :Parameters:
      x : float
        :math:'x \ge 0'
      alpha : float
        > 0
      beta : float
        > 0

    :Note:
      - :math:'E(x)=\beta \Gamma(1+\frac{1}{\alpha})'
      - :math:'Var(x)=\beta^2 \Gamma(1+\frac{2}{\alpha} - \mu^2)'
    """
    # try:
    #     constrain(alpha, lower=0)
    #     constrain(beta, lower=0)
    #     constrain(x, lower=0)
    # except ZeroProbability:
    #     return -Inf
    return flib.weibull(x, alpha, beta)

# Wishart---------------------------------------------------
def rwishart(n, Tau):
    """
    rwishart(n, Tau)

    Return a Wishart random matrix.
    
    Tau is the inverse of the covariance matrix :math:'C'.
    """

    p = Tau.shape[0]    
    # sig = np.linalg.cholesky(np.linalg.inv(Tau))
    sig = np.linalg.cholesky(Tau)
    if n<p:
        raise ValueError, 'Wishart parameter n must be greater than size of matrix.'
    norms = np.random.normal(size=p*(p-1)/2)
    chi_sqs = sqrt(np.random.chisquare(df=np.arange(n,n-p,-1)))
    A= flib.expand_triangular(chi_sqs, norms)
    flib.dtrsm_wrap(sig,A,side='L',uplo='L',transa='T')
    # flib.dtrmm_wrap(sig,A,side='L',uplo='L',transa='N')
    return np.asmatrix(np.dot(A,A.T))
        


def wishart_expval(n, Tau):
    """
    wishart_expval(n, Tau)

    Expected value of wishart distribution.
    """
    return n * np.asarray(Tau.I)

def wishart_like(X, n, Tau):
    r"""
    wishart_like(X, n, Tau)

    Wishart log-likelihood. The Wishart distribution is the probability
    distribution of the maximum-likelihood estimator (MLE) of the covariance
    matrix of a multivariate normal distribution. If Tau=1, the distribution
    is identical to the chi-square distribution with n degrees of freedom.

    .. math::
        f(X \mid n, T) = {\mid T \mid}^{n/2}{\mid X \mid}^{(n-k-1)/2} \exp\left\{ -\frac{1}{2} Tr(TX) \right\}
    
    where :math:'k' is the rank of X.
    
    :Parameters:
      X : matrix
        Symmetric, positive definite.
      n : int
        Degrees of freedom, > 0.
      Tau : matrix
        Symmetric and positive definite

    """
    return flib.blas_wishart(X,n,Tau)

# Wishart, parametrized by covariance ------------------------------------
def rwishart_cov(n, C):
    """
    rwishart(n, C)

    Return a Wishart random matrix.
    """
    p = C.shape[0]    
    sig = np.linalg.cholesky(C)
    if n<p:
        raise ValueError, 'Wishart parameter n must be greater than size of matrix.'
    norms = np.random.normal(size=p*(p-1)/2)
    chi_sqs = sqrt(np.random.chisquare(df=np.arange(n,n-p,-1)))
    A= flib.expand_triangular(chi_sqs, norms)
    flib.dtrmm_wrap(sig,A,side='L',uplo='L',transa='N')
    return np.asmatrix(np.dot(A,A.T))


def wishart_cov_expval(n, C):
    """
    wishart_expval(n, C)

    Expected value of wishart distribution.
    """
    return n * np.asarray(C)

def wishart_cov_like(X, n, C):
    r"""
    wishart_like(X, n, C)

    Wishart log-likelihood. The Wishart distribution is the probability
    distribution of the maximum-likelihood estimator (MLE) of the covariance
    matrix of a multivariate normal distribution. If Tau=1, the distribution
    is identical to the chi-square distribution with n degrees of freedom.

    .. math::
        f(X \mid n, T) = {\mid T \mid}^{n/2}{\mid X \mid}^{(n-k-1)/2} \exp\left\{ -\frac{1}{2} Tr(TX) \right\}
    
    where :math:'k' is the rank of X.
    
    :Parameters:
      X : matrix
        Symmetric, positive definite.
      n : int
        Degrees of freedom, > 0.
      Tau : matrix
        Symmetric and positive definite

    """
    return flib.blas_wishart_cov(X,n,C)


# -----------------------------------------------------------
# DECORATORS
# -----------------------------------------------------------

def name_to_funcs(name, module):
    if type(module) is types.ModuleType:
        module = copy(module.__dict__)
    elif type(module) is dict:
        module = copy(module)
    else:
        raise AttributeError

    try:
       logp = module[name+"_like"]
    except:
        raise "No likelihood found with this name ", name+"_like"

    try:
        random = module['r'+name]
    except:
        random = None
            
    return logp, random
    

def valuewrapper(f):
    """Return a likelihood accepting value instead of x as a keyword argument.
    This is specifically intended for the instantiator above. 
    """
    def wrapper(**kwds):
        value = kwds.pop('value')
        return f(value, **kwds)
    wrapper.__dict__.update(f.__dict__)
    return wrapper
    
def random_method_wrapper(f, size, shape):
    """
    Wraps functions to return values of appropriate shape.
    """
    if f is None:
        return f
    def wrapper(**kwds):
        value = f(size=size, **kwds)
        if shape is not None:
            value= np.reshape(value, shape)
        return value
    return wrapper
        
def fortranlike(f, snapshot, mv=False):
    """
    Decorator function for fortran likelihoods
    ==========================================

    Wrap function f(*args, **kwds) where f is a likelihood defined in flib.

    Assume args = (x, parameter1, parameter2, ...)
    Before passing the arguments to the function, the wrapper makes sure that
    the parameters have the same shape as x.

    mv: multivariate (True/False)

    Add compatibility with GoF (Goodness of Fit) tests
    --------------------------------------------------
    * Add a 'prior' keyword (True/False)
    * If the keyword gof is given and is True, return the GoF (Goodness of Fit)
    points instead of the likelihood.
    * A 'loss' keyword can be given, to specify the loss function used in the
    computation of the GoF points.
    * If the keyword random is given and True, return a random variate instead
    of the likelihood.
    """

    name = f.__name__[:-5]
    # Take a snapshot of the main namespace.


    # Find the functions needed to compute the gof points.
    expval_func = snapshot[name+'_expval']
    random_func = snapshot['r'+name]

    def wrapper(*args, **kwds):
        """
        This wraps a likelihood.
        """

        # Do we really need the shape manipulations, or does f2py take care of
        # all of this on its own?
        # Shape manipulations
        # if not mv:
        #     xshape = np.shape(args[0])
        #     newargs = [np.asarray(args[0])]
        #     for arg in args[1:]:
        #         newargs.append(np.resize(arg, xshape))
        #     for key in kwds.iterkeys():
        #         kwds[key] = kwds[key]
        # else:
        #     """
# x, mu, Tau
            # x: (kxN)
            # mu: (kxN) or (kx1)
            # Tau: (k,k)
            # """
            # 
            # xshape=np.shape(args[0])
            # newargs = [np.asarray(args[0])]
            # newargs.append(np.resize(args[1], xshape))
            # newargs.append(np.asarray(args[2]))

        if kwds.pop('gof', False) and not kwds.pop('prior', False):
            """
Return gof points."""

            loss = kwds.pop('gof', squared_loss)
            #name = kwds.pop('name', name)
            expval = expval_func(*newargs[1:], **kwds)
            y = random_func(*newargs[1:], **kwds)
            gof_points = GOFpoints(newargs[0],y,expval,loss)
            return gof_points
        elif kwds.pop('random', False):
            return random_func(*newargs[1:], **kwds)
        else:
            """
Return likelihood."""

            # try:
            return f(*newargs, **kwds)
            # except ZeroProbability:
            # return -np.Inf


    # Assign function attributes to wrapper.
    wrapper.__doc__ = f.__doc__
    wrapper._pymc = True
    wrapper.__name__ = f.__name__
    wrapper.name = name
    return wrapper


"""
Decorate the likelihoods
"""

snapshot = locals().copy()
likelihoods = {}
for name, obj in snapshot.iteritems():
    if name[-5:] == '_like' and name[:-5] in availabledistributions:
        likelihoods[name[:-5]] = snapshot[name]

def local_decorated_likelihoods(obj):
    """
    New interface likelihoods
    """

    for name, like in likelihoods.iteritems():
        obj[name+'_like'] = fortranlike(like, snapshot)


#local_decorated_likelihoods(locals())
# Decorating the likelihoods breaks the creation of distribution instantiators -DH. 



# Create Stochastic instantiators

for dist in continuous_distributions:
    dist_logp, dist_random = name_to_funcs(dist, locals())
    locals()[capitalize(dist)]= stochastic_from_dist(dist, dist_logp, dist_random)
    
for dist in discrete_distributions:
    dist_logp, dist_random = name_to_funcs(dist, locals())
    locals()[capitalize(dist)]= stochastic_from_dist(dist, dist_logp, dist_random, dtype=np.int)

dist_logp, dist_random = name_to_funcs('bernoulli', locals())
Bernoulli = stochastic_from_dist('bernoulli', dist_logp, dist_random, dtype=np.bool)


def uninformative_like(x):
    """
    uninformative_like(x)
    
    Uninformative log-likelihood. Returns 0 regardless of the value of x.
    """
    return 0.


def one_over_x_like(x):
    """
    one_over_x_like(x)

    returns -Inf if x<0, -log(x) otherwise.
    """
    if any(x<0):
        return -Inf
    else:
        return -sum(log(x))


Uninformative = stochastic_from_dist('uninformative', logp = uninformative_like)
DiscreteUninformative = stochastic_from_dist('uninformative', logp = uninformative_like, dtype=np.int)
OneOverX = stochastic_from_dist('one_over_x_like', logp = one_over_x_like)

if __name__ == "__main__":
    import doctest
    doctest.testmod()



