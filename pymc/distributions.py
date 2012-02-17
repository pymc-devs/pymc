"""
pymc.distributions
 
A collection of common probability distributions. The objects associated
with a distribution called 'dist' are:

  dist_like : function
    The log-likelihood function corresponding to dist. PyMC's convention
    is to sum the log-likelihoods of multiple input values, so all
    log-likelihood functions return a single float.
  rdist : function
    The random variate generator corresponding to dist. These take a
    'size' argument indicating how many variates should be generated.
  dist_expval : function
    Computes the expected value of a dist-distributed variable.
  Dist : Stochastic subclass
    Instances have dist_like as their log-probability function
    and rdist as their random function.
"""

#-------------------------------------------------------------------
# Decorate fortran functions from pymc.flib to ease argument passing
#-------------------------------------------------------------------
# TODO: Add exponweib_expval
# TODO: categorical, mvhypergeometric
# TODO: __all__

__docformat__='reStructuredText'

from . import flib
import pymc
import numpy as np
from .Node import ZeroProbability
from .PyMCObjects import Stochastic, Deterministic
from .CommonDeterministics import Lambda
from numpy import pi, inf
import itertools
import pdb
from . import utils
import warnings

from pymc import six
from pymc.six import print_
xrange = six.moves.xrange

def poiscdf(a, x):
    x = np.atleast_1d(x)
    a = np.resize(a, x.shape)
    values = np.array([flib.gammq(b,y) for b, y in zip(a.ravel(), x.ravel())])
    return values.reshape(x.shape)

# Import utility functions
import inspect, types
from copy import copy
random_number = np.random.random
inverse = np.linalg.pinv

class ArgumentError(AttributeError):
    """Incorrect class argument"""
    pass

sc_continuous_distributions = ['beta', 'cauchy', 'chi2',
                               'degenerate', 'exponential', 'exponweib',
                               'gamma', 'half_normal', 
                               'inverse_gamma', 'laplace', 'logistic',
                               'lognormal', 'noncentral_t', 'normal', 
                               'pareto', 't', 'truncated_pareto', 'uniform',
                               'weibull', 'skew_normal', 'truncated_normal',
                               'von_mises']
sc_bool_distributions = ['bernoulli']
sc_discrete_distributions = ['binomial', 'betabin', 'geometric', 'poisson',
                             'negative_binomial', 'categorical', 'hypergeometric',
                             'discrete_uniform', 'truncated_poisson']

sc_nonnegative_distributions = ['bernoulli', 'beta', 'betabin', 'binomial', 'chi2', 'exponential',
                                'exponweib', 'gamma', 'half_normal',
                                'hypergeometric', 'inverse_gamma', 'lognormal',
                                'weibull']

mv_continuous_distributions = ['dirichlet', 'inverse_wishart', 'mv_normal',
                               'mv_normal_cov', 'mv_normal_chol', 'wishart',
                               'wishart_cov', 'inverse_wishart_prec']

mv_discrete_distributions = ['multivariate_hypergeometric', 'multinomial']

mv_nonnegative_distributions = ['dirichlet', 'inverse_wishart', 'wishart',
                                'wishart_cov', 'multivariate_hypergeometric',
                                'multinomial']

availabledistributions = (sc_continuous_distributions +
                          sc_bool_distributions +
                          sc_discrete_distributions +
                          mv_continuous_distributions +
                          mv_discrete_distributions)

# Changes lower case, underscore-separated names into "Class style"
# capitalized names For example, 'negative_binomial' becomes
# 'NegativeBinomial'
capitalize = lambda name: ''.join([s.capitalize() for s in name.split('_')])


# ==============================================================================
# User-accessible function to convert a logp and random function to a
# Stochastic subclass.
# ==============================================================================

# TODO Document this function
def bind_size(randfun, shape):
    def newfun(*args, **kwargs):
        try:
            return np.reshape(randfun(size=shape, *args, **kwargs),shape)
        except ValueError:
            # Account for non-array return values
            return randfun(size=shape, *args, **kwargs)
    newfun.scalar_version = randfun
    return newfun

def new_dist_class(*new_class_args):
    """
    Returns a new class from a distribution.

    :Parameters:
      dtype : numpy dtype
        The dtype values of instances of this class.
      name : string
        Name of the new class.
      parent_names : list of strings
        The labels of the parents of this class.
      parents_default : list
        The default values of parents.
      docstr : string
        The docstring of this class.
      logp : function
        The log-probability function for this class.
      random : function
        The random function for this class.
      mv : boolean
        A flag indicating whether this class represents array-valued
        variables.

      .. note::
        stochastic_from_dist provides a higher-level version.

      :SeeAlso:
        stochastic_from_dist
    """

    (dtype, name, parent_names, parents_default, docstr, logp, random, mv, logp_partial_gradients) = new_class_args

    class new_class(Stochastic):
        __doc__ = docstr

        def __init__(self, *args, **kwds):
            (dtype, name, parent_names, parents_default, docstr, logp, random, mv, logp_partial_gradients) = new_class_args
            parents=parents_default
            
            # Figure out what argument names are needed.
            arg_keys = ['name', 'parents', 'value', 'observed', 'size', 'trace', 'rseed', 'doc', 'debug', 'plot', 'verbose']
            arg_vals = [None, parents, None, False, None, True, True, None, False, None, -1]
            if 'isdata' in kwds:
                warnings.warn('"isdata" is deprecated, please use "observed" instead.')
                kwds['observed'] = kwds['isdata']
                pass


            # No size argument allowed for multivariate distributions.
            if mv:
                arg_keys.pop(4)
                arg_vals.pop(4)

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
                    raise ValueError('Too many positional arguments provided. Arguments for class ' + self.__class__.__name__ + ' are: ' + str(all_args_needed))


            # Sort keyword arguments
            for k in args_needed:
                if k in parent_names:
                    try:
                        parents[k] = kwds.pop(k)
                    except:
                        if k in parents_default:
                            parents[k] = parents_default[k]
                        else:
                            raise ValueError('No value given for parent ' + k)
                elif k in arg_dict_out.keys():
                    try:
                        arg_dict_out[k] = kwds.pop(k)
                    except:
                        pass

            # Remaining unrecognized arguments raise an error.
            if len(kwds) > 0:
                raise TypeError('Keywords '+ str(kwds.keys()) + ' not recognized. Arguments recognized are ' + str(args_needed))

        # Determine size desired for scalar variables.
        # Notes
        # -----
        # Case | init_val     | parents       | size | value.shape | bind size
        # ------------------------------------------------------------------
        # 1.1  | None         | scalars       | None | 1           | 1
        # 1.2  | None         | scalars       | n    | n           | n
        # 1.3  | None         | n             | None | n           | 1
        # 1.4  | None         | n             | n(m) | n (Error)   | 1 (-)
        # 2.1  | scalar       | scalars       | None | 1           | 1
        # 2.2  | scalar       | scalars       | n    | n           | n
        # 2.3  | scalar       | n             | None | n           | 1
        # 2.4  | scalar       | n             | n(m) | n (Error)   | 1 (-)
        # 3.1  | n            | scalars       | None | n           | n
        # 3.2  | n            | scalars       | n(m) | n (Error)   | n (-)
        # 3.3  | n            | n             | None | n           | 1
        # 3.4  | n            | n             | n(m) | n (Error)   | 1 (-)

            if not mv:

                shape = arg_dict_out.pop('size')
                shape = None if shape is None else tuple(np.atleast_1d(shape))

                init_val = arg_dict_out['value']
                init_val_shape = None if init_val is None else np.shape(init_val)

                if len(parents) > 0:
                    pv = [np.shape(utils.value(v)) for v in parents.values()]
                    biggest_parent = np.argmax([(np.prod(v) if v else 0) for v in pv])
                    parents_shape = pv[biggest_parent]

                    # Scalar parents can support any shape.
                    if np.prod(parents_shape) < 1:
                        parents_shape = None

                else:
                    parents_shape = None

                def shape_error():
                    raise ValueError('Shapes are incompatible: value %s, largest parent %s, shape argument %s'%(shape, init_val_shape, parents_shape))

                if init_val_shape is not None and shape is not None and init_val_shape != shape:
                    shape_error()

                given_shape = init_val_shape or shape
                bindshape = given_shape or parents_shape

                # Check consistency of bindshape and parents_shape
                if parents_shape is not None:
                    # Uncomment to leave broadcasting completely up to NumPy's random functions
                    # if bindshape[-np.alen(parents_shape):]!=parents_shape:
                    # Uncomment to limit broadcasting flexibility to what the Fortran likelihoods can handle.
                    if bindshape<parents_shape:
                        shape_error()

                if random is not None:
                    random = bind_size(random, bindshape)


            elif 'size' in kwds.keys():
                raise ValueError('No size argument allowed for multivariate stochastic variables.')

            
            # Call base class initialization method
            if arg_dict_out.pop('debug'):
                logp = debug_wrapper(logp)
                random = debug_wrapper(random)
            else:
                Stochastic.__init__(self, logp=logp, random=random, logp_partial_gradients = logp_partial_gradients, dtype=dtype, **arg_dict_out)

    new_class.__name__ = name
    new_class.parent_names = parent_names
    new_class.parents_default = parents_default
    new_class.dtype = dtype
    new_class.mv = mv
    new_class.raw_fns = {'logp': logp, 'random': random}

    return new_class


def stochastic_from_dist(name, logp, random=None, logp_partial_gradients={}, dtype=np.float, mv=False):
    """
    Return a Stochastic subclass made from a particular distribution.

    :Parameters:
      name : string
        The name of the new class.
      logp : function
        The log-probability function.
      random : function
        The random function
      dtype : numpy dtype
        The dtype of values of instances.
      mv : boolean
        A flag indicating whether this class represents
        array-valued variables.

    :Example:
      >>> Exponential = stochastic_from_dist('exponential',
                                              logp=exponential_like,
                                              random=rexponential,
                                              dtype=np.float,
                                              mv=False)
      >>> A = Exponential(self_name, value, beta)

    .. note::
      new_dist_class is a more flexible class factory. Also consider
      subclassing Stochastic directly.

    :SeeAlso:
      new_dist_class
    """  

    (args, varargs, varkw, defaults) = inspect.getargspec(logp)
    parent_names = args[1:]
    try:
        parents_default = dict(zip(args[-len(defaults):], defaults))
    except TypeError: # No parents at all.
        parents_default = {}

    name = capitalize(name)

    # Build docstring from distribution
    docstr = name[0]+' = '+name + '(name, '+', '.join(parent_names)+', value=None, observed=False,'
    if not mv:
        docstr += ' size=1,'
    docstr += ' trace=True, rseed=True, doc=None, verbose=-1, debug=False)\n\n'
    docstr += 'Stochastic variable with '+name+' distribution.\nParents are: '+', '.join(parent_names) + '.\n\n'
    docstr += 'Docstring of log-probability function:\n'
    try: docstr += logp.__doc__
    except TypeError: pass  # This will happen when logp doesn't have a docstring
    
    logp=valuewrapper(logp)
    distribution_arguments = logp.__dict__

    wrapped_logp_partial_gradients = {}

    for parameter, func in six.iteritems(logp_partial_gradients):
        wrapped_logp_partial_gradients[parameter] = valuewrapper(logp_partial_gradients[parameter], arguments = distribution_arguments)
    
    return new_dist_class(dtype, name, parent_names, parents_default, docstr,
						 logp, random, mv, wrapped_logp_partial_gradients)


#-------------------------------------------------------------
# Light decorators
#-------------------------------------------------------------

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
      np.asarray([0, 1])
      >>> rbernoulli(.9, size=2)
      np.asarray([1, 1])
      >>> rbernoulli([.1,.9], 2)
      np.asarray([[0, 1],
             [0, 1]])
    """
    
    # Find the order of the arguments.
    refargs, varargs, varkw, defaults = inspect.getargspec(func)
    #vfunc = np.vectorize(self.func)
    npos = len(refargs)-len(defaults) # Number of pos. arg.
    nkwds = len(defaults) # Number of kwds args.
    mv = func.__name__[1:] in mv_continuous_distributions + mv_discrete_distributions

    # Use the NumPy random function directly if this is not a multivariate distribution
    if not mv:
        return func

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
            raise('Dimensions do not agree.')
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
                raise RuntimeError('Arguments size not allowed: %s.' % s)
            largs.append(arr)

        if mv and N >1 and max(dimension)>1 and nr>1:
            raise ValueError('Multivariate distributions cannot take s>1 and multiple values.')

        if mv:
            for i, arg in enumerate(largs[:-1]):
                largs[0] = np.atleast_2d(arg)

        for arg in zip(*largs):
            r.append(func(*arg))

        size = arg[-1]
        vec_stochastics = len(r)>1
        if mv:
            if nr == 1:
                return r[0]
            else:
                return np.vstack(r)
        else:
            if size > 1 and vec_stochastics:
                return np.atleast_2d(r).T
            elif vec_stochastics or size > 1:
                return np.concatenate(r)
            else: # Scalar case
                return r[0][0]

    wrapper.__doc__ = func.__doc__
    wrapper.__name__ = func.__name__
    return wrapper

def debug_wrapper(func, name):
    # Wrapper to debug distributions

    import pdb

    def wrapper(*args, **kwargs):

        print_('Debugging inside %s:' % name)
        print_('\tPress \'s\' to step into function for debugging')
        print_('\tCall \'args\' to list function arguments')

        # Set debugging trace
        pdb.set_trace()

        # Call function
        return func(*args, **kwargs)

    return wrapper


#-------------------------------------------------------------
# Utility functions
#-------------------------------------------------------------

def constrain(value, lower=-np.Inf, upper=np.Inf, allow_equal=False):
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

loss_functions = {'absolute':absolute_loss, 'squared':squared_loss, 'chi_square':chi_square_loss}

def GOFpoints(x,y,expval,loss):
    # Return pairs of points for GOF calculation
    return np.sum(np.transpose([loss(x, expval), loss(y, expval)]), 0)

def gofwrapper(f, loss_function='squared'):
    """
    Goodness-of-fit decorator function for likelihoods
    ==================================================
    Generates goodness-of-fit points for data likelihoods.

    Wrap function f(*args, **kwds) where f is a likelihood.

    Assume args = (x, parameter1, parameter2, ...)
    Before passing the arguments to the function, the wrapper makes sure that
    the parameters have the same shape as x.
    """

    name = f.__name__[:-5]
    # Take a snapshot of the main namespace.

    # Find the functions needed to compute the gof points.
    expval_func = eval(name+'_expval')
    random_func = eval('r'+name)

    def wrapper(*args, **kwds):
        """
        This wraps a likelihood.
        """

        """Return gof points."""

        # Calculate loss
        loss = kwds.pop('gof', loss_functions[loss_function])

        # Expected value, given parameters
        expval = expval_func(*args[1:], **kwds)
        y = random_func(size=len(args[0]), *args[1:], **kwds)
        f.gof_points = GOFpoints(args[0], y, expval, loss)

        """Return likelihood."""

        return f(*args, **kwds)


    # Assign function attributes to wrapper.
    wrapper.__doc__ = f.__doc__
    wrapper.__name__ = f.__name__
    wrapper.name = name

    return wrapper

#--------------------------------------------------------
# Statistical distributions
# random generator, expval, log-likelihood
#--------------------------------------------------------

# Autoregressive lognormal
def rarlognormal(a, sigma, rho, size=1):
    R"""
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
    R"""
    Autoregressive lognormal log-likelihood.

    .. math::
        x_i & = a_i \exp(e_i) \\
        e_i & = \rho e_{i-1} + \epsilon_i

    where :math:`\epsilon_i \sim N(0,\sigma)`.
    """
    return flib.arlognormal(x, np.log(a), sigma, rho, beta=1)


# Bernoulli----------------------------------------------
@randomwrap
def rbernoulli(p,size=None):
    """
    Random Bernoulli variates.
    """

    return np.random.random(size)<p

def bernoulli_expval(p):
    """
    Expected value of bernoulli distribution.
    """

    return p


def bernoulli_like(x, p):
    R"""Bernoulli log-likelihood

    The Bernoulli distribution describes the probability of successes (x=1) and
    failures (x=0).

    .. math::  f(x \mid p) = p^{x} (1-p)^{1-x}

    :Parameters:
      - `x` : Series of successes (1) and failures (0). :math:`x=0,1`
      - `p` : Probability of success. :math:`0 < p < 1`.

    :Example:
       >>> from pymc import bernoulli_like
       >>> bernoulli_like([0,1,0,1], .4)
       -2.854232711280291

    .. note::
      - :math:`E(x)= p`
      - :math:`Var(x)= p(1-p)`

    """

    return flib.bernoulli(x, p)

bernoulli_grad_like = {'p' : flib.bern_grad_p}

# Beta----------------------------------------------
@randomwrap
def rbeta(alpha, beta, size=None):
    """
    Random beta variates.
    """

    return np.random.beta(alpha, beta,size)

def beta_expval(alpha, beta):
    """
    Expected value of beta distribution.
    """

    return 1.0 * alpha / (alpha + beta)


def beta_like(x, alpha, beta):
    R"""
    Beta log-likelihood. The conjugate prior for the parameter
    :math:`p` of the binomial distribution.

    .. math::
        f(x \mid \alpha, \beta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha) \Gamma(\beta)} x^{\alpha - 1} (1 - x)^{\beta - 1}

    :Parameters:
      - `x` : 0 < x < 1
      - `alpha` : alpha > 0
      - `beta` : beta > 0

    :Example:
      >>> from pymc import beta_like
      >>> beta_like(.4,1,2)
      0.182321556793954

    .. note::
      - :math:`E(X)=\frac{\alpha}{\alpha+\beta}`
      - :math:`Var(X)=\frac{\alpha \beta}{(\alpha+\beta)^2(\alpha+\beta+1)}`

    """
    # try:
    #     constrain(alpha, lower=0, allow_equal=True)
    #     constrain(beta, lower=0, allow_equal=True)
    #     constrain(x, 0, 1, allow_equal=True)
    # except ZeroProbability:
    #     return -np.Inf
    return flib.beta_like(x, alpha, beta)

beta_grad_like = {'value' : flib.beta_grad_x,
                  'alpha' : flib.beta_grad_a,
                  'beta' : flib.beta_grad_b}


# Binomial----------------------------------------------
@randomwrap
def rbinomial(n, p, size=None):
    """
    Random binomial variates.
    """
    # return np.random.binomial(n,p,size)
    return np.random.binomial(np.ravel(n),np.ravel(p),size)

def binomial_expval(n, p):
    """
    Expected value of binomial distribution.
    """

    return p*n

def binomial_like(x, n, p):
    R"""
    Binomial log-likelihood.  The discrete probability distribution of the
    number of successes in a sequence of n independent yes/no experiments,
    each of which yields success with probability p.

    .. math::
        f(x \mid n, p) = \frac{n!}{x!(n-x)!} p^x (1-p)^{n-x}

    :Parameters:
      - `x` : [int] Number of successes, > 0.
      - `n` : [int] Number of Bernoulli trials, > x.
      - `p` : Probability of success in each trial, :math:`p \in [0,1]`.

    .. note::
       - :math:`E(X)=np`
       - :math:`Var(X)=np(1-p)`
       
    """

    return flib.binomial(x,n,p)


binomial_grad_like = {'p' : flib.binomial_gp}

# Beta----------------------------------------------
@randomwrap
def rbetabin(alpha, beta, n, size=None):
    """
    Random beta-binomial variates.
    """

    phi = np.random.beta(alpha, beta, size)
    return np.random.binomial(n,phi)

def betabin_expval(alpha, beta, n):
    """
    Expected value of beta-binomial distribution.
    """

    return n * alpha / (alpha + beta)


def betabin_like(x, alpha, beta, n):
    R"""
    Beta-binomial log-likelihood. Equivalent to binomial random
    variables with probabilities drawn from a
    :math:`\texttt{Beta}(\alpha,\beta)` distribution.

    .. math::
        f(x \mid \alpha, \beta, n) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha)} \frac{\Gamma(n+1)}{\Gamma(x+1)\Gamma(n-x+1)} \frac{\Gamma(\alpha + x)\Gamma(n+\beta-x)}{\Gamma(\alpha+\beta+n)}

    :Parameters:
      - `x` : x=0,1,\ldots,n
      - `alpha` : alpha > 0
      - `beta` : beta > 0
      - `n` : n=x,x+1,\ldots

    :Example:
      >>> betabin_like(3,1,1,10)
      -2.3978952727989

    .. note::
      - :math:`E(X)=n\frac{\alpha}{\alpha+\beta}`
      - :math:`Var(X)=n\frac{\alpha \beta}{(\alpha+\beta)^2(\alpha+\beta+1)}`

    """
    return flib.betabin_like(x, alpha, beta, n)

betabin_grad_like = {'alpha' : flib.betabin_ga,
                 'beta' : flib.betabin_gb}

# Categorical----------------------------------------------
# Note that because categorical elements are not ordinal, there
# is no expected value.

#@randomwrap
def rcategorical(p, size=None):
    """
    Categorical random variates.
    """
    out = flib.rcat(p, np.random.random(size=size))
    if sum(out.shape) == 1:
        return out.squeeze()
    else:
        return out

def categorical_like(x, p):
    R"""
    Categorical log-likelihood. The most general discrete distribution.

    .. math::  f(x=i \mid p) = p_i

    for :math:`i \in 0 \ldots k-1`.

    :Parameters:
      - `x` : [int] :math:`x \in 0\ldots k-1`
      - `p` : [float] :math:`p > 0`, :math:`\sum p = 1`
      
    """
    
    p = np.atleast_2d(p)
    if any(abs(np.sum(p, 1)-1)>0.0001):
        print_("Probabilities in categorical_like sum to", np.sum(p, 1))
    if np.array(x).dtype != int:
        #print_("Non-integer values in categorical_like")
        return -inf
    return flib.categorical(x, p)


# Cauchy----------------------------------------------
@randomwrap
def rcauchy(alpha, beta, size=None):
    """
    Returns Cauchy random variates.
    """

    return alpha + beta*np.tan(pi*random_number(size) - pi/2.0)

def cauchy_expval(alpha, beta):
    """
    Expected value of cauchy distribution.
    """

    return alpha

# In wikipedia, the arguments name are k, x0.
def cauchy_like(x, alpha, beta):
    R"""
    Cauchy log-likelihood. The Cauchy distribution is also known as the
    Lorentz or the Breit-Wigner distribution.

    .. math::
        f(x \mid \alpha, \beta) = \frac{1}{\pi \beta [1 + (\frac{x-\alpha}{\beta})^2]}

    :Parameters:
      - `alpha` : Location parameter.
      - `beta` : Scale parameter > 0.

    .. note::
       - Mode and median are at alpha.
       
    """

    return flib.cauchy(x,alpha,beta)

cauchy_grad_like = {'value' : flib.cauchy_grad_x,
                 'alpha' : flib.cauchy_grad_a,
                 'beta' : flib.cauchy_grad_b}

# Chi square----------------------------------------------
@randomwrap
def rchi2(nu, size=None):
    """
    Random :math:`\chi^2` variates.
    """

    return np.random.chisquare(nu, size)

def chi2_expval(nu):
    """
    Expected value of Chi-squared distribution.
    """

    return nu

def chi2_like(x, nu):
    R"""
    Chi-squared :math:`\chi^2` log-likelihood.

    .. math::
        f(x \mid \nu) = \frac{x^{(\nu-2)/2}e^{-x/2}}{2^{\nu/2}\Gamma(\nu/2)}

    :Parameters:
      - `x` : > 0
      - `nu` : [int] Degrees of freedom ( nu > 0 )

    .. note::
      - :math:`E(X)=\nu`
      - :math:`Var(X)=2\nu`

    """

    return flib.gamma(x, 0.5*nu, 1./2)

chi2_grad_like = {'value'  : lambda x, nu : flib.gamma_grad_x    (x, 0.5* nu, 1./2),
                  'nu' : lambda x, nu : flib.gamma_grad_alpha(x, 0.5* nu, 1./2) * .5}

#chi2_grad_like = {'x'  : lambda x, nu : (nu / 2 - 1) / x -.5,
#                  'nu' : flib.chi2_grad_nu }

# Degenerate---------------------------------------------
@randomwrap
def rdegenerate(k, size=1):
    """
    Random degenerate variates.
    """
    return np.ones(size)*k

def degenerate_expval(k):
    """
    Expected value of degenerate distribution.
    """
    return k

def degenerate_like(x, k):
    R"""
    Degenerate log-likelihood.

    .. math::
        f(x \mid k) = \left\{ \begin{matrix} 1 \text{ if } x = k \\ 0 \text{ if } x \ne k\end{matrix} \right.

    :Parameters:
      - `x` : Input value.
      - `k` : Degenerate value.
      
    """
    x = np.atleast_1d(x)
    return sum(np.log([i==k for i in x]))

#def degenerate_grad_like(x, k):
#    R"""
#    degenerate_grad_like(x, k)
#
#    Degenerate gradient log-likelihood.
#
#    .. math::
#        f(x \mid k) = \left\{ \begin{matrix} 1 \text{ if } x = k \\ 0 \text{ if } x \ne k\end{matrix} \right.
#
#    :Parameters:
#      - `x` : Input value.
#      - `k` : Degenerate value.
#    """
#    return np.zeros(np.size(x))*k

# Dirichlet----------------------------------------------
@randomwrap
def rdirichlet(theta, size=1):
    """
    Dirichlet random variates.
    """
    gammas = np.vstack([rgamma(theta,1) for i in xrange(size)])
    if size > 1 and np.size(theta) > 1:
        return (gammas.T/gammas.sum(1))[:-1].T
    elif np.size(theta)>1:
        return (gammas[0]/gammas[0].sum())[:-1]
    else:
        return 1.

def dirichlet_expval(theta):
    """
    Expected value of Dirichlet distribution.
    """
    return theta/np.sum(theta).astype(float)

def dirichlet_like(x, theta):
    R"""
    Dirichlet log-likelihood.

    This is a multivariate continuous distribution.

    .. math::
        f(\mathbf{x}) = \frac{\Gamma(\sum_{i=1}^k \theta_i)}{\prod \Gamma(\theta_i)}\prod_{i=1}^{k-1} x_i^{\theta_i - 1}
        \cdot\left(1-\sum_{i=1}^{k-1}x_i\right)^\theta_k

    :Parameters:
      x : (n, k-1) array
        Array of shape (n, k-1) where `n` is the number of samples
        and `k` the dimension.
        :math:`0 < x_i < 1`,  :math:`\sum_{i=1}^{k-1} x_i < 1`
      theta : array
        An (n,k) or (1,k) array > 0.

    .. note::
        Only the first `k-1` elements of `x` are expected. Can be used
        as a parent of Multinomial and Categorical nevertheless.
        
    """
    x = np.atleast_2d(x)
    theta = np.atleast_2d(theta)
    if (np.shape(x)[-1]+1) != np.shape(theta)[-1]:
        raise ValueError('The dimension of x in dirichlet_like must be k-1.')
    return flib.dirichlet(x,theta)

# Exponential----------------------------------------------
@randomwrap
def rexponential(beta, size=None):
    """
    Exponential random variates.
    """

    return np.random.exponential(1./beta,size)

def exponential_expval(beta):
    """
    Expected value of exponential distribution.
    """
    return 1./beta


def exponential_like(x, beta):
    R"""
    Exponential log-likelihood.

    The exponential distribution is a special case of the gamma distribution
    with alpha=1. It often describes the time until an event.

    .. math:: f(x \mid \beta) = \beta e^{-\beta x}

    :Parameters:
      - `x` : x > 0
      - `beta` : Rate parameter (beta > 0).

    .. note::
      - :math:`E(X) = 1/\beta`
      - :math:`Var(X) = 1/\beta^2`
      - PyMC's beta is named 'lambda' by Wikipedia, SciPy, Wolfram MathWorld and other sources.
      
    """

    return flib.gamma(x, 1, beta)

exponential_grad_like = {'value' : lambda x, beta : flib.gamma_grad_x(x, 1.0, beta),
                         'beta' : lambda x, beta : flib.gamma_grad_beta(x, 1.0, beta)}

# Exponentiated Weibull-----------------------------------
@randomwrap
def rexponweib(alpha, k, loc=0, scale=1, size=None):
    """
    Random exponentiated Weibull variates.
    """

    q = np.random.uniform(size=size)
    r = flib.exponweib_ppf(q,alpha,k)
    return loc + r*scale

def exponweib_expval(alpha, k, loc, scale):
    # Not sure how we can do this, since the first moment is only
    # tractable at particular values of k
    raise NotImplementedError('exponweib_expval has not been implemented yet.')

def exponweib_like(x, alpha, k, loc=0, scale=1):
    R"""
    Exponentiated Weibull log-likelihood.

    The exponentiated Weibull distribution is a generalization of the Weibull
    family. Its value lies in being able to model monotone and non-monotone
    failure rates.

    .. math::
        f(x \mid \alpha,k,loc,scale)  & = \frac{\alpha k}{scale} (1-e^{-z^k})^{\alpha-1} e^{-z^k} z^{k-1} \\
        z & = \frac{x-loc}{scale}

    :Parameters:
      - `x` : x > 0
      - `alpha` : Shape parameter
      - `k` : k > 0
      - `loc` : Location parameter
      - `scale` : Scale parameter (scale > 0).

    """
    return flib.exponweib(x,alpha,k,loc,scale)

"""
commented out because tests fail
exponweib_grad_like = {'value' : flib.exponweib_gx,
                   'alpha' : flib.exponweib_ga,
                   'k' : flib.exponweib_gk,
                   'loc' : flib.exponweib_gl,
                   'scale' : flib.exponweib_gs}
"""
# Gamma----------------------------------------------
@randomwrap
def rgamma(alpha, beta, size=None):
    """
    Random gamma variates.
    """

    return np.random.gamma(shape=alpha,scale=1./beta,size=size)

def gamma_expval(alpha, beta):
    """
    Expected value of gamma distribution.
    """
    return 1. * np.asarray(alpha) / beta

def gamma_like(x, alpha, beta):
    R"""
    Gamma log-likelihood.

    Represents the sum of alpha exponentially distributed random variables, each
    of which has mean beta.

    .. math::
        f(x \mid \alpha, \beta) = \frac{\beta^{\alpha}x^{\alpha-1}e^{-\beta x}}{\Gamma(\alpha)}

    :Parameters:
      - `x` : math:`x \ge 0`
      - `alpha` : Shape parameter (alpha > 0).
      - `beta` : Rate parameter (beta > 0).

    .. note::
       - :math:`E(X) = \frac{\alpha}{\beta}`
       - :math:`Var(X) = \frac{\alpha}{\beta^2}`

    """

    return flib.gamma(x, alpha, beta)


gamma_grad_like = {'value'     : flib.gamma_grad_x,
                   'alpha' : flib.gamma_grad_alpha,
                   'beta'  : flib.gamma_grad_beta}

# GEV Generalized Extreme Value ------------------------
# Modify parameterization -> Hosking (kappa, xi, alpha)
@randomwrap
def rgev(xi, mu=0, sigma=1, size=None):
    """
    Random generalized extreme value (GEV) variates.
    """

    q = np.random.uniform(size=size)
    z = flib.gev_ppf(q,xi)
    return z*sigma + mu

def gev_expval(xi, mu=0, sigma=1):
    """
    Expected value of generalized extreme value distribution.
    """
    return mu - (sigma / xi) + (sigma / xi) * flib.gamfun(1 - xi)

def gev_like(x, xi, mu=0, sigma=1):
    R"""
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
def rgeometric(p, size=None):
    """
    Random geometric variates.
    """

    return np.random.geometric(p, size)

def geometric_expval(p):
    """
    Expected value of geometric distribution.
    """
    return 1. / p

def geometric_like(x, p):
    R"""
    Geometric log-likelihood. The probability that the first success in a
    sequence of Bernoulli trials occurs on the x'th trial.

    .. math::
        f(x \mid p) = p(1-p)^{x-1}

    :Parameters:
      - `x` : [int] Number of trials before first success (x > 0).
      - `p` : Probability of success on an individual trial, :math:`p \in [0,1]`

    .. note::
      - :math:`E(X)=1/p`
      - :math:`Var(X)=\frac{1-p}{p^2}`

    """

    return flib.geometric(x, p)

geometric_grad_like = {'p' : flib.geometric_gp}

# Half Cauchy----------------------------------------------
@randomwrap
def rhalf_cauchy(alpha, beta, size=None):
    """
    Returns half-Cauchy random variates.
    """

    return abs(alpha + beta*np.tan(pi*random_number(size) - pi/2.0))

def half_cauchy_expval(alpha, beta):
    """
    Expected value of cauchy distribution is undefined.
    """

    return inf

# In wikipedia, the arguments name are k, x0.
def half_cauchy_like(x, alpha, beta):
    R"""
    Half-Cauchy log-likelihood. Simply the absolute value of Cauchy.

    .. math::
        f(x \mid \alpha, \beta) = \frac{2}{\pi \beta [1 + (\frac{x-\alpha}{\beta})^2]}

    :Parameters:
      - `alpha` : Location parameter.
      - `beta` : Scale parameter (beta > 0).

    .. note::
      - x must be non-negative.
    """

    x = np.atleast_1d(x)
    if sum(x<0): return -inf
    return flib.cauchy(x,alpha,beta) + len(x)*np.log(2)

# Half-normal----------------------------------------------
@randomwrap
def rhalf_normal(tau, size=None):
    """
    Random half-normal variates.
    """

    return abs(np.random.normal(0, np.sqrt(1/tau), size))

def half_normal_expval(tau):
    """
    Expected value of half normal distribution.
    """

    return np.sqrt(2. * pi / np.asarray(tau))

def half_normal_like(x, tau):
    R"""
    Half-normal log-likelihood, a normal distribution with mean 0 limited
    to the domain :math:`x \in [0, \infty)`.

    .. math::
        f(x \mid \tau) = \sqrt{\frac{2\tau}{\pi}}\exp\left\{ {\frac{-x^2 \tau}{2}}\right\}

    :Parameters:
      - `x` : :math:`x \ge 0`
      - `tau` : tau > 0

    """

    return flib.hnormal(x, tau)

half_normal_grad_like = {'value'   : flib.hnormal_gradx,
                 'tau' : flib.hnormal_gradtau}

# Hypergeometric----------------------------------------------
def rhypergeometric(n, m, N, size=None):
    """
    Returns hypergeometric random variates.
    """
    if n==0:
        return np.zeros(size,dtype=int)
    elif n==N:
        out = np.empty(size,dtype=int)
        out.fill(m)
        return out
    return np.random.hypergeometric(n, N-n, m, size)

def hypergeometric_expval(n, m, N):
    """
    Expected value of hypergeometric distribution.
    """
    return 1. * n * m / N

def hypergeometric_like(x, n, m, N):
    R"""
    Hypergeometric log-likelihood. 
    
    Discrete probability distribution that describes the number of successes in 
    a sequence of draws from a finite population without replacement.

    .. math::

        f(x \mid n, m, N) = \frac{\left({ \begin{array}{c} {m} \\ {x} \\ 
        \end{array} }\right)\left({ \begin{array}{c} {N-m} \\ {n-x} \\ 
        \end{array}}\right)}{\left({ \begin{array}{c} {N} \\ {n} \\ 
        \end{array}}\right)}


    :Parameters:
      - `x` : [int] Number of successes in a sample drawn from a population.
      - `n` : [int] Size of sample drawn from the population.
      - `m` : [int] Number of successes in the population.
      - `N` : [int] Total number of units in the population.

    .. note::
        
        :math:`E(X) = \frac{n n}{N}`
    """

    return flib.hyperg(x, n, m, N)

# Inverse gamma----------------------------------------------
@randomwrap
def rinverse_gamma(alpha, beta,size=None):
    """
    Random inverse gamma variates.
    """

    return 1. / np.random.gamma(shape=alpha, scale=1./beta, size=size)

def inverse_gamma_expval(alpha, beta):
    """
    Expected value of inverse gamma distribution.
    """
    return 1. * np.asarray(beta) / (alpha-1.)

def inverse_gamma_like(x, alpha, beta):
    R"""
    Inverse gamma log-likelihood, the reciprocal of the gamma distribution.

    .. math::
        f(x \mid \alpha, \beta) = \frac{\beta^{\alpha}}{\Gamma(\alpha)} x^{-\alpha - 1} \exp\left(\frac{-\beta}{x}\right)

    :Parameters:
      - `x` : x > 0
      - `alpha` : Shape parameter (alpha > 0).
      - `beta` : Scale parameter (beta > 0).

    .. note::
    
       :math:`E(X)=\frac{\beta}{\alpha-1}`  for :math:`\alpha > 1`
       :math:`Var(X)=\frac{\beta^2}{(\alpha-1)^2(\alpha)}`  for :math:`\alpha > 2`
       
    """

    return flib.igamma(x, alpha, beta)

inverse_gamma_grad_like = {'value' : flib.igamma_grad_x,
             'alpha' : flib.igamma_grad_alpha,
             'beta' : flib.igamma_grad_beta}

# Inverse Wishart---------------------------------------------------

def rinverse_wishart(n, C):
    """
    Return an inverse Wishart random matrix.

    :Parameters:
      - `n` : [int] Degrees of freedom (n > 0).
      - `C` : Symmetric and positive definite scale matrix
    """
    wi = rwishart(n, np.asmatrix(C).I).I
    flib.symmetrize(wi)
    return wi

def inverse_wishart_expval(n, C):
    """
    Expected value of inverse Wishart distribution.
    
    :Parameters:
      - `n` : [int] Degrees of freedom (n > 0).
      - `C` : Symmetric and positive definite scale matrix
    
    """
    return np.asarray(C)/(n-len(C)-1)

def inverse_wishart_like(X, n, C):
    R"""
    Inverse Wishart log-likelihood. The inverse Wishart distribution
    is the conjugate prior for the covariance matrix of a multivariate
    normal distribution.

    .. math::
        f(X \mid n, T) = \frac{{\mid T \mid}^{n/2}{\mid X
        \mid}^{(n-k-1)/2} \exp\left\{ -\frac{1}{2} Tr(TX^{-1})
        \right\}}{2^{nk/2} \Gamma_p(n/2)}

    where :math:`k` is the rank of X.

    :Parameters:
      - `X` : Symmetric, positive definite matrix.
      - `n` : [int] Degrees of freedom (n > 0).
      - `C` : Symmetric and positive definite scale matrix

    .. note::
       Step method MatrixMetropolis will preserve the symmetry of
       Wishart variables.

    """
    return flib.blas_inv_wishart(X, n, C)

def rinverse_wishart_prec(n, Tau):
    """
    Return an inverse Wishart random matrix.

    :Parameters:
      - `n` : [int] Degrees of freedom (n > 0).
      - `Tau` : Symmetric and positive definite precision matrix

    """
    wi = rwishart(n, np.asmatrix(Tau)).I
    flib.symmetrize(wi)
    return wi

def inverse_wishart_prec_expval(X, n, Tau):
    """
    Expected value of inverse Wishart distribution.

    :Parameters:
      - `n` : [int] Degrees of freedom (n > 0).
      - `Tau` : Symmetric and positive definite precision matrix

    """
    return inverse_wishart_like(X, n, inverse(Tau))

def inverse_wishart_prec_like(X, n, Tau):
    """
    Inverse Wishart log-likelihood

    For an alternative parameterization based on :math:`C=Tau^{-1}`, see
    `inverse_wishart_like`.

    :Parameters:
      - `X` : Symmetric, positive definite matrix.
      - `n` : [int] Degrees of freedom (n > 0).
      - `Tau` : Symmetric and positive definite precision matrix

    """
    return inverse_wishart_like(X, n, inverse(Tau))

# Double exponential (Laplace)--------------------------------------------
@randomwrap
def rlaplace(mu, tau, size=None):
    """
    Laplace (double exponential) random variates.
    """

    u = np.random.uniform(-0.5, 0.5, size)
    return mu - np.sign(u)*np.log(1 - 2*np.abs(u))/tau

rdexponential = rlaplace

def laplace_expval(mu, tau):
    """
    Expected value of Laplace (double exponential) distribution.
    """
    return mu

dexponential_expval = laplace_expval

def laplace_like(x, mu, tau):
    R"""
    Laplace (double exponential) log-likelihood.

    The Laplace (or double exponential) distribution describes the
    difference between two independent, identically distributed exponential
    events. It is often used as a heavier-tailed alternative to the normal.

    .. math::
        f(x \mid \mu, \tau) = \frac{\tau}{2}e^{-\tau |x-\mu|}

    :Parameters:
      - `x` : :math:`-\infty < x < \infty`
      - `mu` : Location parameter :math: `-\infty < mu < \infty`
      - `tau` : Scale parameter :math:`\tau > 0`

    .. note::
      - :math:`E(X) = \mu`
      - :math:`Var(X) = \frac{2}{\tau^2}`
    """

    return flib.gamma(np.abs(x-mu), 1, tau) - np.log(2)

laplace_grad_like = {'value'   : lambda x, mu, tau: flib.gamma_grad_x(np.abs(x- mu), 1, tau) * np.sign(x - mu),
                     'mu'  : lambda x, mu, tau: -flib.gamma_grad_x(np.abs(x- mu), 1, tau) * np.sign(x - mu),
                     'tau' : lambda x, mu, tau: flib.gamma_grad_beta(np.abs(x- mu), 1, tau)}


dexponential_like = laplace_like
dexponential_grad_like = laplace_grad_like

# Logistic-----------------------------------
@randomwrap
def rlogistic(mu, tau, size=None):
    """
    Logistic random variates.
    """

    u = np.random.random(size)
    return mu + np.log(u/(1-u))/tau


def logistic_expval(mu, tau):
    """
    Expected value of logistic distribution.
    """
    return mu


def logistic_like(x, mu, tau):
    R"""
    Logistic log-likelihood.

    The logistic distribution is often used as a growth model; for example,
    populations, markets. Resembles a heavy-tailed normal distribution.

    .. math::
        f(x \mid \mu, tau) = \frac{\tau \exp(-\tau[x-\mu])}{[1 + \exp(-\tau[x-\mu])]^2}

    :Parameters:
      - `x` : :math:`-\infty < x < \infty`
      - `mu` : Location parameter :math:`-\infty < mu < \infty`
      - `tau` : Scale parameter (tau > 0)

    .. note::
      - :math:`E(X) = \mu`
      - :math:`Var(X) = \frac{\pi^2}{3\tau^2}`
    """

    return flib.logistic(x, mu, tau)


# Lognormal----------------------------------------------
@randomwrap
def rlognormal(mu, tau,size=None):
    """
    Return random lognormal variates.
    """

    return np.random.lognormal(mu, np.sqrt(1./tau),size)

def lognormal_expval(mu, tau):
    """
    Expected value of log-normal distribution.
    """
    return np.exp(mu + 1./2/tau)

def lognormal_like(x, mu, tau):
    R"""
    Log-normal log-likelihood. 
    
    Distribution of any random variable whose logarithm is normally 
    distributed. A variable might be modeled as log-normal if it can be thought 
    of as the multiplicative product of many small independent factors.

    .. math::
        f(x \mid \mu, \tau) = \sqrt{\frac{\tau}{2\pi}}\frac{
        \exp\left\{ -\frac{\tau}{2} (\ln(x)-\mu)^2 \right\}}{x}

    :Parameters:
      - `x` : x > 0
      - `mu` : Location parameter.
      - `tau` : Scale parameter (tau > 0).

    .. note::
    
       :math:`E(X)=e^{\mu+\frac{1}{2\tau}}`
       :math:`Var(X)=(e^{1/\tau}-1)e^{2\mu+\frac{1}{\tau}}`

    """
    return flib.lognormal(x,mu,tau)

lognormal_grad_like = {'value'   : flib.lognormal_gradx,
                       'mu'  : flib.lognormal_gradmu,
                       'tau' : flib.lognormal_gradtau}


# Multinomial----------------------------------------------
#@randomwrap
def rmultinomial(n,p,size=None):
    """
    Random multinomial variates.
    """
    # Leaving size=None as the default means return value is 1d array
    # if not specified-- nicer.

    # Single value for p:
    if len(np.shape(p))==1:
        return np.random.multinomial(n,p,size)

    # Multiple values for p:
    if np.isscalar(n):
        n = n * np.ones(np.shape(p)[0],dtype=np.int)
    out = np.empty(np.shape(p))
    for i in xrange(np.shape(p)[0]):
        out[i,:] = np.random.multinomial(n[i],p[i,:],size)
    return out

def multinomial_expval(n,p):
    """
    Expected value of multinomial distribution.
    """
    return np.asarray([pr * n for pr in p])

def multinomial_like(x, n, p):
    R"""
    Multinomial log-likelihood. 
    
    Generalization of the binomial
    distribution, but instead of each trial resulting in "success" or
    "failure", each one results in exactly one of some fixed finite number k
    of possible outcomes over n independent trials. 'x[i]' indicates the number
    of times outcome number i was observed over the n trials.

    .. math::
        f(x \mid n, p) = \frac{n!}{\prod_{i=1}^k x_i!} \prod_{i=1}^k p_i^{x_i}

    :Parameters:
      x : (ns, k) int
          Random variable indicating the number of time outcome i is
          observed. :math:`\sum_{i=1}^k x_i=n`, :math:`x_i \ge 0`.
      n : int
          Number of trials.
      p : (k,)
          Probability of each one of the different outcomes.
          :math:`\sum_{i=1}^k p_i = 1)`, :math:`p_i \ge 0`.

    .. note::
       - :math:`E(X_i)=n p_i`
       - :math:`Var(X_i)=n p_i(1-p_i)`
       - :math:`Cov(X_i,X_j) = -n p_i p_j`
       - If :math: `\sum_i p_i < 0.999999` a log-likelihood value of -inf
       will be returned.

    """
    # flib expects 2d arguments. Do we still want to support multiple p
    # values along realizations ?
    x = np.atleast_2d(x)
    p = np.atleast_2d(p)

    return flib.multinomial(x, n, p)

# Multivariate hypergeometric------------------------------
def rmultivariate_hypergeometric(n, m, size=None):
    """
    Random multivariate hypergeometric variates.

    Parameters:
      - `n` : Number of draws.
      - `m` : Number of items in each categoy.
    """

    N = len(m)
    urn = np.repeat(np.arange(N), m)

    if size:
        draw = np.array([[urn[i] for i in np.random.permutation(len(urn))[:n]]
                         for j in range(size)])

        r = [[np.sum(draw[j]==i) for i in range(len(m))]
             for j in range(size)]
    else:
        draw = np.array([urn[i] for i in np.random.permutation(len(urn))[:n]])

        r = [np.sum(draw==i) for i in range(len(m))]
    return np.asarray(r)

def multivariate_hypergeometric_expval(n, m):
    """
    Expected value of multivariate hypergeometric distribution.

    Parameters:
      - `n` : Number of draws.
      - `m` : Number of items in each categoy.
    """
    m= np.asarray(m, float)
    return n * (m / m.sum())


def multivariate_hypergeometric_like(x, m):
    R"""
    Multivariate hypergeometric log-likelihood
    
    Describes the probability of drawing x[i] elements of the ith category, 
    when the number of items in each category is given by m.

    .. math::
        \frac{\prod_i \left({ \begin{array}{c} {m_i} \\ {x_i} \\ 
        \end{array}}\right)}{\left({ \begin{array}{c} {N} \\ {n} \\ 
        \end{array}}\right)}


    where :math:`N = \sum_i m_i` and :math:`n = \sum_i x_i`.

    :Parameters:
      - `x` : [int sequence] Number of draws from each category, (x < m).
      - `m` : [int sequence] Number of items in each categoy.
      
    """
    return flib.mvhyperg(x, m)


# Multivariate normal--------------------------------------
def rmv_normal(mu, tau, size=1):
    """
    Random multivariate normal variates.
    """

    sig = np.linalg.cholesky(tau)
    mu_size = np.shape(mu)

    if size==1:
        out = np.random.normal(size=mu_size)
        try:
            flib.dtrsm_wrap(sig , out, 'L', 'T', 'L', 1.)
        except:
            out = np.linalg.solve(sig, out)
        out+=mu
        return out
    else:
        if not hasattr(size,'__iter__'):
            size = (size,)
        tot_size = np.prod(size)
        out = np.random.normal(size = (tot_size,) + mu_size)
        for i in xrange(tot_size):
            try:
                flib.dtrsm_wrap(sig , out[i,:], 'L', 'T', 'L', 1.)
            except:
                out[i,:] = np.linalg.solve(sig, out[i,:])
            out[i,:] += mu
        return out.reshape(size+mu_size)

def mv_normal_expval(mu, tau):
    """
    Expected value of multivariate normal distribution.
    """
    return mu

def mv_normal_like(x, mu, tau):
    R"""
    Multivariate normal log-likelihood

    .. math::
        f(x \mid \pi, T) = \frac{|T|^{1/2}}{(2\pi)^{1/2}} \exp\left\{ -\frac{1}{2} (x-\mu)^{\prime}T(x-\mu) \right\}

    :Parameters:
      - `x` : (n,k)
      - `mu` : (k) Location parameter sequence.
      - `Tau` : (k,k) Positive definite precision matrix.

    .. seealso:: :func:`mv_normal_chol_like`, :func:`mv_normal_cov_like`

    """
    # TODO: Vectorize in Fortran
    if len(np.shape(x))>1:
        return np.sum([flib.prec_mvnorm(r,mu,tau) for r in x])
    else:
        return flib.prec_mvnorm(x,mu,tau)

# Multivariate normal, parametrized with covariance---------------------------
def rmv_normal_cov(mu, C, size=1):
    """
    Random multivariate normal variates.
    """
    mu_size = np.shape(mu)
    if size==1:
        return np.random.multivariate_normal(mu, C, size).reshape(mu_size)
    else:
        return np.random.multivariate_normal(mu, C, size).reshape((size,)+mu_size)

def mv_normal_cov_expval(mu, C):
    """
    Expected value of multivariate normal distribution.
    """
    return mu

def mv_normal_cov_like(x, mu, C):
    R"""
    Multivariate normal log-likelihood parameterized by a covariance
    matrix.

    .. math::
        f(x \mid \pi, C) = \frac{1}{(2\pi|C|)^{1/2}} \exp\left\{ -\frac{1}{2} (x-\mu)^{\prime}C^{-1}(x-\mu) \right\}

    :Parameters:
      - `x` : (n,k)
      - `mu` : (k) Location parameter.
      - `C` : (k,k) Positive definite covariance matrix.

    .. seealso:: :func:`mv_normal_like`, :func:`mv_normal_chol_like`

    """
    # TODO: Vectorize in Fortran
    if len(np.shape(x))>1:
        return np.sum([flib.cov_mvnorm(r,mu,C) for r in x])
    else:
        return flib.cov_mvnorm(x,mu,C)


# Multivariate normal, parametrized with Cholesky factorization.----------
def rmv_normal_chol(mu, sig, size=1):
    """
    Random multivariate normal variates.
    """
    mu_size = np.shape(mu)

    if size==1:
        out = np.random.normal(size=mu_size)
        try:
            flib.dtrmm_wrap(sig , out, 'L', 'N', 'L', 1.)
        except:
            out = np.dot(sig, out)
        out+=mu
        return out
    else:
        if not hasattr(size,'__iter__'):
            size = (size,)
        tot_size = np.prod(size)
        out = np.random.normal(size = (tot_size,) + mu_size)
        for i in xrange(tot_size):
            try:
                flib.dtrmm_wrap(sig , out[i,:], 'L', 'N', 'L', 1.)
            except:
                out[i,:] = np.dot(sig, out[i,:])
            out[i,:] += mu
        return out.reshape(size+mu_size)


def mv_normal_chol_expval(mu, sig):
    """
    Expected value of multivariate normal distribution.
    """
    return mu

def mv_normal_chol_like(x, mu, sig):
    R"""
    Multivariate normal log-likelihood.

    .. math::
        f(x \mid \pi, \sigma) = \frac{1}{(2\pi)^{1/2}|\sigma|)} \exp\left\{ -\frac{1}{2} (x-\mu)^{\prime}(\sigma \sigma^{\prime})^{-1}(x-\mu) \right\}

    :Parameters:
      - `x` : (n,k)
      - `mu` : (k) Location parameter.
      - `sigma` : (k,k) Lower triangular matrix.

    .. seealso:: :func:`mv_normal_like`, :func:`mv_normal_cov_like`

    """
    # TODO: Vectorize in Fortran
    if len(np.shape(x))>1:
        return np.sum([flib.chol_mvnorm(r,mu,sig) for r in x])
    else:
        return flib.chol_mvnorm(x,mu,sig)



# Negative binomial----------------------------------------
@randomwrap
def rnegative_binomial(mu, alpha, size=None):
    """
    Random negative binomial variates.
    """
    # Using gamma-poisson mixture rather than numpy directly
    # because numpy apparently rounds
    mu = np.asarray(mu, dtype=float)
    pois_mu = np.random.gamma(alpha, mu/alpha, size)
    return np.random.poisson(pois_mu, size)
    # return np.random.negative_binomial(alpha, alpha / (mu + alpha), size)

def negative_binomial_expval(mu, alpha):
    """
    Expected value of negative binomial distribution.
    """
    return mu


def negative_binomial_like(x, mu, alpha):
    R"""
    Negative binomial log-likelihood. 
    
    The negative binomial
    distribution describes a Poisson random variable whose rate
    parameter is gamma distributed. PyMC's chosen parameterization is
    based on this mixture interpretation.

    .. math::
        f(x \mid \mu, \alpha) = \frac{\Gamma(x+\alpha)}{x! \Gamma(\alpha)} (\alpha/(\mu+\alpha))^\alpha (\mu/(\mu+\alpha))^x

    :Parameters:
      - `x` : Input data (x > 0).
      - `mu` : mu > 0
      - `alpha` : alpha > 0

    .. note::
      - :math:`E[x]=\mu`
      - In Wikipedia's parameterization,
        :math:`r=\alpha`
        :math:`p=\alpha/(\mu+\alpha)`
        :math:`\mu=r(1-p)/p`

    """
    if any(alpha > 1e10):
        # Return Poisson when alpha gets very large
        return flib.poisson(x, mu)
    return flib.negbin2(x, mu, alpha)

negative_binomial_grad_like = {'mu'    : flib.negbin2_gmu,
                               'alpha' : flib.negbin2_ga}

# Normal---------------------------------------------------
@randomwrap
def rnormal(mu, tau,size=None):
    """
    Random normal variates.
    """
    return np.random.normal(mu, 1./np.sqrt(tau), size)

def normal_expval(mu, tau):
    """
    Expected value of normal distribution.
    """
    return mu

def normal_like(x, mu, tau):
    R"""
    Normal log-likelihood.

    .. math::
        f(x \mid \mu, \tau) = \sqrt{\frac{\tau}{2\pi}} \exp\left\{ -\frac{\tau}{2} (x-\mu)^2 \right\}

    :Parameters:
      - `x` : Input data.
      - `mu` : Mean of the distribution.
      - `tau` : Precision of the distribution, which corresponds to
        :math:`1/\sigma^2` (tau > 0).

    .. note::
       - :math:`E(X) = \mu`
       - :math:`Var(X) = 1/\tau`

    """
    # try:
    #     constrain(tau, lower=0)
    # except ZeroProbability:
    #     return -np.Inf

    return flib.normal(x, mu, tau)

def t_normal_grad_x(x, mu, tau):
    return flib.normal_grad_x(x,mu, tau)

def t_normal_grad_mu(x, mu, tau):
    return flib.normal_grad_mu(x,mu, tau)
def t_normal_grad_tau(x, mu, tau):
    return flib.normal_grad_tau(x,mu, tau)

normal_grad_like = {'value' : t_normal_grad_x,
             'mu' : t_normal_grad_mu,
             'tau' : t_normal_grad_tau}

#normal_grad_like = {'x' : flib.normal_grad_x,
#             'mu' : flib.normal_grad_mu,
#             'tau' : flib.normal_grad_tau}

# von Mises--------------------------------------------------
@randomwrap
def rvon_mises(mu, kappa, size=None):
    """
    Random von Mises variates.
    """
    # TODO: Just return straight from numpy after release 1.3
    return (np.random.mtrand.vonmises(mu, kappa, size) + np.pi)%(2.*np.pi)-np.pi

def von_mises_expval(mu, kappa):
    """
    Expected value of von Mises distribution.
    """
    return mu

def von_mises_like(x, mu, kappa):
    R"""
    von Mises log-likelihood.

    .. math::
        f(x \mid \mu, k) = \frac{e^{k \cos(x - \mu)}}{2 \pi I_0(k)}

    where `I_0` is the modified Bessel function of order 0.

    :Parameters:
      - `x` : Input data.
      - `mu` : Mean of the distribution.
      - `kappa` : Dispersion of the distribution

    .. note::
       - :math:`E(X) = \mu`

    """
    return flib.vonmises(x, mu, kappa)
    
# Pareto---------------------------------------------------
@randomwrap
def rpareto(alpha, m, size=None):
    """
    Random Pareto variates.
    """
    return m / (random_number(size)**(1./alpha))
    
def pareto_expval(alpha, m):
    """
    Expected value of Pareto distribution.
    """
    
    if alpha <= 1:
        return inf
    return alpha*m/(alpha-1)
    
def pareto_like(x, alpha, m):
    R"""
    Pareto log-likelihood. The Pareto is a continuous, positive 
    probability distribution with two parameters. It is often used
    to characterize wealth distribution, or other examples of the
    80/20 rule.

    .. math::
        f(x \mid \alpha, m) = \frac{\alpha m^{\alpha}}{x^{\alpha+1}}

    :Parameters:
      - `x` : Input data (x > m)
      - `alpha` : Shape parameter (alpha>0)
      - `m` : Scale parameter (m>0)

    .. note::
       - :math:`E(x)=\frac{\alpha m}{\alpha-1} if \alpha > 1`
       - :math:`Var(x)=\frac{m^2 \alpha}{(\alpha-1)^2(\alpha-2)} if \alpha > 2`

    """
    return flib.pareto(x, alpha, m)
    
# Truncated Pareto---------------------------------------------------
@randomwrap
def rtruncated_pareto(alpha, m, b, size=None):
    """
    Random bounded Pareto variates.
    """
    u = random_number(size)
    return (-(u*b**alpha - u*m**alpha - b**alpha)/(b**alpha * m**alpha))**(-1./alpha)
    
def truncated_pareto_expval(alpha, m, b):
    """
    Expected value of truncated Pareto distribution.
    """
    
    if alpha <= 1:
        return inf
    part1 = (m**alpha)/(1. - (m/b)**alpha)
    part2 = 1.*alpha/(alpha-1)
    part3 = (1./(m**(alpha-1)) - 1./(b**(alpha-1.)))
    return part1*part2*part3
    
def truncated_pareto_like(x, alpha, m, b):
    R"""
    Truncated Pareto log-likelihood. The Pareto is a continuous, positive 
    probability distribution with two parameters. It is often used
    to characterize wealth distribution, or other examples of the
    80/20 rule.

    .. math::
        f(x \mid \alpha, m, b) = \frac{\alpha m^{\alpha} x^{-\alpha}}{1-(m/b)**{\alpha}}

    :Parameters:
      - `x` : Input data (x > m)
      - `alpha` : Shape parameter (alpha>0)
      - `m` : Scale parameter (m>0)
      - `b` : Upper bound (b>m)

    """
    return flib.truncated_pareto(x, alpha, m, b)

# Poisson--------------------------------------------------
@randomwrap
def rpoisson(mu, size=None):
    """
    Random poisson variates.
    """

    return np.random.poisson(mu,size)


def poisson_expval(mu):
    """
    Expected value of Poisson distribution.
    """

    return mu


def poisson_like(x,mu):
    R"""
    Poisson log-likelihood. 
    
    The Poisson is a discrete probability
    distribution.  It is often used to model the number of events
    occurring in a fixed period of time when the times at which events
    occur are independent. The Poisson distribution can be derived as
    a limiting case of the binomial distribution.

    .. math::
        f(x \mid \mu) = \frac{e^{-\mu}\mu^x}{x!}

    :Parameters:
      - `x` : [int] :math:`x \in {0,1,2,...}`
      - `mu` : Expected number of occurrences during the given interval, :math:`\mu \geq 0`.

    .. note::
       - :math:`E(x)=\mu`
       - :math:`Var(x)=\mu`
    
    """
    return flib.poisson(x,mu)

poisson_grad_like = {'mu' : flib.poisson_gmu}

# Truncated Poisson--------------------------------------------------
@randomwrap
def rtruncated_poisson(mu, k, size=None):
    """
    Random truncated Poisson variates with minimum value k, generated
    using rejection sampling.
    """

    # Calculate m
    try:
        k=k-1
        m = max(0, np.floor(k+1-mu))
    except (TypeError, ValueError):
        # More than one mu
        k=np.array(k)-1
        return np.array([rtruncated_poisson(x, i, size)
                         for x,i in zip(mu, np.resize(k, np.size(mu)))]).T

    # Calculate constant for acceptance probability
    C = np.exp(flib.factln(k+1)-flib.factln(k+1-m))

    # Empty array to hold random variates
    rvs = np.empty(0, int)

    total_size = np.prod(size or 1)

    while(len(rvs)<total_size):

        # Propose values by sampling from untruncated Poisson with mean mu + m
        proposals = np.random.poisson(mu+m, (total_size*4, np.size(m))).squeeze()

        # Acceptance probability
        a = C * np.array([np.exp(flib.factln(y-m)-flib.factln(y))
                          for y in proposals])
        a *= proposals > k

        # Uniform random variates
        u = np.random.random(total_size*4)

        rvs = np.append(rvs, proposals[u<a])

    return np.reshape(rvs[:total_size], size)


def truncated_poisson_expval(mu, k):
    """
    Expected value of Poisson distribution truncated to be no smaller than k.
    """

    return mu/(1.-poiscdf(k, mu))


def truncated_poisson_like(x,mu,k):
    R"""
    Truncated Poisson log-likelihood. 
    
    The Truncated Poisson is a
    discrete probability distribution that is arbitrarily truncated to
    be greater than some minimum value k. For example, zero-truncated
    Poisson distributions can be used to model counts that are
    constrained to be non-negative.

    .. math::
        f(x \mid \mu, k) = \frac{e^{-\mu}\mu^x}{x!(1-F(k|\mu))}

    :Parameters:
      - `x` : [int] :math:`x \in {0,1,2,...}`
      - `mu` : Expected number of occurrences during the given interval,
               :math:`\mu \geq 0`.
      - `k` : Truncation point representing the minimum allowable value.

    .. note::
       - :math:`E(x)=\frac{\mu}{1-F(k|\mu)}`
       - :math:`Var(x)=\frac{\mu}{1-F(k|\mu)}`
       
    """
    return flib.trpoisson(x,mu,k)

truncated_poisson_grad_like = {'mu' : flib.trpoisson_gmu}

# Truncated normal distribution--------------------------
@randomwrap
def rtruncated_normal(mu, tau, a=-np.inf, b=np.inf, size=None):
    """
    Random truncated normal variates.
    """
    
    sigma = 1./np.sqrt(tau)
    na = pymc.utils.normcdf((a-mu)/sigma)
    nb = pymc.utils.normcdf((b-mu)/sigma)
    
    # Use the inverse CDF generation method.
    U = np.random.mtrand.uniform(size=size)	
    q = U * nb + (1-U)*na
    R = pymc.utils.invcdf(q)
    
    # Unnormalize
    return R*sigma + mu

rtruncnorm = rtruncated_normal

def truncated_normal_expval(mu, tau, a, b):
    """Expected value of the truncated normal distribution.

    .. math::
       E(X) =\mu + \frac{\sigma(\varphi_1-\varphi_2)}{T}
       
       
    where
    
    .. math::
       T & =\Phi\left(\frac{B-\mu}{\sigma}\right)-\Phi
       \left(\frac{A-\mu}{\sigma}\right)\text \\
       \varphi_1 &=
       \varphi\left(\frac{A-\mu}{\sigma}\right) \\
       \varphi_2 &=
       \varphi\left(\frac{B-\mu}{\sigma}\right) \\
       
    and :math:`\varphi = N(0,1)` and :math:`tau & 1/sigma**2`.
    
    :Parameters:
      - `mu` : Mean of the distribution.
      - `tau` : Precision of the distribution, which corresponds to 1/sigma**2 (tau > 0).
      - `a` : Left bound of the distribution.
      - `b` : Right bound of the distribution.
       
    """
    phia = np.exp(normal_like(a, mu, tau))
    phib = np.exp(normal_like(b, mu, tau))
    sigma = 1./np.sqrt(tau)
    Phia = pymc.utils.normcdf((a-mu)/sigma)
    if b == np.inf:
        Phib = 1.0
    else:
        Phib = pymc.utils.normcdf((b-mu)/sigma)
    return (mu + (phia-phib)/(Phib - Phia))[0]

truncnorm_expval = truncated_normal_expval

def truncated_normal_like(x, mu, tau, a=None, b=None):
    R"""
    Truncated normal log-likelihood.

    .. math::
        f(x \mid \mu, \tau, a, b) = \frac{\phi(\frac{x-\mu}{\sigma})} {\Phi(\frac{b-\mu}{\sigma}) - \Phi(\frac{a-\mu}{\sigma})},

    where :math:`\sigma^2=1/\tau`, `\phi` is the standard normal PDF and `\Phi` is the standard normal CDF.

    :Parameters:
      - `x` : Input data.
      - `mu` : Mean of the distribution.
      - `tau` : Precision of the distribution, which corresponds to 1/sigma**2 (tau > 0).
      - `a` : Left bound of the distribution.
      - `b` : Right bound of the distribution.
    """
    x = np.atleast_1d(x)
    if a is None: a = -np.inf
    a = np.atleast_1d(a)
    if b is None: b = np.inf
    b = np.atleast_1d(b)
    mu = np.atleast_1d(mu)
    sigma = (1./np.atleast_1d(np.sqrt(tau)))
    if (x < a).any() or (x>b).any():
        return -np.inf
    else:
        n = len(x)
        phi = normal_like(x, mu, tau)
        lPhia = pymc.utils.normcdf((a-mu)/sigma, log=True)
        lPhib = pymc.utils.normcdf((b-mu)/sigma, log=True)
        try:
            d = utils.log_difference(lPhib, lPhia)
        except ValueError:
            return -np.inf
        # d = np.log(Phib-Phia)
        if len(d) == n:
            Phi = d.sum()
        else:
            Phi = n*d
        if np.isnan(Phi) or np.isinf(Phi):
            return -np.inf
        return phi - Phi

truncnorm_like = truncated_normal_like

# Azzalini's skew-normal-----------------------------------
@randomwrap
def rskew_normal(mu,tau,alpha,size=()):
    """
    Skew-normal random variates.
    """
    size_ = size or (1,)
    len_ = np.prod(size_)
    return flib.rskewnorm(len_,mu,tau,alpha,np.random.normal(size=2*len_)).reshape(size)

def skew_normal_expval(mu,tau,alpha):
    """
    Expectation of skew-normal random variables.
    """
    delta = alpha / np.sqrt(1.+alpha**2)
    return mu + np.sqrt(2/pi/tau) * delta
    
def skew_normal_like(x,mu,tau,alpha):
    R"""
    Azzalini's skew-normal log-likelihood

    .. math::
        f(x \mid \mu, \tau, \alpha) = 2 \Phi((x-\mu)\sqrt{\tau}\alpha) \phi(x,\mu,\tau)

    where :math: \Phi is the normal CDF and :math: \phi is the normal PDF.

    :Parameters:
      - `x` : Input data.
      - `mu` : Mean of the distribution.
      - `tau` : Precision of the distribution (> 0).
      - `alpha` : Shape parameter of the distribution.

    .. note::
      See http://azzalini.stat.unipd.it/SN/
    """
    return flib.sn_like(x, mu, tau, alpha)


# Student's t-----------------------------------
@randomwrap
def rt(nu, size=None):
    """
    Student's t random variates.
    """
    return rnormal(0,1,size) / np.sqrt(rchi2(nu,size)/nu)

def t_expval(nu):
    """
    Expectation of Student's t random variables.
    """
    return 0
    
def t_like(x, nu):
    R"""
    Student's T log-likelihood. 
    
    Describes a zero-mean normal variable
    whose precision is gamma distributed. Alternatively, describes the
    mean of several zero-mean normal random variables divided by their
    sample standard deviation.

    .. math::
        f(x \mid \nu) = \frac{\Gamma(\frac{\nu+1}{2})}{\Gamma(\frac{\nu}{2}) \sqrt{\nu\pi}} \left( 1 + \frac{x^2}{\nu} \right)^{-\frac{\nu+1}{2}}

    :Parameters:
      - `x` : Input data.
      - `nu` : Degrees of freedom.

    """
    nu = np.asarray(nu)
    return flib.t(x, nu)

# Non-central Student's t-----------------------------------
@randomwrap
def rnoncentral_t(mu, lam, nu, size=None):
    """
    Non-central Student's t random variates.
    """
    tau = rgamma(nu/2., nu/(2.*lam), size)
    return rnormal(mu, tau)

def noncentral_t_expval(mu, lam, nu):
    """noncentral_t_expval(mu, lam, nu)

    Expectation of non-central Student's t random variables. Only defined
    for nu>1.
    """
    if nu>1:
        return mu
    return inf

def noncentral_t_like(x, mu, lam, nu):
    R"""
    Non-central Student's T log-likelihood. 
    
    Describes a normal variable whose precision is gamma distributed.

    .. math::
        f(x|\mu,\lambda,\nu) = \frac{\Gamma(\frac{\nu +
        1}{2})}{\Gamma(\frac{\nu}{2})}
        \left(\frac{\lambda}{\pi\nu}\right)^{\frac{1}{2}}
        \left[1+\frac{\lambda(x-\mu)^2}{\nu}\right]^{-\frac{\nu+1}{2}}

    :Parameters:
      - `x` : Input data.
      - `mu` : Location parameter.
      - `lambda` : Scale parameter. 
      - `nu` : Degrees of freedom.

    """
    mu = np.asarray(mu)
    lam = np.asarray(lam)
    nu = np.asarray(nu)
    return flib.nct(x, mu, lam, nu)

def t_grad_setup(x, nu, f):
    nu = np.asarray(nu)

    return f(x, nu)

t_grad_like = {'value'  : lambda x, nu : t_grad_setup(x, nu, flib.t_grad_x),
               'nu' : lambda x, nu : t_grad_setup(x, nu, flib.t_grad_nu)}

# Half-non-central t-----------------------------------------------
@randomwrap
def rhalf_noncentral_t(mu, lam, nu, size=None):
    """
    Half-non-central Student's t random variates.
    """
    return abs(rnoncentral_t(mu, lam, nu, size=size))

def noncentral_t_expval(mu, lam, nu):
    """
    Expectation of non-central Student's t random variables. Only defined
    for nu>1.
    """
    if nu>1:
        return mu
    return inf
    
def noncentral_t_like(x, mu, lam, nu):
    R"""
    Non-central Student's T log-likelihood. Describes a normal variable
    whose precision is gamma distributed.

    .. math::
        f(x|\mu,\lambda,\nu) = \frac{\Gamma(\frac{\nu +
        1}{2})}{\Gamma(\frac{\nu}{2})}
        \left(\frac{\lambda}{\pi\nu}\right)^{\frac{1}{2}}
        \left[1+\frac{\lambda(x-\mu)^2}{\nu}\right]^{-\frac{\nu+1}{2}}

    :Parameters:
      - `x` : Input data.
      - `mu` : Location parameter.
      - `lambda` : Scale parameter. 
      - `nu` : Degrees of freedom.

    """
    mu = np.asarray(mu)
    lam = np.asarray(lam)
    nu = np.asarray(nu)
    return flib.nct(x, mu, lam, nu)

# DiscreteUniform--------------------------------------------------
@randomwrap
def rdiscrete_uniform(lower, upper, size=None):
    """
    Random discrete_uniform variates.
    """
    return np.random.randint(lower, upper+1, size)

def discrete_uniform_expval(lower, upper):
    """
    Expected value of discrete_uniform distribution.
    """
    return (upper - lower) / 2.

def discrete_uniform_like(x,lower, upper):
    R"""
    Discrete uniform log-likelihood.

    .. math::
        f(x \mid lower, upper) = \frac{1}{upper-lower}

    :Parameters:
      - `x` : [int] :math:`lower \leq x \leq upper`
      - `lower` : Lower limit.
      - `upper` : Upper limit (upper > lower).

    """
    return flib.duniform_like(x, lower, upper)


# Uniform--------------------------------------------------
@randomwrap
def runiform(lower, upper, size=None):
    """
    Random uniform variates.
    """
    return np.random.uniform(lower, upper, size)

def uniform_expval(lower, upper):
    """
    Expected value of uniform distribution.
    """
    return (upper - lower) / 2.

def uniform_like(x,lower, upper):
    R"""
    Uniform log-likelihood.

    .. math::
        f(x \mid lower, upper) = \frac{1}{upper-lower}

    :Parameters:
      - `x` : :math:`lower \leq x \leq upper`
      - `lower` : Lower limit.
      - `upper` : Upper limit (upper > lower).

    """

    return flib.uniform_like(x, lower, upper)

uniform_grad_like = {'value' : flib.uniform_grad_x,
             'lower' : flib.uniform_grad_l,
             'upper' : flib.uniform_grad_u}

# Weibull--------------------------------------------------
@randomwrap
def rweibull(alpha, beta,size=None):
    """
    Weibull random variates.
    """
    tmp = -np.log(runiform(0, 1, size))
    return beta * (tmp ** (1. / alpha))

def weibull_expval(alpha,beta):
    """
    Expected value of weibull distribution.
    """
    return beta * gammaln((alpha + 1.) / alpha)

def weibull_like(x, alpha, beta):
    R"""
    Weibull log-likelihood

    .. math::
        f(x \mid \alpha, \beta) = \frac{\alpha x^{\alpha - 1}
        \exp(-(\frac{x}{\beta})^{\alpha})}{\beta^\alpha}

    :Parameters:
      - `x` : :math:`x \ge 0`
      - `alpha` : alpha > 0
      - `beta` : beta > 0

    .. note::
      - :math:`E(x)=\beta \Gamma(1+\frac{1}{\alpha})`
      - :math:`Var(x)=\beta^2 \Gamma(1+\frac{2}{\alpha} - \mu^2)`
      
    """
    return flib.weibull(x, alpha, beta)

weibull_grad_like = {'value' : flib.weibull_gx,
                 'alpha' : flib.weibull_ga,
                 'beta' : flib.weibull_gb}

# Wishart---------------------------------------------------

def rwishart(n, Tau):
    """
    Return a Wishart random matrix.

    Tau is the inverse of the 'covariance' matrix :math:`C`.
    """
    p = np.shape(Tau)[0]
    sig = np.linalg.cholesky(Tau)
    if n<p:
        raise ValueError('Wishart parameter n must be greater '
                         'than size of matrix.')
    norms = np.random.normal(size=p*(p-1)/2)
    chi_sqs = np.sqrt(np.random.chisquare(df=np.arange(n,n-p,-1)))
    A = flib.expand_triangular(chi_sqs, norms)

    flib.dtrsm_wrap(sig, A, side='L', uplo='L', transa='T', alpha=1.)
    w = np.asmatrix(np.dot(A,A.T))
    flib.symmetrize(w)
    return w

def wishart_expval(n, Tau):
    """
    Expected value of wishart distribution.
    """
    return n * np.asarray(Tau.I)

def wishart_like(X, n, Tau):
    R"""
    Wishart log-likelihood. 
    
    The Wishart distribution is the probability
    distribution of the maximum-likelihood estimator (MLE) of the precision
    matrix of a multivariate normal distribution. If Tau=1, the distribution
    is identical to the chi-square distribution with n degrees of freedom.

    For an alternative parameterization based on :math:`C=T{-1}`, see
    `wishart_cov_like`.

    .. math::
        f(X \mid n, T) = {\mid T \mid}^{n/2}{\mid X \mid}^{(n-k-1)/2} \exp\left\{ -\frac{1}{2} Tr(TX) \right\}

    where :math:`k` is the rank of X.

    :Parameters:
      X : matrix
        Symmetric, positive definite.
      n : int
        Degrees of freedom, > 0.
      Tau : matrix
        Symmetric and positive definite

    .. note::
      Step method MatrixMetropolis will preserve the symmetry of Wishart variables.

    """
    return flib.blas_wishart(X,n,Tau)

# Wishart, parametrized by covariance ------------------------------------
def rwishart_cov(n, C):
    """
    Return a Wishart random matrix.

    :Parameters:
      n : int
        Degrees of freedom, > 0.
      C : matrix
        Symmetric and positive definite
    """
    # return rwishart(n, np.linalg.inv(C))

    p = np.shape(C)[0]
    # Need cholesky decomposition of precision matrix C^-1?
    sig = np.linalg.cholesky(C)

    if n<p:
        raise ValueError('Wishart parameter n must be greater '
                         'than size of matrix.')

    norms = np.random.normal(size=p*(p-1)/2)
    chi_sqs = np.sqrt(np.random.chisquare(df=np.arange(n,n-p,-1)))
    A = flib.expand_triangular(chi_sqs, norms)

    flib.dtrmm_wrap(sig, A, side='L', uplo='L', transa='N', alpha=1.)
    w = np.asmatrix(np.dot(A,A.T))
    flib.symmetrize(w)
    return w

def wishart_cov_expval(n, C):
    """
    Expected value of wishart distribution.
    
    :Parameters:
      n : int
        Degrees of freedom, > 0.
      C : matrix
        Symmetric and positive definite
    """
    return n * np.asarray(C)

def wishart_cov_like(X, n, C):
    R"""
    wishart_like(X, n, C)

    Wishart log-likelihood. The Wishart distribution is the probability
    distribution of the maximum-likelihood estimator (MLE) of the covariance
    matrix of a multivariate normal distribution. If C=1, the distribution
    is identical to the chi-square distribution with n degrees of freedom.

    For an alternative parameterization based on :math:`T=C^{-1}`, see
    `wishart_like`.

    .. math::
        f(X \mid n, C) = {\mid C^{-1} \mid}^{n/2}{\mid X \mid}^{(n-k-1)/2} \exp\left\{ -\frac{1}{2} Tr(C^{-1}X) \right\}

    where :math:`k` is the rank of X.

    :Parameters:
      X : matrix
        Symmetric, positive definite.
      n : int
        Degrees of freedom, > 0.
      C : matrix
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
        raise KeyError("No likelihood found with this name ", name+"_like")

    try:
        random = module['r'+name]
    except:
        random = None

    try:
        grad_logp = module[name + "_grad_like"]
    except:
        grad_logp = {}

    return logp, random, grad_logp

def valuewrapper(f, arguments = None):
    """Return a likelihood accepting value instead of x as a keyword argument.
    This is specifically intended for the instantiator above.
    """
    def wrapper(**kwds):
        value = kwds.pop('value')
        return f(value, **kwds)
    
    if arguments is None: 
        wrapper.__dict__.update(f.__dict__)
    else :
        wrapper.__dict__.update(arguments)
        
    return wrapper

"""
Decorate the likelihoods
"""

snapshot = locals().copy()
likelihoods = {}
for name, obj in six.iteritems(snapshot):
    if name[-5:] == '_like' and name[:-5] in availabledistributions:
        likelihoods[name[:-5]] = snapshot[name]

def local_decorated_likelihoods(obj):
    """
    New interface likelihoods
    """

    for name, like in six.iteritems(likelihoods):
        obj[name+'_like'] = gofwrapper(like, snapshot)


# local_decorated_likelihoods(locals())
# Decorating the likelihoods breaks the creation of distribution
# instantiators -DH.

# Create Stochastic instantiators

def _inject_dist(distname, kwargs={}, ns=locals()):
    """
    Reusable function to inject Stochastic subclasses into module
    namespace
    """

    dist_logp, dist_random, grad_logp = name_to_funcs(distname, ns)
    classname = capitalize(distname)
    ns[classname]= stochastic_from_dist(distname, dist_logp,
                                        dist_random,
										grad_logp, **kwargs)


for dist in sc_continuous_distributions:
    _inject_dist(dist)

for dist in mv_continuous_distributions:
    _inject_dist(dist, kwargs={'mv' : True})

for dist in sc_bool_distributions:
    _inject_dist(dist, kwargs={'dtype' : np.bool})
    
for dist in sc_discrete_distributions:
    _inject_dist(dist, kwargs={'dtype' : np.int})

for dist in mv_discrete_distributions:
    _inject_dist(dist, kwargs={'dtype' : np.int, 'mv' : True})
    



def uninformative_like(x):
    """
    Uninformative log-likelihood. Returns 0 regardless of the value of x.
    """
    return 0.


def one_over_x_like(x):
    """
    returns -np.Inf if x<0, -np.log(x) otherwise.
    """
    if np.any(x<0):
        return -np.Inf
    else:
        return -np.sum(np.log(x))


Uninformative = stochastic_from_dist('uninformative', logp=uninformative_like)
DiscreteUninformative = stochastic_from_dist('uninformative', logp=uninformative_like, dtype=np.int)
DiscreteUninformative.__name__ = 'DiscreteUninformative'
OneOverX = stochastic_from_dist('one_over_x', logp = one_over_x_like)

# Conjugates of Dirichlet get special treatment, can be parametrized
# by first k-1 'p' values

def extend_dirichlet(p):
    """
    extend_dirichlet(p)

    Concatenates 1-sum(p) to the end of p and returns.
    """
    if len(np.shape(p))>1:
        return np.hstack((p, np.atleast_2d(1.-np.sum(p))))
    else:
        return np.hstack((p,1.-np.sum(p)))


def mod_categorical_like(x,p):
    """
    Categorical log-likelihood with parent p of length k-1.

    An implicit k'th category  is assumed to exist with associated
    probability 1-sum(p).

    ..math::
        f(x=i \mid p, m, s) = p_i,
    ..math::
        i \in 0\ldots k-1

    :Parameters:
      x : integer
        :math: `x \in 0\ldots k-1`
      p : (k-1) float
        :math: `p > 0`
        :math: `\sum p < 1`
      minval : integer
      step : integer
        :math: `s \ge 1`
    """
    return categorical_like(x,extend_dirichlet(p))


def mod_categorical_expval(p):
    """
    Expected value of categorical distribution with parent p of length k-1.

    An implicit k'th category  is assumed to exist with associated
    probability 1-sum(p).
    """
    p = extend_dirichlet(p)
    return np.sum([p*i for i, p in enumerate(p)])


def rmod_categor(p,size=None):
    """
    Categorical random variates with parent p of length k-1.

    An implicit k'th category  is assumed to exist with associated
    probability 1-sum(p).
    """
    return rcategorical(extend_dirichlet(p), size)

class Categorical(Stochastic):
    __doc__ = """
C = Categorical(name, p, value=None, dtype=np.int, observed=False,
size=1, trace=True, rseed=False, cache_depth=2, plot=None)

Stochastic variable with Categorical distribution.
Parent is: p

If parent p is Dirichlet and has length k-1, an implicit k'th
category is assumed to exist with associated probability 1-sum(p.value).

Otherwise parent p's value should sum to 1.

Docstring of categorical_like (case where P is not a Dirichlet):
    """\
    + categorical_like.__doc__ +\
    """
Docstring of categorical_like (case where P is a Dirichlet):
    """\
    + categorical_like.__doc__


    parent_names = ['p', 'minval', 'step']

    def __init__(self, name, p, value=None, dtype=np.int, observed=False,
                 size=None, trace=True, rseed=False, cache_depth=2, plot=None,
                 verbose=-1,**kwds):

        if value is not None:
            if np.isscalar(value):
                self.size = None
            else:
                self.size = len(value)
        else:
            self.size = size

        if isinstance(p, Dirichlet):
            Stochastic.__init__(self, logp=valuewrapper(mod_categorical_like),
                                doc='A Categorical random variable', name=name,
                                parents={'p':p}, random=bind_size(rmod_categor, self.size),
                                trace=trace, value=value, dtype=dtype,
                                rseed=rseed, observed=observed,
                                cache_depth=cache_depth, plot=plot,
                                verbose=verbose, **kwds)
        else:
            Stochastic.__init__(self, logp=valuewrapper(categorical_like),
                                doc='A Categorical random variable', name=name,
                                parents={'p':p},
                                random=bind_size(rcategorical, self.size),
                                trace=trace, value=value, dtype=dtype,
                                rseed=rseed, observed=observed,
                                cache_depth=cache_depth, plot=plot,
                                verbose=verbose, **kwds)

# class ModCategorical(Stochastic):
#     __doc__ = """
# C = ModCategorical(name, p, minval, step[, trace=True, value=None,
#    rseed=False, observed=False, cache_depth=2, plot=None, verbose=0])
#
# Stochastic variable with ModCategorical distribution.
# Parents are: p, minval, step.
#
# If parent p is Dirichlet and has length k-1, an implicit k'th
# category is assumed to exist with associated probability 1-sum(p.value).
#
# Otherwise parent p's value should sum to 1.
#
# Docstring of mod_categorical_like (case where P is not a Dirichlet):
#     """\
#     + mod_categorical_like.__doc__ +\
#     """
# Docstring of mod_categorical_like (case where P is a Dirichlet):
#     """\
#     + mod_categorical_like.__doc__
#
#
#     parent_names = ['p', 'minval', 'step']
#
#     def __init__(self, name, p, minval=0, step=1, value=None, dtype=np.float, observed=False, size=1, trace=True, rseed=False, cache_depth=2, plot=None, verbose=0, **kwds):
#
#         if value is not None:
#             if np.isscalar(value):
#                 self.size = 1
#             else:
#                 self.size = len(value)
#         else:
#             self.size = size
#
#         if isinstance(p, Dirichlet):
#             Stochastic.__init__(self, logp=valuewrapper(mod_categorical_like), doc='A ModCategorical random variable', name=name,
#                 parents={'p':p,'minval':minval,'step':step}, random=bind_size(rmod_categor, self.size), trace=trace, value=value, dtype=dtype,
#                 rseed=rseed, observed=observed, cache_depth=cache_depth, plot=plot, verbose=verbose, **kwds)
#         else:
#             Stochastic.__init__(self, logp=valuewrapper(mod_categorical_like), doc='A ModCategorical random variable', name=name,
#                 parents={'p':p,'minval':minval,'step':step}, random=bind_size(rmod_categorical, self.size), trace=trace, value=value, dtype=dtype,
#                 rseed=rseed, observed=observed, cache_depth=cache_depth, plot=plot, verbose=verbose, **kwds)

def mod_rmultinom(n,p):
    return rmultinomial(n,extend_dirichlet(p))


def mod_multinom_like(x,n,p):
    return multinomial_like(x,n,extend_dirichlet(p))

class Multinomial(Stochastic):
    """
M = Multinomial(name, n, p, trace=True, value=None,
   rseed=False, observed=False, cache_depth=2, plot=None])

A multinomial random variable. Parents are p, minval, step.

If parent p is Dirichlet and has length k-1, an implicit k'th
category is assumed to exist with associated probability 1-sum(p.value).

Otherwise parent p's value should sum to 1.
    """

    parent_names = ['n', 'p']

    def __init__(self, name, n, p, trace=True, value=None, rseed=False,
                 observed=False, cache_depth=2, plot=None, verbose=-1,
                 **kwds):

        if isinstance(p, Dirichlet):
            Stochastic.__init__(self, logp=valuewrapper(mod_multinom_like),
                                doc='A Multinomial random variable', name=name,
                                parents={'n':n,'p':p}, random=mod_rmultinom,
                                trace=trace, value=value, dtype=np.int,
                                rseed=rseed, observed=observed,
                                cache_depth=cache_depth, plot=plot,
                                verbose=verbose, **kwds)
        else:
            Stochastic.__init__(self, logp=valuewrapper(multinomial_like),
                                doc='A Multinomial random variable', name=name,
                                parents={'n':n,'p':p}, random=rmultinomial,
                                trace=trace, value=value, dtype=np.int,
                                rseed=rseed, observed=observed,
                                cache_depth=cache_depth, plot=plot,
                                verbose=verbose, **kwds)

def Impute(name, dist_class, imputable, **parents):
    """
    This function accomodates missing elements for the data of simple
    Stochastic distribution subclasses. The masked_values argument is an
    object of type numpy.ma.MaskedArray, which contains the raw data and
    a boolean mask indicating missing values. The resulting list contains
    a list of stochastics of type dist_class, with the extant values as data
    stochastics and the missing values as variable stochastics.

    :Arguments:
      - name : string
        Name of the data stochastic
      - dist_class : Stochastic
        Stochastic subclass such as Poisson, Normal, etc.
      - imputable : numpy.ma.core.MaskedArray or iterable
        A masked array with missing elements (where mask=True, value
        is assumed missing), or any iterable that contains None
        elements that will be imputed.
      - parents (optional): dict
        Arbitrary keyword arguments.
    """

    dims = np.shape(imputable)
    masked_values = np.ravel(imputable)

    if not type(masked_values) == np.ma.core.MaskedArray:
        # Generate mask

        mask = [v is None or np.isnan(v) for v in masked_values]
        # Generate masked array
        masked_values = np.ma.masked_array(masked_values, mask)

    # Initialise list
    vars = []
    for i in xrange(len(masked_values)):

        # Name of element
        this_name = name + '[%i]'%i
        # Dictionary to hold parents
        these_parents = {}
        # Parse parents
        for key, parent in six.iteritems(parents):

            try:
                # If parent is a PyMCObject
                shape = np.shape(parent.value)
            except AttributeError:
                shape = np.shape(parent)

            if shape == dims:
                these_parents[key] = Lambda(key + '[%i]'%i,
                                            lambda p=np.ravel(parent),
                                            i=i: p[i])
            elif shape == np.shape(masked_values):
                these_parents[key] = Lambda(key + '[%i]'%i, lambda p=parent,
                                            i=i: p[i])
            else:
                these_parents[key] = parent

        if masked_values.mask[i]:
            # Missing values
            vars.append(dist_class(this_name, **these_parents))
        else:
            # Observed values
            vars.append(dist_class(this_name, value=masked_values[i],
                                   observed=True, **these_parents))
    return np.reshape(vars, dims)

ImputeMissing = Impute

if __name__ == "__main__":
    import doctest
    doctest.testmod()



