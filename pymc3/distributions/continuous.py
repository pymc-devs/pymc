"""
pymc3.distributions

A collection of common probability distributions for stochastic
nodes in PyMC.

"""
from __future__ import division

from .dist_math import *

from . import transforms 

import numpy as np
import numpy.random as nr
import scipy.special as sp
import scipy.stats as st

from theano.tensor.basic import as_tensor_variable
import theano
from theano import get_scalar_constant_value

__all__ = ['Uniform', 'Flat', 'Normal', 'Beta', 'Exponential', 'Laplace',
           'T', 'StudentT', 'Cauchy', 'HalfCauchy', 'Gamma', 'Weibull','Bound',
           'Tpos', 'Lognormal', 'ChiSquared', 'HalfNormal', 'Wald', 'InverseGaussian',
           'Pareto', 'InverseGamma', 'ExGaussian', 
           'Erlang', 'Gumbel', 'GeneralizedExtremeValue', 'Frechet']

class PositiveContinuous(Continuous):
    """Base class for positive continuous distributions"""
    def __init__(self, transform=transforms.log, *args, **kwargs):
        super(PositiveContinuous, self).__init__(transform=transform, *args, **kwargs)

class UnitContinuous(Continuous):
    """Base class for continuous distributions on [0,1]"""
    def __init__(self, transform=transforms.logodds, *args, **kwargs):
        super(UnitContinuous, self).__init__(transform=transform, *args, **kwargs)



def get_named_nodes(graph):
    """
    Return the named nodes in a theano graph as a dictionary of name:node pairs.
    """
    return _get_named_nodes(graph, {})

def _get_named_nodes(graph, nodes):
    if graph.owner == None:
        if graph.name is not None:
            nodes.update({graph.name:graph})
    else:
        for i in graph.owner.inputs:
            nodes.update(_get_named_nodes(i, nodes))
    return nodes


"""
draw_values 

    fixing named variables (i.e., in the model context)
        
        a) by value
        b) by random sampling
        
        >>> with Model():
        ...    m = Normal('m', mu=0., tau=1e-3)
        ...    s = Gamma('s', alpha=1e3, beta=1e3)
        ...    n = Normal('n', mu=m, sd=s)
        ...    # m is fixed by value, s will be sampled.
        ...    y = n.random(point={'m':10}, size=1000)
        ...
    
"""
def draw_values(items, point=None):
    # Distribution parameters may be nodes which have named node-inputs
    # specified in the point. Need to find the node-inputs to replace them.
    givens = {}
    for item in items:
        if hasattr(item, 'name'):            
            named_nodes = get_named_nodes(item)
            if item.name in named_nodes:
                named_nodes.pop(item.name)
            for name, node in named_nodes.iteritems():
                if point is not None and name in point:
                    if not name in givens:
                        # How to ensure point[name] becomes a theano float?
                        givens[name] = (node, as_tensor_variable(point[name]))
    givens = [value for value in givens.itervalues()] 
    
    values = [None for _ in items] 
    for i, item in enumerate(items):
        if hasattr(item, 'name'):
            if point is not None and item.name in point:
                values[i] = point[item.name]
            elif hasattr(item, 'random'):
                values[i] = item.random(point=point, size=None)
            else:
                try:#
                    values[i] = theano.function([], item, givens=givens)()
                except TypeError:
                    try:
                        values[i] = item.tag.test_value
                    except AttributeError:
                        # So what happens if this fails? Eh?
                        values[i] = get_scalar_constant_value(item)
        else:# So this is guaranteed to be just a number?
            values[i] = item
    if len(values) == 1:
        return values[0]
    else:
        return values

        
        
def get_tau_sd(tau=None, sd=None):
    """
    Find precision and standard deviation

    .. math::
        \tau = \frac{1}{\sigma^2}

    Parameters
    ----------
    tau : array-like, optional
    sd : array-like, optional

    Results
    -------
    Returns tuple (tau, sd)

    Notes
    -----
    If neither tau nor sd is provided, returns (1., 1.)
    """
    if tau is None:
        if sd is None:
            sd = 1.
            tau = 1.
        else:
            tau = sd ** -2.
    else:
        if sd is not None:
            raise ValueError("Can't pass both tau and sd")
        else:
            sd = tau ** -.5

    # cast tau and sd to float in a way that works for both np.arrays
    # and pure python
    tau = 1. * tau
    sd = 1. * sd

    return (tau, sd)


class Uniform(Continuous):
    """
    Continuous uniform log-likelihood.

    .. math::
        f(x \mid lower, upper) = \frac{1}{upper-lower}

    Parameters
    ----------
    lower : float
        Lower limit (defaults to 0)
    upper : float
        Upper limit (defaults to 1)
    """
    def __init__(self, lower=0, upper=1, transform='interval', *args, **kwargs):
        super(Uniform, self).__init__(*args, **kwargs)        
        self.lower = lower
        self.upper = upper
        self.mean = (upper + lower) / 2.
        self.median = self.mean

        if transform is 'interval':
            self.transform = transforms.interval(lower, upper)    
    
    def pdf(self, x, point=None):
        """
        The probability density function (pdf) of the Uniform distribution.
        
        Parameters
        ----------
            x : float
                The value at which to evaluate the pdf.
            point : dict
                Parameter values to fix.                
        """
        lower, upper = draw_values([self.lower, self.upper], point=point)
        p = np.zeros_like(x)
        i = np.logical_and(x >= lower, x <= upper)
        p[i] = 1. / (upper - lower)
        return p    
    
    def cdf(self, x, point=None):
        """
        The cumulative distribution function (cdf) of the Uniform distribution.
        
        Parameters
        ----------
            x : float
                The value at which to evaluate the cdf.
            point : dict
                Parameter values to fix.                
        """
        lower, upper = draw_values([self.lower, self.upper], point=point)
        p = np.zeros_like(x)
        i = x >= upper
        p[i] = 1
        i = np.logical_and(x >= lower,  x < upper)
        p[i] = (x - lower) / (upper - lower)
        return p    
    
    def random(self, size=None, point=None, **kwargs):
        """
        Generate samples from the Uniform distribution.
        
        Parameters
        ----------
        size : integer
            The number (or array shape) of samples to generate. If not specified, returns one sample.
        point : dict
            Model parameters which are to be fixed.
        """
        lower, upper = draw_values([self.lower, self.upper], point=point)
        return nr.uniform(self.upper, self.lower, size)
    
    def logp(self, x):
        """
        Return the log-likelihood of x given the current
        parameters of the distribution. 
        
        This function is optimised by theano.        
        
        Parameters
        ----------
            x : float
            The iid x(s) to evaluate the log-likelihood. 
        """
        lower = self.lower
        upper = self.upper

        return bound(
            -log(upper - lower),
            lower <= x, x <= upper)
   


class Flat(Continuous):
    """
    Uninformative log-likelihood that returns 0 regardless of
    the passed x.
    """
    def __init__(self, *args, **kwargs):
        super(Flat, self).__init__(*args, **kwargs)
        self.median = 0   
    
    def logp(self, x):
        return zeros_like(x)




class Normal(Continuous):
    """
    Normal log-likelihood.

    .. math::
        f(x \mid \mu, \tau) = \sqrt{\frac{\tau}{2\pi}} \exp\left\{ -\frac{\tau}{2} (x-\mu)^2 \right\}

    Parameters
    ----------
    mu : float
        Mean of the distribution.
    tau : float
        Precision of the distribution, which corresponds to
        :math:`1/\sigma^2` (tau > 0).
    sd : float
        Standard deviation of the distribution. Alternative parameterization.

    .. note::
    - :math:`E(X) = \mu`
    - :math:`Var(X) = 1/\tau`

    """
    def __init__(self, mu=0.0, tau=None, sd=None, *args, **kwargs):
        super(Normal, self).__init__(*args, **kwargs)
        self.mu, self.tau = self.parametrize(mu, tau, sd)
        self.mean = self.median = self.mode = self.mu
        self.variance = 1. / self.tau

    def parametrize(self, mu=None, tau=None, sd=None):
        try:
            tau, sd = get_tau_sd(tau, sd)
        except ValueError as err:
            if check:
                raise err
        return mu, tau
    
    def pdf(self, x, point=None):
        """
        The probability density function (pdf) of the Normal distribution.
        
        Parameters
        ----------
            x : float
                The value at which to evaluate the pdf.
            point : dict
                Parameter values to fix.                
        """
        mu, tau = draw_values([self.mu, self.tau], point=point)                                     
        return np.sqrt(0.5 * tau / pi) * np.exp(-0.5 * tau * (x - mu) ** 2) 
    
    
    def cdf(self, x, point=None):
        """
        The cumulative distribution function (cdf) of the Normal distribution.
        
        Parameters
        ----------
            x : float
                The value at which to evaluate the cdf.
            point : dict
                Parameter values to fix.                
        """
        mu, tau = draw_values([self.mu, self.tau], point=point)   
        return 0.5 * (1. + sp.erf(np.sqrt(0.5 * tau) * (x - mu)))
    
    
    def random(self, size=None, point=None):
        """
        Generate samples from the Normal distribution.
        
        Parameters
        ----------
        size : integer
            The number (or array shape) of samples to generate. If not specified, returns one sample.
        point : dict
            Model parameters which are to be fixed.
        """
        mu, tau = draw_values([self.mu, self.tau], point=point)   
        return nr.normal(loc=mu, scale=1./tau, size=size)
    
    def logp(self, x):
        """
        Return the log-likelihood of x given the current
        parameters of the distribution. 
        
        This function is optimised by theano.        
        
        Parameters
        ----------
        x : float
            The iid x(s) to evaluate the log-likelihood. 
        """
        tau = self.tau
        mu = self.mu

        return bound(
            (-tau * (x - mu) ** 2 + log(tau / pi / 2.)) / 2.,
            tau > 0
        )




class HalfNormal(PositiveContinuous):
    """
    Half-normal log-likelihood, a normal distribution with mean 0 limited
    to the domain :math:`x \in [0, \infty)`.

    .. math::
        f(x \mid \tau) = \sqrt{\frac{2\tau}{\pi}}\exp\left\{ {\frac{-x^2 \tau}{2}}\right\}

    Parameters
    ----------
    tau : float
        precision (tau > 0)
    sd : float
        Alternative parameterization (sd > 0).

    """
    def __init__(self, tau=None, sd=None, *args, **kwargs):
        super(HalfNormal, self).__init__(*args, **kwargs)
        self.tau = self.parametrize(tau=tau, sd=sd)
        self.mean = sqrt(2 / (pi * self.tau))
        self.variance = (1. - 2. / pi) / self.tau        
    
    def parametrize(self, tau=None, sd=None):
        tau, sd = get_tau_sd(tau, sd)
        return tau
    
    def pdf(self, x, point=None):
        """
        The probability density function (pdf) of the Half-normal distribution.
        
        Parameters
        ----------
            x : float
                The value at which to evaluate the pdf.
            point : dict
                Parameter values to fix.                
        """
        tau = draw_values([self.tau], point=point)
        p = np.sqrt(2. * tau / np.pi) * np.exp(-0.5 * tau * (x ** 2))
        p[x < 0.] = 0.
        return  p    
    
    def cdf(self, x, point=None):
        """
        The cumulative distribution function (cdf) of the Half-normal distribution.
        
        Parameters
        ----------
            x : float
                The value at which to evaluate the cdf.
            point : dict
                Parameter values to fix.                
        """
        tau = draw_values([self.tau], point=point)
        return np.erf(sqrt(0.5 * tau) * x)    
    
    def random(self, size=None, point=None):
        """
        Generate samples from the Half-normal distribution.
        
        Parameters
        ----------
        size : integer
            The number (or array shape) of samples to generate. If not specified, returns one sample.
        point : dict
            Model parameters which are to be fixed.
        """
        tau = draw_values([self.tau], point=point)
        return np.abs(nr.normal(loc=0., scale=tau ** -0.5, size=size))
       
    def logp(self, x):
        """
        Return the log-likelihood of x given the current
        parameters of the distribution. 
        
        This function is optimised by theano.        
        
        Parameters
        ----------
            x : float
            The iid x(s) to evaluate the log-likelihood. 
        """
        tau = self.tau

        return bound(
            -0.5 * tau * (x ** 2) + 0.5 * log(tau * 2. / pi),
            tau > 0.,
            x >= 0.
        )

    
class Wald(PositiveContinuous):    
    """
    Wald random variable with support :math:`x \in (0, \infty)`.
      
    .. math::    
        f(x \mid \mu, \lambda) = \left(\frac{\lambda}{2\pi)}\right)^{1/2}x^{-3/2}
        \exp\left\{ -\frac{\lambda}{2x}\left(\frac{x-\mu}{\mu}\right)^2\right\}
       
    Parameters
    ----------
    mu : float, optional 
        Mean of the distribution (mu > 0).
    lam : float, optional
        Relative precision (lam > 0). 
    phi : float, optional
        Shape. Alternative parametrisation where phi = lam / mu (phi > 0).
    alpha : float, optional
        Shift/location (alpha >= 0).
        
    The Wald can be instantiated by specifying mu only (so lam=1),
    mu and lam, mu and phi, or lam and phi.    
    
    .. note::    
        - :math:`E(X) = \mu`
        - :math:`Var(X) = \frac{\mu^3}{\lambda}`
      
    References
    ----------
    .. [Tweedie1957]
       Tweedie, M. C. K. (1957). 
       Statistical Properties of Inverse Gaussian Distributions I. 
       The Annals of Mathematical Statistics, Vol. 28, No. 2, pp. 362-377
         
    .. [Michael1976]
        Michael, J. R., Schucany, W. R. and Hass, R. W. (1976). 
        Generating Random Variates Using Transformations with Multiple Roots. 
        The American Statistician, Vol. 30, No. 2, pp. 88-90
    """
    def __init__(self, mu=None, lam=None, phi=None, alpha=0., *args, **kwargs):
        super(Wald, self).__init__(*args, **kwargs)
        self.mu, self.lam, self.alpha = self.parametrize(mu, lam, phi, alpha)           
        self.mean = self.mu + alpha
        self.mode = self.mu * ( sqrt(1. + (1.5 * self.mu / self.lam) ** 2) - 1.5 * self.mu / self.lam ) + alpha
        self.variance = (self.mu ** 3) / self.lam           
    
    def parametrize(self, mu, lam, phi, alpha):
        print 'param', mu, lam, phi, alpha
        if mu is None:
            if lam is not None and phi is not None:
                return lam / phi, lam, alpha
        else:
            if lam is None:
                if phi is None:
                    return mu, 1., alpha
                else:
                    return mu, mu * phi, alpha
            else:
                if phi is None:
                   return mu, lam, alpha
               
        raise ValueError('Wald distribution must specify either mu only, mu and lam, mu and phi, or lam and phi.')

    def pdf(self, x, point=None):
        """
        The probability density function (pdf) of the Wald distribution.
        
        Parameters
        ----------
            x : float
                The value at which to evaluate the pdf.
            point : dict
                Parameter values to fix.                
        """
        mu, lam, alpha = draw_values([self.mu, self.lam, self.alpha], point=point)
        return support(np.sqrt(lam / (2. * np.pi)) * ((x - alpha) ** -1.5) * \
            np.exp(-0.5 * lam / (x - alpha) * ((x - alpha - mu) / mu) ** 2),
            x - alpha > 0
            )
    
    def cdf(self, x, point=None):
        """
        The cumulative distribution function (cdf) of the Wald distribution.
        
        Parameters
        ----------
            x : float
                The value at which to evaluate the cdf.
            point : dict
                Parameter values to fix.                
        """
        mu, lam = draw_values([self.mu, self.lam], point=point)      
        std_cdf = lambda x: 0.5 + 0.5 * sp.erf(x / np.sqrt(2))
        l = np.sqrt(lam / (x - alpha))
        m = (x - mu) / mu
        return std_cdf(l * (m - 1)) + np.exp(2. * lam / mu) + std_cdf(l * (m - 1))
    
    def random(self, point=None, size=None):
        """
        Generate samples from the Wald distribution.
        
        Parameters
        ----------
        size : integer
            The number (or array shape) of samples to generate. If not specified, returns one sample.
        point : dict
              Model parameters which are to be fixed.
        """
        mu, lam = draw_values([self.mu, self.lam], point=point)
        v = nr.normal(loc=0., scale=1., size=size) ** 2
        x = mu + (mu ** 2) * v / (2. * lam) - mu/(2. * lam) * \
            np.sqrt(4. * mu * lam * v + (mu * v) ** 2)
        z = nr.uniform(low=0., high=1., size=size)
        # i = z > mu / (mu + x)
        # x[i] = (mu**2) / x[i]
        i = np.floor(z - mu / (mu + x)) * 2 + 1
        x = (x ** -i) * (mu ** (i + 1))
        return x + alpha
      
    def logp(self, x):
        """
        Return the log-likelihood of x given the current
        parameters of the distribution. 
        
        This function is optimised by theano.        
        
        Parameters
        ----------
            x : float
            The iid x(s) to evaluate the log-likelihood. 
        """        
        mu = self.mu 
        lam = self.lam 
        alpha = self.alpha 
        # alpha *must* be iid. Otherwise this is wrong.        
        return bound(logpow(lam / (2. * pi), 0.5) - logpow(x - alpha, 1.5) 
                    - 0.5 * lam / (x - alpha) * ((x - alpha - mu) / (mu)) ** 2,
                 mu > 0.,
                 lam > 0.,
                 x > 0.,
                 alpha >=0.,
                 x - alpha > 0)

InverseGaussian = Wald


class Beta(UnitContinuous):
    """
    Beta log-likelihood. The conjugate prior for the parameter
    :math:`p` of the binomial distribution.

    .. math::
        f(x \mid \alpha, \beta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha) \Gamma(\beta)} 
            x^{\alpha - 1} (1 - x)^{\beta - 1}

    Parameters
    ----------
    alpha : float
        alpha > 0
    beta : float
        beta > 0

    Alternative parameterization:
    mu : float
        1 > mu > 0
    sd : float
        sd > 0
    .. math::
        alpha = mu * sd
        beta = (1 - mu) * sd

    .. note::
    - :math:`E(X)=\frac{\alpha}{\alpha+\beta}`
    - :math:`Var(X)=\frac{\alpha \beta}{(\alpha+\beta)^2(\alpha+\beta+1)}`

    """
    def __init__(self, alpha=None, beta=None, mu=None, sd=None, *args, **kwargs):
        super(Beta, self).__init__(*args, **kwargs)
        alpha, beta = self.parametrize(alpha, beta, mu, sd)
        self.alpha = alpha
        self.beta = beta
        self.mean = alpha / (alpha + beta)
        self.variance = alpha * beta / (
            (alpha + beta) ** 2 * (alpha + beta + 1))        
    
    def parametrize(self, alpha=None, beta=None, mu=None, sd=None):
        if (alpha is not None) and (beta is not None):
            pass
            #mu = alpha / (alpha + beta)
            #sd = (alpha * beta / (alpha + beta) ** 2 / (alpha + beta + 1)) ** 0.5
        elif (mu is not None) and (sd is not None):
            alpha = (1 - mu) * (mu ** 2) / (sd ** 2) - mu
            beta = (1 - mu) * ((1 - mu) * mu / (sd ** 2) - 1.)
        else:
            raise ValueError('Incompatible parameterization. Either use alpha and beta, or mu and sd to specify distribution. ')
    
        return alpha, beta

    def pdf(self, x, point=None):
        """
        Probability density function (pdf) of the Beta distribution.
        
        Parameters
        ----------
        x : float
            The value at for which the pdf is to be evaluated.
        point : dict
            Model parameters and xs to be fixed.
        """
        alpha, beta = draw_values([self.alpha, self.beta], point=point)
        return st.beta(a=alpha, b=beta).pdf(x)    
    
    def cdf(self, x, point=None):
        """
        The cumulative distribution function (cdf) of the Beta distribution.
        
        Parameters
        ----------
            x : float
                The value at which to evaluate the cdf.
            point : dict
                Parameter values to fix.                
        """
        alpha, beta = draw_values([self.alpha, self.beta], point=point)  
        i = np.logical_and(x >=0., x <= 1.)    
        c = np.zeros_like(x)
        c[i] = sp.btdtr(alpha, beta, x[i])
        return support(c, i)

    def random(self, point=None, size=None, **kwargs):
        """
        Generate samples from the Beta distribution.
         
        Parameters
        ----------
        size : integer
            The number (or array shape) of samples to generate. If not specified, returns one sample.
        point : dict
            Model parameters which are to be fixed.
        """
        alpha, beta = draw_values([self.alpha, self.beta], point=point)  
        return nr.beta(alpha, beta, size)
    
    def logp(self, x):
        """
        Return the log-likelihood of x given the current
        parameters of the distribution. 
        
        This function is optimised by theano.        
        
        Parameters
        ----------
            x : float
            The iid x(s) to evaluate the log-likelihood. 
        """        
        alpha = self.alpha
        beta = self.beta

        return bound(
            gammaln(alpha + beta) - gammaln(alpha) - gammaln(beta) +
            logpow(x, alpha - 1) + logpow(1 - x, beta - 1),
            0. <= x, x <= 1.,
            alpha > 0.,
            beta > 0.)


class Exponential(PositiveContinuous):
    """
    Exponential distribution

    Parameters
    ----------
    lam : float
        lam > 0
        rate or inverse scale
    """
    def __init__(self, lam, *args, **kwargs):
        super(Exponential, self).__init__(*args, **kwargs)
        self.lam = lam
        self.mean = 1. / lam
        self.median = self.mean * log(2)
        self.mode = 0

        self.variance = lam ** -2

    def pdf(self, x, point=None):
        """
        Probability density function (pdf) of the Exponential distribution.
        
        Parameters
        ----------
        x : float
            The value at for which the pdf is to be evaluated.
        point : dict
            Model parameters and xs to be fixed.
        """
        lam = draw_values([self.lam], point=point)       
        return support(st.expon.pdf(x, scale=1./lam), 
                       x >= 0., 
                       lam > 0.)
    
    def cdf(self, x, point=None):
        """
        The cumulative distribution function (cdf) of the Exponential distribution.
        
        Parameters
        ----------
            x : float
                The value at which to evaluate the cdf.
            point : dict
                Parameter values to fix.                
        """
        lam = draw_values([self.lam], point=point)          
        return support(1. - np.exp(-lam * x),
                       x >= 0,
                       lam > 0)
    
    def random(self, point=None, size=None):
        """
        Generate samples from the Exponential distribution.
        
        Parameters
        ----------
        size : integer
            The number (or array shape) of samples to generate. If not specified, returns one sample.
        point : dict
            Model parameters which are to be fixed.
        """
        lam = draw_values([self.lam], point=point)
        return nr.exponential(lam, size)
    
    def logp(self, x):
        """
        Return the log-likelihood of x given the current
        parameters of the distribution. 
        
        This function is optimised by theano.        
        
        Parameters
        ----------
            x : float
            The iid x(s) to evaluate the log-likelihood. 
        """        
        lam = self.lam
        return bound(log(lam) - lam * x,
                     x > 0.,
                     lam > 0)

class Laplace(Continuous):
    """
    Laplace distribution.
    
    .. math::
    
        f(x \mid \mu, b) = \frac{1}{2b}\exp\left{\frac{\lvert x-mu\rvert}{b}\right}

    Parameters
    ----------
    mu : float
        mean
    b : float
        scale
    """

    def __init__(self, mu, b, *args, **kwargs):
        super(Laplace, self).__init__(*args, **kwargs)
        self.b = b
        self.mean = self.median = self.mode = self.mu = mu

        self.variance = 2 * b ** 2
    
    def pdf(self, x, point=None):
        """
        Probability density function (pdf) of the Laplace distribution.
        
        Parameters
        ----------
        x : float
            The value at for which the pdf is to be evaluated.
        point : dict
            Model parameters to be fixed.
        """
        mu, b = draw_values([self.mu, self.b], point=point)            
        return support(0.5 / b * np.exp(-np.abs(x - mu) / b), 
                       b > 0.)    
    
    def cdf(self, x, point=None):
        """
        The cumulative distribution function (cdf) of the Laplace distribution.
        
        Parameters
        ----------
            x : float
                The value at which to evaluate the cdf.
            point : dict
                Parameter values to fix.                
        """
        mu, b = draw_values([self.mu, self.b], point=point)     
        i = np.sign(x - mu)
        c = np.ceil((i + 1.) / 2) - i * 0.5 * np.exp(-np.abs((x - mu) / b))
        return c

    def random(self, point=None, size=None):
        """
        Generate samples from the Exponential distribution.
        
        Parameters
        ----------
        size : integer
            The number (or array shape) of samples to generate. If not specified, returns one sample.
        point : dict
            Model parameters which are to be fixed.
        """
        mu, b = draw_values([self.mu, self.b], point=point)      
        return nr.laplace(mu, b, size)
    
    def logp(self, x):
        """
        Return the log-likelihood of x given the current
        parameters of the distribution. 
        
        This function is optimised by theano.        
        
        Parameters
        ----------
            x : float
            The iid x(s) to evaluate the log-likelihood. 
        """        
        mu = self.mu
        b = self.b

        return -log(2 * b) - abs(x - mu) / b


class Lognormal(PositiveContinuous):
    """
    Log-normal log-likelihood.

    Distribution of any random variable whose logarithm is normally
    distributed. A variable might be modeled as log-normal if it can
    be thought of as the multiplicative product of many small
    independent factors.

    .. math::
        f(x \mid \mu, \tau) = \sqrt{\frac{\tau}{2\pi}}x^{-1}\frac{
        \exp\left\{-\frac{\tau}{2} (\ln(x)-\mu)^2 \right\}}{x}

    Parameters
    ----------
    mu - float
        Location parameter. 
    tau - float
        Shape parameter (tau > 0).

    .. note::

       :math:`E(X)=e^{\mu+\frac{1}{2\tau}}`
       :math:`Var(X)=(e^{1/\tau}-1)e^{2\mu+\frac{1}{\tau}}`

    """
    def __init__(self, mu=0, tau=1, *args, **kwargs):
        super(Lognormal, self).__init__(*args, **kwargs)
        self.mu = mu
        self.tau = tau
        self.mean = exp(mu + 1./(2*tau))
        self.median = exp(mu)
        self.mode = exp(mu - 1./tau)

        self.variance = (exp(1./tau) - 1) * exp(2*mu + 1./tau)

    def pdf(self, x, point=None):
        """
        Probability density function (pdf) of the Lognormal distribution.
        
        Parameters
        ----------
        x : float
            The value at for which the pdf is to be evaluated.
        point : dict
            Model parameters and xs to be fixed.
        """
        mu, tau = draw_values([self.mu, self.tau], point=point)       
       
        return support(np.sqrt(0.5 * tau / np.pi) * (x ** -1) * \
                              np.exp(-0.5 * tau * (np.log(x) - mu) ** 2),
                       tau > 0., 
                       x > 0.)

    def cdf(self, x, point=None):
        """
        The cumulative distribution function (cdf) of the Lognormal distribution.
        
        Parameters
        ----------
            x : float
                The value at which to evaluate the cdf.
            point : dict
                Parameter values to fix.                
        """
        mu, tau = draw_values([self.mu, self.tau], point=point)            
        return support(0.5 + 0.5 * sp.erf(np.sqrt(0.5 * tau) * (np.log(x) - mu)),
                       tau > 0., 
                       x >= 0.)

    def random(self, point=None, size=None):
        """
        Generate samples from the Exponential distribution.
        
        Parameters
        ----------
        size : integer
            The number (or array shape) of samples to generate. If not specified, returns one sample.
        point : dict
            Model parameters which are to be fixed.
        """
        mu, tau = draw_values([self.mu, self.tau], point=point)  
        return np.exp(mu + 1. / np.sqrt(tau) * nr.normal(loc=0., scale=1., size=size))
    
    def logp(self, x):
        """
        Return the log-likelihood of x given the current
        parameters of the distribution. 
        
        This function is optimised by theano.        
        
        Parameters
        ----------
            x : float
            The iid x(s) to evaluate the log-likelihood. 
        """        
        mu = self.mu
        tau = self.tau

        return bound(
            -0.5*tau*(log(x) - mu)**2 + 0.5*log(tau/(2.*pi)) - log(x),
            tau > 0)

class T(Continuous):
    """
    Student's T log-likelihood.
    
    Describes a location-scale family for the standard (central)
    T distribution. 
    
    .. math::
        f(x|\mu,\lambda,\nu) = \frac{\Gamma(\frac{\nu +
        1}{2})}{\Gamma(\frac{\nu}{2})}
        \left(\frac{\lambda}{\pi\nu}\right)^{\frac{1}{2}}
        \left[1+\frac{\lambda(x-\mu)^2}{\nu}\right]^{-\frac{\nu+1}{2}}
        
    Parameters
    ----------
    nu : int
        Degrees of freedom (nu > 0).
    mu : float
        Location parameter.
    sd : float
        scale parameter (sd > 0)
    lam : float
        Precision parameter (lam > 0), where lam = 1 / sd^2. 
    """
    def __init__(self, nu, mu=0., lam=None, sd=None, *args, **kwargs):
        super(T, self).__init__(*args, **kwargs)
        self.nu = as_tensor_variable(nu)
        self.mu, self.lam, self.sd = self.parametrize(mu, lam, sd)
        self.mean = self.median = self.mode = self.mu

        self.variance = switch((self.nu > 2) * 1, (self.nu / (self.nu - 2.)) , inf)
    
    def parametrize(self, mu, lam, sd):
        lam, sd = get_tau_sd(tau=lam, sd=sd)
        return mu, lam, sd
    
    def pdf(self, x, point=None):
        """
        Probability density function (pdf) of the T distribution.
        
        Parameters
        ----------
        x : float
            The value at for which the pdf is to be evaluated.
        point : dict
            Parameter values to be fixed.
        """
        nu, mu, lam = draw_values([self.nu, self.mu, self.lam], point=point)     
        return support(st.t.pdf(x, nu, mu, lam ** -0.5),
                       lam > 0.,
                       nu > 0.)
    
    def cdf(self, x, point=None):
        """
        The cumulative distribution function (cdf) of the T distribution.
        
        Parameters
        ----------
            x : float
                The value at which to evaluate the cdf.
            point : dict
                Parameter values to fix.                
        """
        nu, mu, lam = draw_values([self.nu, self.mu, self.lam], point=point) 
        return support(st.t.cdf(x, nu, loc=mu, scale=sd),
                       nu > 0.)    
    
    def random(self, point=None, size=None):
        """
        Generate samples from the T distribution.
        
        Parameters
        ----------
        size : integer
            The number (or array shape) of samples to generate. If not specified, returns one sample.
        point : dict
            Model parameters which are to be fixed.
        """
        nu, mu, lam = draw_values([self.nu, self.mu, self.lam], point=point) 
        return st.t.rvs(nu, loc=mu, scale=sd, size=size)
    
    def logp(self, x):
        """
        Return the log-likelihood of x given the current
        parameters of the distribution. 
        
        This function is optimised by theano.        
        
        Parameters
        ----------
            x : float
            The iid x(s) to evaluate the log-likelihood. 
        """        
        nu = self.nu
        mu = self.mu
        lam = self.lam
        sd = self.sd
        
        return bound(
            gammaln((nu + 1.0) / 2.0) + .5 * log(lam / (nu * pi)) - \
                gammaln(nu / 2.0) - (nu + 1.0) / 2.0 * log(1.0 + lam * (x - mu) ** 2 / nu),
            lam > 0.,
            nu > 0.,
            sd > 0.)


StudentT = T


class Pareto(PositiveContinuous):
    """
    Pareto log-likelihood. The Pareto is a continuous, positive
    probability distribution with two parameters. It is often used
    to characterize wealth distribution, or other examples of the
    80/20 rule.

    .. math::
        f(x \mid \alpha, m) = \frac{\alpha m^{\alpha}}{x^{\alpha+1}}

    Parameters
    ----------
    alpha : float
        Shape parameter (alpha > 0)
    m : float
        Location parameter (m > 0)

    .. note::
       - :math:`E(x)=\frac{\alpha m}{\alpha-1} if \alpha > 1`
       - :math:`Var(x)=\frac{m^2 \alpha}{(\alpha-1)^2(\alpha-2)} if \alpha > 2`

    """
    def __init__(self, alpha, m, *args, **kwargs):
        super(Pareto, self).__init__(*args, **kwargs)
        self.alpha = as_tensor_variable(alpha)
        self.m = as_tensor_variable(m)
        
        alpha = self.alpha
        m = self.m
        
        self.mean = switch(gt(alpha, 1.), alpha * m / (alpha - 1.), inf)
        self.median = m * 2.** (1. / alpha)
        self.variance = switch(gt(alpha, 2), (alpha * m**2) / ((alpha - 2.) * (alpha - 1.) ** 2), inf)

    def pdf(self, x, point=None):
        """
        Probability density function (pdf) of the Pareto distribution.
        
        Parameters
        ----------
        x : float
            The value at for which the pdf is to be evaluated.
        point : dict
            Model parameters and xs to be fixed.
        """
        alpha, m = draw_values([self.alpha, self.m], point=point)       
        return support(alpha * (m ** alpha) / (x ** (alpha + 1)),
                alpha > 0,
                m > 0,
                x >= m)
    
    def cdf(self, x, point=None):
        """
        The cumulative distribution function (cdf) of the Pareto distribution.
        
        Parameters
        ----------
            x : float
                The value at which to evaluate the cdf.
            point : dict
                Parameter values to fix.                
        """        
        alpha, m = draw_values([self.slpha, self.m], point=point)        
        return support(1. - (m / x) ** alpha,
                       nu > 0.)
    
    def random(self, point=None, size=None):
        """
        Generate samples from the Pareto distribution.
        
        Parameters
        ----------
        size : integer
            The number (or array shape) of samples to generate. If not specified, returns one sample.
        point : dict
            Model parameters which are to be fixed.
        """
        alpha, m = draw_values([self.alpha, self.m], point=point)    
        u = nr.uniform(size=size)
        return m * (1. - u) ** (-1. / alpha)
    
    def logp(self, x):
        """
        Return the log-likelihood of x given the current
        parameters of the distribution. 
        
        This function is optimised by theano.        
        
        Parameters
        ----------
            x : float
            The iid x(s) to evaluate the log-likelihood. 
        """        
        alpha = self.alpha
        m = self.m
        return bound(
            log(alpha) + logpow(m, alpha) - logpow(x, alpha+1),
            alpha > 0,
            m > 0,
            x >= m)


class Cauchy(Continuous):
    """
    Cauchy log-likelihood. The Cauchy distribution is also known as the
    Lorentz or the Breit-Wigner distribution.

    .. math::
        f(x \mid \alpha, \beta) = \frac{1}{\pi \beta [1 + (\frac{x-\alpha}{\beta})^2]}

    Parameters
    ----------
    alpha : float
        Location parameter
    beta : float
        Scale parameter > 0
    .. note::
    Mode and median are at alpha.

    """

    def __init__(self, alpha, beta, *args, **kwargs):
        super(Cauchy, self).__init__(*args, **kwargs)
        self.median = self.mode = self.alpha = alpha
        self.beta = beta
        
    def pdf(self, x, point=None):
        """
        Probability density function (pdf) of the Cauchy distribution.
        
        Parameters
        ----------
        x : float
            The value at for which the pdf is to be evaluated.
        point : dict
            Model parameters and xs to be fixed.
        """
        alpha, beta = draw_values([self.alpha, self.beta], point=point)       
        return support(st.cauchy.pdf(x, alpha, beta),
                beta > 0.)
    
    def cdf(self, x, point=None):
        """
        The cumulative distribution function (cdf) of the Cauchy distribution.
        
        Parameters
        ----------
            x : float
                The value at which to evaluate the cdf.
            point : dict
                Parameter values to fix.                
        """
        alpha, beta = draw_values([self.alpha, self.beta], point=point)       
        return support(1. / pi * np.arctan((x - alpha) / beta) + 0.5,
                beta > 0.)
    
    def random(self, point=None, size=None):
        """
        Generate samples from the Cauchy distribution.
        
        Parameters
        ----------
        size : integer
            The number (or array shape) of samples to generate. If not specified, returns one sample.
        point : dict
            Model parameters which are to be fixed.
        """
        alpha, beta = draw_values([self.alpha, self.beta], point=point)    
        u = nr.uniform(size=size)   
        return support(alpha + beta * np.tan(np.pi*(u - 0.5)),
                beta > 0.)
    
    def logp(self, x):
        """
        Return the log-likelihood of x given the current
        parameters of the distribution. 
        
        This function is optimised by theano.        
        
        Parameters
        ----------
            x : float
            The iid x(s) to evaluate the log-likelihood. 
        """        
        alpha = self.alpha
        beta = self.beta
        return bound(
            -log(pi) - log(beta) - log(1 + ((x - alpha) / beta) ** 2),
            beta > 0)


class HalfCauchy(PositiveContinuous):
    """
    Half-Cauchy log-likelihood. Simply the absolute x of Cauchy.

    .. math::
        f(x \mid \beta) = \frac{2}{\pi \beta [1 + (\frac{x}{\beta})^2]}

    Parameters
    ----------
    beta - float
        Scale parameter (beta > 0).

    .. note::
      - x must be non-negative.
    """
    def __init__(self, beta, *args, **kwargs):
        super(HalfCauchy, self).__init__(*args, **kwargs)
        self.mode = 0
        self.median = beta
        self.beta = beta

    def pdf(self, x, point=None):
        """
        Probability density function (pdf) of the Half-Cauchy distribution.
        
        Parameters
        ----------
        x : float
            The value at for which the pdf is to be evaluated.
        point : dict
            Model parameters and xs to be fixed.
        """
        beta = draw_values([self.beta], point=point)       
        return support(2. / (pi * beta * (1. + (x / beta) ** 2)),
                beta > 0.,
                x >= 0.)
    
    def cdf(self, x, point=None):
        """
        The cumulative distribution function (cdf) of the Half-Cauchy distribution.
        
        Parameters
        ----------
            x : float
                The value at which to evaluate the cdf.
            point : dict
                Parameter values to fix.                
        """
        beta = draw_values([self.beta], point=point) 
        # This may be wrong.      
        return support(2. / pi * np.arctan(x / beta),
                beta > 0.,
                x >= 0.)
    
    def random(self, point=None, size=None):
        """
        Generate samples from the Half-Cauchy distribution.
        
        Parameters
        ----------
        size : integer
            The number (or array shape) of samples to generate. If not specified, returns one sample.
        point : dict
            Model parameters which are to be fixed.
            
        Returns
        -------
        An array of samples of the same shape  
        """
        beta = draw_values([self.beta], point=point) 
        u = nr.uniform(size=size)      
        return support(beta * np.abs(np.tan(np.pi*(u + 0.5))),
                beta > 0)

    def logp(self, value):
        """
        Return the log-likelihood of x given the current
        parameters of the Half-Cauchy distribution. 
        
        This function is optimised by theano.        
        
        Parameters
        ----------
            x : float
            The iid x(s) to evaluate the log-likelihood. 
        """        
        beta = self.beta
        return bound(
            log(2) - log(pi) - log(beta) - log(1 + (value / beta) ** 2),
            beta > 0,
            value >= 0)

class Gamma(PositiveContinuous):
    """
    Gamma log-likelihood.

    Represents the sum of alpha exponentially distributed random variables, each
    of which has mean beta.

    .. math::
        f(x \mid \alpha, \beta) = \frac{\beta^{\alpha}x^{\alpha-1}e^{-\beta x}}{\Gamma(\alpha)}

    Parameters
    ----------
    x : float
        math:`x \ge 0`
    alpha : float
        Shape parameter (alpha > 0).
    beta : float
        Rate parameter (beta > 0).

    Alternative parameterization:
    mu : float
        mu > 0
    sd : float
        sd > 0
    .. math::
        alpha =  \frac{mu^2}{sd^2}
        beta = \frac{mu}{sd^2}

    .. note::
    - :math:`E(X) = \frac{\alpha}{\beta}`
    - :math:`Var(X) = \frac{\alpha}{\beta^2}`

    """
    def __init__(self, alpha=None, beta=None, mu=None, sd=None, *args, **kwargs):
        super(Gamma, self).__init__(*args, **kwargs)
        alpha, beta = self.parametrize(alpha, beta, mu, sd)
        self.alpha = alpha
        self.beta = beta
        self.mean = alpha / beta
        self.mode = maximum((alpha - 1) / beta, 0)
        self.variance = alpha / beta ** 2

    def parametrize(self, alpha=None, beta=None, mu=None, sd=None):
        if (alpha is not None) and (beta is not None):
            pass
        elif (mu is not None) and (sd is not None):
            alpha = (mu / sd) ** 2
            beta = mu / (sd ** 2)
        else:
            raise ValueError('Incompatible parameterization. Either use alpha and beta, or mu and sd to specify distribution. ')

        return alpha, beta

    def pdf(self, x, point=None):
        """
        Probability density function (pdf) of the Gamma distribution.
        
        Parameters
        ----------
        x : float
            The value at for which the pdf is to be evaluated.
        point : dict
            Model parameters and xs to be fixed.
        """
        alpha, beta = draw_values([self.alpha, self.beta], point=point)
        if not alpha > 0 or not beta > 0:
            return np.zeros_like(x) + np.NaN
        return support(st.gamma.pdf(x, a=alpha , scale=1./beta),
                alpha > 0.,
                beta > 0.,
                x >= 0.)
    
    def cdf(self, x, point=None):
        """
        The cumulative distribution function (cdf) of the Gamma distribution.
        
        Parameters
        ----------
            x : float
                The value at which to evaluate the cdf.
            point : dict
                Parameter values to fix.                
        """
        # This may be wrong.      
        alpha, beta = draw_values([self.alpha, self.beta], point=point)       
        return support(sp.gammainc(alpha , beta * x) / sp.gamma(alpha),
                alpha > 0.,
                beta > 0.,
                x >= 0.)
    
    def random(self, point=None, size=None):
        """
        Generate samples from the Gamma distribution.
        
        Parameters
        ----------
        size : integer
            The number (or array shape) of samples to generate. If not specified, returns one sample.
        point : dict
            Model parameters which are to be fixed.
            
        Returns
        -------
        An array of samples of the same shape  
        """
        alpha, beta = draw_values([self.alpha, self.beta], point=point)         
        return support(st.gamma.rvs(alpha, scale=1. / beta, size=size),
                alpha > 0.,
                beta > 0.)

    def logp(self, x):
        """
        Return the log-likelihood of x given the current
        parameters of the Half-Cauchy distribution. 
        
        This function is optimised by theano.        
        
        Parameters
        ----------
            x : float
            The iid x(s) to evaluate the log-likelihood. 
        """        
        alpha = self.alpha
        beta = self.beta
        return bound(
            -gammaln(alpha) + logpow(
                beta, alpha) - beta * x + logpow(x, alpha - 1),

            x >= 0,
            alpha > 0,
            beta > 0)


class InverseGamma(PositiveContinuous):
    """
    Inverse gamma log-likelihood, the reciprocal of the gamma distribution.

    .. math::
        f(x \mid \alpha, \beta) = \frac{\beta^{\alpha}}{\Gamma(\alpha)} x^{-\alpha - 1} \exp\left(\frac{-\beta}{x}\right)

    Parameters
    ----------
      alpha : float
          Shape parameter (alpha > 0).
      beta : float
          Scale parameter (beta > 0).

    .. note::

       :math:`E(X)=\frac{\beta}{\alpha-1}`  for :math:`\alpha > 1`
       :math:`Var(X)=\frac{\beta^2}{(\alpha-1)^2(\alpha)}`  for :math:`\alpha > 2`

    """
    def __init__(self, alpha, beta=1, *args, **kwargs):
        super(InverseGamma, self).__init__(*args, **kwargs)
        self.alpha = as_tensor_variable(alpha)
        self.beta = as_tensor_variable(beta)
        
        alpha = self.alpha
        beta = self.beta
        
        self.mean = (alpha > 1) * beta / (alpha - 1.) or inf
        self.mode = beta / (alpha + 1.)
        self.variance = switch(gt(alpha, 2), (beta ** 2) / (alpha * (alpha - 1.)**2), inf)

    def pdf(self, x, point=None):
        """
        Probability density function (pdf) of the Inverse Gamma distribution.
        
        Parameters
        ----------
        x : float
            The value at for which the pdf is to be evaluated.
        point : dict
            Model parameters and xs to be fixed.
        """
        alpha, beta = draw_values([self.alpha, self.beta], point=point)       
        return support((beta ** alpha) / sp.gamma(alpha) * (x ** (-alpha - 1)) * \
                       np.exp(-beta / x),
                alpha > 0.,
                beta > 0.,
                x >= 0.)
    
    def cdf(self, x, point=None):
        """
        The cumulative distribution function (cdf) of the Gamma distribution.
        
        Parameters
        ----------
            x : float
                The value at which to evaluate the cdf.
            point : dict
                Parameter values to fix.                
        """
        alpha, beta = draw_values([self.alpha, self.beta], point=point)       
        return support(sp.gammainc(alpha , beta / x) / sp.gamma(alpha),
                alpha > 0.,
                beta > 0.,
                x >= 0.)
    
    def random(self, point=None, size=None):
        """
        Generate samples from the Gamma distribution.
        
        Parameters
        ----------
        size : integer
            The number (or array shape) of samples to generate. If not specified, returns one sample.
        point : dict
            Model parameters which are to be fixed.
            
        Returns
        -------
        An array of samples of the same shape  
        """
        alpha, beta = draw_values([self.alpha, self.beta], point=point)         
        return support(st.invgamma.rvs(alpha, scale=1. / beta, size=size),
                alpha > 0.,
                beta > 0.)

    def logp(self, x):
        """
        Return the log-likelihood of x given the current
        parameters of the Half-Cauchy distribution. 
        
        This function is optimised by theano.        
        
        Parameters
        ----------
            x : float
            The iid x(s) to evaluate the log-likelihood. 
        """        
        alpha = self.alpha
        beta = self.beta
        return bound(
            logpow(beta, alpha) - gammaln(alpha) - beta / x + logpow(x, -alpha-1),

            x > 0,
            alpha > 0,
            beta > 0)


class ChiSquared(Gamma):
    """
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
    def __init__(self, nu, *args, **kwargs):
        self.nu = nu
        super(ChiSquared, self).__init__(alpha=nu/2., beta=0.5, *args, **kwargs)


class Weibull(PositiveContinuous):
    """
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
      - :math:`median(x)=\Gamma(\log(2))^{1/\alpha}`
      - :math:`Var(x)=\beta^2 \Gamma(1+\frac{2}{\alpha} - \mu^2)`

    """
    def __init__(self, alpha, beta, *args, **kwargs):
        super(Weibull, self).__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.mean = beta * exp(gammaln(1 + 1./alpha))
        self.median = beta * exp(gammaln(log(2)))**(1./alpha)
        self.variance = (beta**2) * exp(gammaln(1 + 2./alpha - self.mean**2))

    def pdf(self, x, point=None):
        """
        Probability density function (pdf) of the Weibull distribution.
        
        Parameters
        ----------
        x : float
            The value at for which the pdf is to be evaluated.
        point : dict
            Model parameters and xs to be fixed.
        """
        alpha, beta = draw_values([self.alpha, self.beta], point=point)
        
        return support(alpha * (x ** (alpha - 1)) * (beta ** -alpha) * \
                       np.exp(-(x / beta) ** alpha),
                alpha > 0.,
                beta > 0.,
                x >= 0.)
    
    def cdf(self, x, point=None):
        """
        The cumulative distribution function (cdf) of the Weibull distribution.
        
        Parameters
        ----------
            x : float
                The value at which to evaluate the cdf.
            point : dict
                Parameter values to fix.                
        """
        # This may be wrong.      
        alpha, beta = draw_values([self.alpha, self.beta], point=point)       
        return support(1. - exp(-(x / beta) ** alpha),
                alpha > 0.,
                beta > 0.,
                x >= 0.)
    
    def random(self, point=None, size=None):
        """
        Generate samples from the Weibull distribution.
        
        Parameters
        ----------
        size : integer
            The number (or array shape) of samples to generate. If not specified, returns one sample.
        point : dict
            Model parameters which are to be fixed.
            
        Returns
        -------
        An array of samples of the same shape  
        """
        alpha, beta = draw_values([self.alpha, self.beta], point=point) 
        u = nr.uniform(size=size)
        return support(beta * (-log(u)) ** alpha,
                alpha > 0.,
                beta > 0.)        
        
    def logp(self, x):
        """
        Return the log-likelihood of a value given the current
        parameters of the Half-Cauchy distribution. 
        
        This function is optimised by theano.        
        
        Parameters
        ----------
            x : float
            The iid x(s) to evaluate the log-likelihood. 
        """            
        alpha = self.alpha
        beta = self.beta
        return bound(
            (log(alpha) - log(beta) + (alpha - 1)*log(x/beta)
            - (x/beta)**alpha),
            x >= 0,
            alpha > 0,
            beta > 0)


class Bounded(Continuous):
    """A bounded distribution."""
    def __init__(self, distribution, lower, upper, *args, **kwargs):
        self.dist = distribution.dist(*args, **kwargs)

        self.__dict__.update(self.dist.__dict__)
        self.__dict__.update(locals())

        if hasattr(self.dist, 'mode'):
            self.mode = self.dist.mode

    def logp(self, x):
        return bound(
            self.dist.logp(x),

            self.lower <= x, x <= self.upper)
        

class Bound(object):
    """Creates a new bounded distribution"""
    def __init__(self, distribution, lower=-inf, upper=inf):
        self.distribution = distribution
        self.lower = lower
        self.upper = upper

    def __call__(self, *args, **kwargs):
        first, args = args[0], args[1:]

        return Bounded(first, self.distribution, self.lower, self.upper, *args, **kwargs)

    def dist(*args, **kwargs):
        return Bounded.dist(self.distribution, self.lower, self.upper, *args, **kwargs)


Tpos = Bound(T, 0)


class ExGaussian(Continuous):    
    """
    Expoentially modified Gaussian random variable with 
    support :math:`x \in [-\infty, \infty]`.This results from
    the convolution of a normal distribution with an exponential
    distribution.
     
    .. math::
       f(x \mid \mu, \sigma, \tau) = \frac{1}{\nu}\;
       \exp\left\{\frac{\mu-x}{\nu}+\frac{\sigma^2}{2\nu^2}\right\}
       \Phi\left(\frac{x-\mu}{\sigma}-\frac{\sigma}{\nu}\right)
    
    where :math:`\Phi` is the cumulative distribution function of the
    standard normal distribution.
     
    Parameters
    ----------
    mu : float
        Mean of the normal distribution (-inf < mu < inf).
    sigma : float
        Standard deviation of the normal distribution (sigma > 0).     
    nu : float
        Mean of the exponential distribution (nu > 0).
         
    .. note::    
        - :math:`E(X) = \mu + \nu`
        - :math:`Var(X) = \sigma^2 + \nu^2`

    References
    ----------
    .. [Rigby2005]
        Rigby R.A. and Stasinopoulos D.M. (2005).
        "Generalized additive models for location, scale and shape"
        Applied Statististics., 54, part 3, pp 507-554.
    .. [Lacouture2008]
        Lacouture, Y. and Couseanou, D. (2008). 
        "How to use MATLAB to fit the ex-Gaussian and other probability functions to a distribution of response times".
        Tutorials in Quantitative Methods for Psychology, Vol. 4, No. 1, pp 35-45.
    """
    def __init__(self, mu, sigma, nu, *args, **kwargs):
        super(ExGaussian, self).__init__(*args, **kwargs)
        self.mu = mu     
        self.sigma = sigma
        self.nu = nu
        self.mean = mu + nu
        self.variance = (sigma ** 2) + (nu ** 2)    
    
    def pdf(self, x, point=None):
        """
        Proability density function (pdf) for the ExGaussian distribution.
        
        Parameters
        ----------
        x : float
            The value at which to evaluate the pdf.
        point : Point
            Fixed Parameter values.
        """
        mu, sigma, nu = draw_values([self.mu, self.sigma, self.nu], point=point)
        
        if nu >  0.05 * sigma:
            p = 1. / nu * np.exp((mu - x) / nu + 0.5 * (sigma / nu) ** 2) * \
                            st.norm.cdf((x - mu) / sigma - sigma / nu)
        else:
            p = 1. / (sigma * np.sqrt(2. * pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2) 
            
        return support(p,
                       sigma > 0.,
                       nu > 0.)

    def cdf(self, x, point=None):
        """
        Cumulative distribution function (cdf) for the ExGaussian disribution.
        
        Parameters
        ----------
        x : float
            The value at which to evaluate the cdf.
        point : Point
            Fixed Parameter values.
        """
        mu, sigma, nu = draw_values([self.mu, self.sigma, self.nu], point=point)
    
        u = (mu - x) / nu
        v = sigma / nu
        _cdf = lambda x, m, s: st.normal.cdf(x, loc=m, scale=s)
        return support(_cdf(u, 0., v) - np.exp(-u + (v ** 2) / 2 + \
                                              np.log(_cdf(u, v ** 2, v))),
                       beta > 0)
    
    def random(self, point=None, size=None):
        """
        Generate samples from the Gumbel random variable.
        
        Parameters
        ----------
        point : Point
            Parameters to fix.
        size : int
            The number (or array shape) of samples to generate.
        """
        mu, sigma, nu = draw_values([self.mu, self.sigma, self.nu], point=point)

        u = nr.uniform(low=0., high=1., size=size)
        n = nr.normal(mu, sigma, size=size)    
        return n - nu * log(u)
     
    def logp(self, x):        
        mu = self.mu
        sigma = self.sigma
        nu = self.nu         
        lp = switch(gt(nu,  0.05 * sigma),# This condition suggested by exGAUS.R from gamlss 
                    -log(nu) + (mu - x) / nu + 0.5 * (sigma / nu) ** 2 + \
                        logpow(std_cdf((x - mu) / sigma - sigma / nu), 1.),
                    -log(sigma * sqrt(2. * pi)) - 0.5 * ((x - mu) / sigma) ** 2) 
         
        return bound(lp,
                 sigma > 0.,
                 nu > 0.)
 

class Erlang(Gamma):
    """
    Erlang log-likelihood.

    .. math::
        f(x \mid b, c) = \frac{(x/b)^(c-1)e^{-x/b}}{b\gamma(c)}

    Parameters
    ----------
    x : float
        x >= 0
    b : float
        Scale parameter (b > 0).
    c : integer
        Shape parameter (c > 0).

    .. note::
        This is a gamma variate with :math:`\beta = 1/b` and :math:`\alpha = c` 
    - :math:`E(X) = bc`
    - :math:`Var(X) = b^2c`

    """
    def __init__(self, b, c, *args, **kwargs):
        super(Erlang, self).__init__(alpha=c, beta=1/b, *args, **kwargs)
        
       
class Gumbel(Continuous):
    """
    Gumbel distribution:
    
    .. math::
    
        f(x \mid \alpha, \beta) = frac{1}{\beta}\exp\left\{-\frac{x-\alpha}{\beta}-
            \exp\left\{-\frac{x-\alpha}{\beta}\right\}\right\}
            
    Parameters
    ----------
    alpha : float
        Location
    beta : float
        Scale (beta > 0)
        
    .. note:: 
    - :math:`E(X) = \alpha - \gamma\beta` where :math:`\gamma` is the Euler constant
    - :math:`Var(X) = (\beta^2pi^2) \div 6`
    """
    def __init__(self, alpha, beta, *args, **kwargs):
        super(Gumbel, self).__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        
        self.mean = alpha + EULER * beta
        self.mode = alpha
        self.variance = (beta * pi) ** 2
        
    def pdf(self, x, point=None):
        """
        Proability density function (pdf) for the Type I Gumbel distribution.
        
        Parameters
        ----------
        x : float
            The value at which to evaluate the pdf.
        point : Point
            Fixed Parameter values.
        """
        alpha, beta = draw_values(self.alpha, self.beta, Point=Point)
        z = (x - alpha) / beta
        exp = np.exp
        return support(1. / beta * exp(-z - exp(z)),
                       x > -inf,
                       x < -inf,
                       beta > 0)
    
    def cdf(self, x, point=None):
        """
        Cumulative distribution function (cdf) for the Gumbel disribution.
        
        Parameters
        ----------
        x : float
            The value at which to evaluate the cdf.
        point : Point
            Fixed Parameter values.
        """
        alpha, beta = draw_values(self.alpha, self.beta, Point=Point)
        z = (x - alpha) / beta
        exp = np.exp
        return support(1. - exp(-exp(z)),
                       x > -inf,
                       x < -inf,
                       beta > 0)
    
    def random(self, point=None, size=None):
        """
        Generate samples from the Gumbel random variable.
        
        Parameters
        ----------
        point : Point
            Parameters to fix.
        size : int
            The number (or array shape) of samples to generate.
        """
        alpha, beta = draw_values([self.alpha, self.beta], point=point)
        return alpha - beta * log(-log(nr.uniform(size=size)))
    
    def logp(self, x):
        alpha = self.alpha
        beta = self.beta       
        
        return bound(-log(beta) - (x - alpha) / beta + exp(-(x - alpha) / beta),
                     beta > 0.)


class Frechet(Continuous):
    """
    Frechet distribution:
    
    .. math::
    
        f(x \mid \alpha, \mu, \sigma) = frac{\alpha}{\sigma}\left(\frac{x-\mu}{\sigma}\right)^{-1-\lpha}
            \exp\left\{-left(\frac{x-\mu}{\sigma}\right)^{-\alpha}\right}
            
    Parameters
    ----------
    alpha : float
        Shape (alpha > 0)
    mu : float
        Location
    sigma : float
        Scale (sigma > 0)
        
    .. note:: 
    - :math:`\begin{cases}E(X) = \mu + \sigma\Gamma\left(1-\frac{1}{\alpha}\right)&\text{when }\alpha>1\\\infty&\text{otherwise}\end{cases}`
    - :math:`\begin{cases}Var(X) = \sigma^2\left(\Gamma(1-\frac{2}{\alpha})+\left(\Gamma(1-\frac{1}{\alpha})\right)^2\right)&\text{when }\alpha>2\\\infty&\text{otherwise}\end{cases}`
    
    """
    def __init__(self, alpha, mu=0., sigma=1., *args, **kwargs):
        super(Frechet, self).__init__(*args, **kwargs)
        self.alpha = alpha
        self.mu = mu
        self.sigma = sigma        
        
        self.mean = switch(ge(mu, 1), sigma * gamma(1. - 1. / alpha) , inf)
        self.mode = mu + sigma * (alpha / (1 + alpha)) ** (1. / alpha)
    
    def pdf(self, x, point=None):
        """
        Probability density function (pdf) for the Frechet disribution.
        
        Parameters
        ----------
        x : float
            The value at which to evaluate the pdf.
        point : Point
            Fixed Parameter values.
        """
        alpha, mu, sigma = draw_values([self.alpha, self.mu, self.sigma], point=point)
        exp = np.exp
        
        z = (x - mu) / sigma    
        return support((alpha / sigma) * z ** (-1 - alpha) * exp(-(z**(-alpha))), 
                       sigma > 0, 
                       x - mu > 0)
        
    def cdf(self, x, point=None):
        """
        Cumulative distribution function (cdf) for the Frechet disribution.
        
        Parameters
        ----------
        x : float
            The value at which to evaluate the cdf.
        point : Point
            Fixed Parameter values.
        """
        alpha, mu, sigma = draw_values([self.alpha, self.mu, self.sigma], point=point)
        exp = np.exp
          
        return support(exp(-((x - mu) / sigma) ** (-alpha)),
                       sigma > 0, 
                       x - mu > 0)
                
    def random(self, point=None, size=None):
        """
        Generate samples from the Frechet random variable.
        
        Parameters
        ----------
        point : Point
            Parameters to fix.
        size : int
            The number of samples to generate
        """
        alpha, mu, sigma = draw_values([self.alpha, self.mu, self.sigma], point=point)
        u = nr.uniform(size=size)
        return (-np.log(u)) ** (-1. / alpha) * sigma + mu
        
    def logp(self, x):
        alpha = self.alpha
        mu = self.mu
        sigma = self.sigma  
        
        z = (x - mu) / sigma
        return bound(log(alpha/sigma) - (1. + alpha) * z - (z ** - alpha),
                     sigma > 0.,
                     x > mu)
        
    
class GeneralizedExtremeValue(Continuous):
    r"""
    Generalized extreme value (GEV) distibution.
    
    .. math::    
        f(x \mid \mu, \sigma, \xi) =
        \begin{cases}
        \\
        \frac{1}{\sigma}\,\exp\left\{-\left(1+\xi\, z\right)^{1/\xi}\,\right\}
            \left(1-\xi\,z\right)^{-1 + 1/\xi} & \text{if } \xi = 0 \\
        \frac{1}{\sigma}\,\exp\left\{-z - \exp\left\{-z\right\}\right\}& \text{otherwise}
        \\
        \end{cases}
                
        \text{where } z = \frac{x-\mu}{\sigma}
        
    Parameters
    ----------
    mu : float
        Location
    sigma : float
        Scale
    xi : float 
        Shape. When xi = 0, this is equivalent to the Gumbel distribution,
        when xi > 0, the Freshet distribution, and when xi < 0,
        the reverse Weibull distribution.
    eps : float
        The region around 0 (i.e., +/- eps) for which xi is considered equivalent to 0.

    .. note::
    - :math:`E(X)=\begin{cases}\mu+\gamma\sigma & \xi = 0\\
        \mu+\sigma*(\Gamma(1-\xi)-1)/\xi & \xi \leq 1 \\
        \inf & \text{otherwise}\end{cases}`
    - :math:`Var(X)=\begin{cases}
        \frac{(\sigma\pi)^2}{6} & \xi = 0 \\
        \sigma^2\Gamma(1-\xi) - \frac{\Gamma(1-2\xi)^2}{\xi^2} & \xi \leq \frac{1}{2} \\
        \inf & \text{otherwise}
        \end{cases}
    """
    def __init__(self, mu, sigma, xi, eps=1e-9, *args, **kwargs):
        super(GeneralizedExtremeValue, self).__init__(*args, **kwargs)
        self.mu = as_tensor_variable(mu)
        self.sigma = as_tensor_variable(sigma)
        self.xi = as_tensor_variable(xi)
        self.eps = eps
        
        mu = self.mu
        sigma = self.sigma
        xi = self.xi
        
        self.mean = switch(ge(xi, -eps) & le(xi, eps), mu + EULER * sigma,
                      switch(eq(xi, 1), mu + sigma * (gamma(1. - xi) - 1.) / xi, inf))
        self.mode = switch(ge(xi, -eps) & le(xi, eps), mu, mu + sigma * ((1. + xi) ** (-xi) - 1.) / xi)
        self.median = switch(ge(xi, -eps) & le(xi, eps), mu - sigma * log(log(2.)), 
                        mu + sigma * ((log(2) ** -xi) - 1) / xi)
                      
        self.variance = switch(ge(xi, -eps) & le(xi, eps), ((sigma * pi) ** 2) / 6.,
                      switch(le(xi, 0.5), (sigma ** 2) * (gamma(1. - xi) - gamma(1. - 2. * xi) ** 2) / (xi ** 2),
                             inf))

    def pdf(self, x, point=None):
        """
        Probability density function (pdf) for the GEV disribution.
        
        Parameters
        ----------
        x : float
            The value at which to evaluate the pdf.
        point : Point
            Fixed Parameter values.
        """
        mu, sigma, xi, eps = draw_values([self.mu, self.sigma, self.xi, self.eps], point=point)        
        z = (x - mu) / sigma
        if np.abs(xi) < eps:
            xi = 0.        
        if xi == 0:
            p = np.exp(-np.exp(-z) - z) / sigma
            x_in_support = np.logical_not(np.isnan(p))
        else:
            p = np.exp(-(1. + xi * z) ** -(1. / xi)) * \
                (1. + xi * z) ** (-1. - (1. / xi)) / sigma
            x_in_support = np.logical_and(1. + z * xi > 0., np.logical_not(np.isnan(p)))
        return support(p, x_in_support)
    
    def cdf(self, x, point=None):
        """
        Cumulative distribution function (cdf) for the GEV disribution.
        
        Parameters
        ----------
        x : float
            The value at which to evaluate the cdf.
        point : Point
            Fixed Parameter values.
        """
        mu, sigma, xi, eps = draw_values([self.mu, self,sigma, self.xi, self.eps], point=point)        
        z = (x - mu) / sigma
        if xi > -eps and xi < eps:
            return np.exp(np.exp(-z))
        else:
            return np.exp((1. + xi * z) ** (-1. / xi))    
    
    def random(self, point=None, size=None):
        """
        Generate samples from the GEV random variable.
        
        Parameters
        ----------
        point : Point
            Parameters to fix.
        size : int
            The number of samples to generate
        """
        mu, sigma, xi, eps = draw_values([self.mu, self.sigma, self.xi, self.eps], point=point)        
        if xi > -eps and xi < eps:
            return mu - sigma * np.log(nr.exponential(scale=1., size=size))
        else:
            return mu + sigma * (nr.exponential(scale=1., size=size) ** (-sigma) - 1) / sigma
            
    def logp(self, x):
        mu = self.mu
        sigma = self.sigma
        xi = self.xi        
        eps = self.eps
        
        z = (x - mu) / sigma
        lp = switch(ge(xi, -eps) & le(xi, eps),
               -log(sigma) - z - exp(-z),
               -log(sigma) - (1. + xi * z) ** (1. / xi) + log(1. + xi * z) ** (-1. - 1. / xi)
               )
        return bound(lp,
                     sigma > 0.,
                     # Maybe unecessary:
                     x >= switch(ge(xi, 0.), mu - sigma / xi, -inf),
                     x <= switch(le(xi, 0.), mu - sigma / xi, inf))
