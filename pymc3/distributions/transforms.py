from .dist_math import *
from ..model import FreeRV
import theano

from ..theanof import gradient

__all__ = ['transform', 'stick_breaking', 'logodds', 'log']

class Transform(object):
    """A transformation of a random variable from one space into another."""
    def __init__(self, name, forward, backward, jacobian_det):
        """
        Parameters
        ----------
        name : str
        forward : function 
            forward transformation
        backward : function 
            backward transformation
        jacobian_det : function 
            jacobian determinant of the transformation"""
        self.__dict__.update(locals())

    def apply(self, dist):
        return TransformedDistribution.dist(dist, self)

    def __str__(self):
        return self.name + " transform"

class ElemwiseTransform(Transform):
    def jacobian_det(self, x):
        grad = t.reshape(gradient(t.sum(self.backward(x)), [x]),x.shape)

        j = t.log(t.abs_(grad))
        return j

class TransformedDistribution(Distribution):
    """A distribution that has been transformed from one space into another."""
    def __init__(self, dist, transform, *args, **kwargs):
        """
        Parameters
        ----------
        dist : Distribution 
        transform : Transform
        args, kwargs
            arguments to Distribution"""
        forward = transform.forward 
        testval = forward(dist.default())
        
        self.dist = dist
        self.transform_used = transform
        v = forward(FreeRV(name='v', distribution=dist))
        self.type = v.type

        super(TransformedDistribution, self).__init__(v.shape.tag.test_value,
                v.dtype, 
                testval, dist.defaults, 
                *args, **kwargs)

    def logp(self, x):
        return self.dist.logp(self.transform_used.backward(x)) + self.transform_used.jacobian_det(x)

transform = Transform

class Log(ElemwiseTransform):
    name = "log"
    def __init__(self): 
        pass

    def backward(self, x):
        return exp(x)

    def forward(self, x):
        return t.log(x)

log = Log()

inverse_logit = t.nnet.sigmoid

def logit(x):
    return t.log(x/(1-x))

class LogOdds(ElemwiseTransform):
    name = "logodds"
    
    def __init__(self): 
        pass

    def backward(self, x): 
        return inverse_logit(x)

    def forward(self, x):
        return logit(x)

logodds = LogOdds()


class Interval(ElemwiseTransform):
    """Transform from real line interval [a,b] to whole real line."""

    name = "interval"

    def __init__(self, a,b):
        self.a = a
        self.b=b

    def backward(self, x):
        a, b = self.a, self.b
        r =  (b - a) * t.exp(x) / (1 + t.exp(x)) + a
        return r

    def forward(self, x):
        a, b = self.a, self.b
        r = t.log((x - a) / (b - x))
        return r

def interval(a, b):
    return Interval(a, b)

class SumTo1(Transform): 
    """Transforms K dimensional simplex space (values in [0,1] and sum to 1) to K - 1 vector of values in [0,1]
    """
    name = "sumto1"

    def __init__(self): 
        pass

    def backward(self, y):
        return concatenate([y, 1-sum(y, keepdims=True)])

    def forward(self, x):  
        return x[:-1]

    def jacobian_det(self, x):
        return 0

sum_to_1 = SumTo1()

class StickBreaking(Transform): 
    """Transforms K dimensional simplex space (values in [0,1] and sum to 1) to K - 1 vector of real values.
    
    Primarily borrowed from the STAN implementation.
    """

    name = "stickbreaking"

    def __init__(self):
        pass

    def forward(self, x):
        #reverse cumsum
        x0 = x[:-1]
        s = t.extra_ops.cumsum(x0[::-1], 0)[::-1] + x[-1]
        z = x0/s
        Km1 = x.shape[0] - 1
        k = arange(Km1)[(slice(None), ) + (None, ) * (x.ndim - 1)]
        eq_share = - t.log(Km1 - k) # logit(1./(Km1 + 1 - k)) 
        y =  logit(z) - eq_share
        return y

    def backward(self, y):
        Km1 = y.shape[0]
        k = arange(Km1)[(slice(None), ) + (None, ) * (y.ndim - 1)]
        eq_share = - t.log(Km1 - k) # logit(1./(Km1 + 1 - k)) 
        z = inverse_logit(y + eq_share)
        yl = concatenate([z, ones(y[:1].shape)])
        yu = concatenate([ones(y[:1].shape), 1-z])
        S = t.extra_ops.cumprod(yu, 0)
        x = S * yl
        return x

    def jacobian_det(self, y): 
        Km1 = y.shape[0]
        k = arange(Km1)[(slice(None), ) + (None, ) * (y.ndim - 1)]
        eq_share =  -t.log(Km1 - k) #logit(1./(Km1 + 1 - k)) 
        yl = y + eq_share
        yu = concatenate([ones(y[:1].shape), 1-inverse_logit(yl)])
        S = t.extra_ops.cumprod(yu, 0)
        return t.log(S[:-1]) - t.log(1+exp(yl)) - t.log(1+exp(-yl))

stick_breaking = StickBreaking()
