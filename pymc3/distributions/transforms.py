from .dist_math import *
from ..model import FreeRV
import theano

from ..theanof import gradient

__all__ = ['transform', 'logtransform', 'simplextransform', 'stick_breaking', 'logodds', 'log']

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
        return name + " transform"

class ElemwiseTransform(Transform):
    def jacobian_det(self, x):
        grad = gradient(t.sum(self.backward(x)), [x])

        j =  t.sum(t.log(t.abs_(grad)))
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
logtransform = log

logistic = t.nnet.sigmoid

def logit(x):
    return t.log(x/(1-x))

class LogOdds(ElemwiseTransform):
    name = "logodds"
    
    def __init__(self): 
        pass

    def backward(self, x): 
        return logistic(x)

    def forward(self, x):
        return logit(x)

logodds = LogOdds()


class Interval(ElemwiseTransform):
    name = "interval"

    def __init__(self, a,b):
        self.a=a
        self.b=b

    def backward(self, x):
        a, b = self.a, self.b
        r =  (b - a) * t.exp(x) / (1 + t.exp(x)) + a
        return r

    def forward(self, x):
        a, b = self.a, self.b
        r = t.log((x - a) / (b - x))
        return r

def interval(a,b):
    return Interval(a,b)

class SumTo1(Transform): 
    name = "sumto1"

    def __init__(self): 
        pass

    def backward(self, x):
        return x[:-1]

    def forward(self, y):  
        return concatenate([y, 1-sum(y, keepdims=True)])
sum_to_1 = SumTo1()

class StickBreaking(Transform): 
    def __init__(self):
        pass

    name = "stickbreaking"

    def forward(self, x):
        #reverse cumsum
        x0 = x[:-1]
        s = t.extra_ops.cumsum(x0[::-1])[::-1] + x[-1]
        z = x0/s
        Km1 = x.shape[0] - 1
        k = arange(Km1)
        eq_share = - t.log(Km1 - k) # logit(1./(Km1 + 1 - k)) 
        y =  logit(z) - eq_share
        return y

    def backward(self, y):
        Km1 = y.shape[0]
        k = arange(Km1)
        eq_share = - t.log(Km1 - k) # logit(1./(Km1 + 1 - k)) 
        z = logistic(y + eq_share)
        yl = concatenate([z, [1]])
        yu = concatenate([[1], 1-z])
        S = t.extra_ops.cumprod(yu)
        x = S * yl
        return x

    def jacobian_det(self, y): 
        Km1 = y.shape[0]
        k = arange(Km1)
        eq_share =  -t.log(Km1 - k) #logit(1./(Km1 + 1 - k)) 
        yl = y + eq_share
        yu = concatenate([[1], 1-logistic(yl)])
        S = t.extra_ops.cumprod(yu)
        return sum(t.log(S[:-1]) - t.log(1+exp(yl)) - t.log(1+exp(-yl)))

stick_breaking = StickBreaking()
simplextransform = stick_breaking
