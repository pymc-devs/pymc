from .dist_math import *
from ..model import FreeRV
import theano

__all__ = ['transform', 'logtransform', 'logoddstransform', 'interval_transform', 'simplextransform']

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

class LogTransform(Transform):
    name = "log"
    def __init__(self): 
        pass

    def backward(self, x):
        return log(x)

    def forward(self, x):
        return exp(x)

    def jacobian_det(self, x):
        return x
logtransform = LogTransform()

logistic = t.nnet.sigmoid

class LogOddsTransform(Transform):
    name = "logodds"
    
    def __init__(self): 
        pass

    def backward(self, x): 
        return log(x/(1-x))

    def forward(self, x):
        return logistic(x)

    def jacobian_det(self, x):
        ex = exp(-x)
        return log(ex/(ex +1)**2)
logoddstransform = LogOddsTransform()


class IntervalTransform(Transform):
    name = "interval"

    def __init__(self, a,b):
        self.a=a
        self.b=b

    def backward(self, x):
        a, b= self.a, self.b
        r= log((x-a)/(b-x))
        return r

    def forward(self, x):
        a, b= self.a, self.b
        r =  (b-a)*exp(x)/(1+exp(x)) + a
        return r

    def jacobian_det(self, x):
        a, b= self.a, self.b
        ex = exp(-x)
        jac = log(ex*(b-a)/(ex + 1)**2)
        return jac

def interval_transform(a,b):
    return IntervalTransform(a,b)


class SumTo1Transform(Transform): 
    name = "sumto1"

    def __init__(self): 
        pass

    def backward(self, x):
        return x[:-1]

    def forward(self, y):  
        return concatenate([y, 1-sum(y, keepdims=True)])

class SimplexTransform(Transform): 
    def __init__(self):
        pass

    name = "simplex"
    def forward(self, x):
        return theano.tensor.zeros_like(t.as_tensor_variable(x[:-1]))

    def backward(self, y):
        Km1 = y.shape[0]
        k = arange(Km1)
        eq_share = logit(1./(Km1 + 1 - k))
        z = logistic(y + eq_share)
        yl = concatenate([z, [1]])
        yu = concatenate([[1], 1-z])
        #S,_ = theano.scan(fn=lambda prior_result, s_i: prior_result * s_i, sequences=[yu], outputs_info=t.ones((), dtype='float64'))
        S = t.extra_ops.cumprod(yu)
        x = S * yl
        print (S.tag.test_value)
        return x

    def jacobian_det(self, y): 
        Km1 = y.shape[0]
        k = arange(Km1)
        eq_share = logit(1./(Km1 + 1 - k))
        yl = logistic(y + eq_share)
        yu = concatenate([[1], 1-yl])
        #S,_ = theano.scan(fn=lambda prior_result, s_i: prior_result * s_i, sequences=[yu], outputs_info=t.ones((), dtype='float64'))
        S = t.extra_ops.cumprod(yu)
        return sum(log(S[:-1]) - log(1+exp(yl)) - log(1+exp(-yl)))
simplextransform = SimplexTransform()

class SimplexArrTransform(Transform): 
    def __init__(self):
        pass

    name = "simplexarr"
    def forward(self, x):
        return x[:-1]

    def backward(self, y):  
        return concatenate([y, 1-sum(y, keepdims=True)])

    def jacobian_det(self, y): 
        return 0.
simplexarrtransform = SimplexArrTransform()

