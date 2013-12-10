from .dist_math import *
from ..model import FreeRV

__all__ = ['transform', 'logtransform', 'simplextransform']



class Transform(object):
    def __init__(self, name, forward, backward, jacobian_det, getshape=None):
        self.__dict__.update(locals())

    def apply(self, dist):
        return TransformedDistribution.dist(dist, self)

    def __str__(self):
        return name + " transform"

class TransformedDistribution(Distribution):
    def __init__(self, dist, transform, *args, **kwargs):
        Distribution.__init__(self, *args, **kwargs)
        forward = transform.forward 
        try:
            testval = forward(dist.testval)
        except TypeError:
            testval = dist.testval
        
        if hasattr(dist, "mode"):
            mode = forward(dist.mode)
        if hasattr(dist, "median"):
            mode = forward(dist.median)

        
        _v = forward(FreeRV(name='_v', distribution=dist))
        type = _v.type
        shape = _v.shape.tag.test_value
        dtype = _v.dtype

        default_testvals = dist.default_testvals
        self.__dict__.update(locals())


    def logp(self, x):
        return self.dist.logp(self.transform.backward(x)) + self.transform.jacobian_det(x)


transform = Transform

logtransform = transform("log", log, exp, idfn)

simplextransform = transform("simplex",
                             lambda p: p[:-1],
                             lambda p: concatenate(
                             [p, 1 - sum(p, keepdims=True)]),
                             lambda p: constant([0]))
