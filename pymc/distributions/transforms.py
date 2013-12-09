from .dist_math import *

__all__ = ['transform', 'logtransform', 'simplextransform']



class Transform(object):
    def __init__(self, name, forward, backward, jacobian_det, getshape=None):
        self.__dict__.update(locals())

    def apply(self, dist):
        return TransformedDistribtuion(dist, self)

    def __str__(self):
        return name + " transform"

class TransformedDistribtuion(Distribution):
    def __init__(self, dist, transform, *args, **kwargs):
        TensorDist.__init__(self, *args, **kwargs)
        try:
            testval = forward(dist.testval)
        except TypeError:
            testval = dist.testval
        
        if hasattr(dist, "mode"):
            mode = forward(dist.mode)
        if hasattr(dist, "median"):
            mode = forward(dist.median)

        _v = forward(dist.makevar('_v'))
        type = _v.type
        shape = _v.shape.tag.test_value
        dtype = _v.dtype

        default_testvals = dist.default_testvals
        self.__dict__.update(locals())


    def logp(self, x):
        return self.dist.logp(self.backward(x)) + self.jacobian_det(x)


transform = Transform

logtransform = transform("log", log, exp, idfn)

simplextransform = transform("simplex",
                             lambda p: p[:-1],
                             lambda p: concatenate(
                             [p, 1 - sum(p, keepdims=True)]),
                             lambda p: constant([0]))
