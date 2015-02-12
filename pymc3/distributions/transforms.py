from .dist_math import *
from ..model import FreeRV

__all__ = ['transform', 'logtransform', 'simplextransform']

class Transform(object):
    """A transformation of a random variable from one space into another."""
    def __init__(self, name, forward, backward, jacobian_det):
        """
        Parameters
        ----------
        name : str
        forward : function 
            forward transformation
        backwards : function 
            backwards transformation
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
        try:
            testval = forward(dist.testval)
        except TypeError:
            testval = dist.testval
        
        if hasattr(dist, "mode"):
            self.mode = forward(dist.mode)
        if hasattr(dist, "median"):
            self.mode = forward(dist.median)
        
        self.dist = dist
        self.transform = transform
        v = forward(FreeRV(name='v', distribution=dist))
        self.type = v.type

        super(TransformedDistribution, self).__init__(v.shape.tag.test_value,
                v.dtype, 
                testval, dist.defaults, 
                *args, **kwargs)


    def logp(self, x):
        return self.dist.logp(self.transform.backward(x)) + self.transform.jacobian_det(x)

transform = Transform

logtransform = transform("log", log, exp, idfn)

simplextransform = transform("simplex",
                             lambda p: p[:-1],
                             lambda p: concatenate(
                             [p, 1 - sum(p, keepdims=True)]),
                             lambda p: constant([0]))
