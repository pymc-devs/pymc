from dist_math import *

__all__ = ['transform', 'logtransform', 'simplextransform']


@quickclass(object)
def transform(name, forward, backward, jacobian_det, getshape=None):
    def apply(dist):

        @quickclass(TensorDist)
        def TransformedDistribtuion():
            try:
                testval = forward(dist.testval)
            except TypeError:
                testval = dist.testval

            def logp(x):
                return dist.logp(backward(x)) + jacobian_det(x)

            if hasattr(dist, "mode"):
                mode = forward(dist.mode)
            if hasattr(dist, "median"):
                mode = forward(dist.median)

            _v = forward(dist.makevar('_v'))
            type = _v.type
            shape = _v.shape.tag.test_value
            dtype = _v.dtype

            default_testvals = dist.default_testvals

            return locals()

        return TransformedDistribtuion.dist()

    def __str__():
        return name + " transform"

    return locals()

logtransform = transform("log", log, exp, idfn)

simplextransform = transform("simplex",
                             lambda p: p[:-1],
                             lambda p: concatenate(
                             [p, 1 - sum(p, keepdims=True)]),
                             lambda p: constant([0]))
