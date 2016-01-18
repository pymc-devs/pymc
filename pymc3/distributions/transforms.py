import theano.tensor as T

from ..model import FreeRV
from ..theanof import gradient
from .distribution import Distribution

__all__ = ['transform', 'stick_breaking', 'logodds', 'log']


class Transform(object):
    """A transformation of a random variable from one space into another.

    Attributes
    ----------
    name : str
    """
    name = ""

    def forward(self, x):
        raise NotImplementedError

    def backward(self, z):
        raise NotImplementedError

    def jacobian_det(self, x):
        raise NotImplementedError

    def apply(self, dist):
        return TransformedDistribution.dist(dist, self)

    def __str__(self):
        return self.name + " transform"


class ElemwiseTransform(Transform):
    def jacobian_det(self, x):
        grad = T.reshape(gradient(T.sum(self.backward(x)), [x]), x.shape)
        return T.log(T.abs_(grad))


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

        super(TransformedDistribution, self).__init__(
            v.shape.tag.test_value, v.dtype,
            testval, dist.defaults, *args, **kwargs)

    def logp(self, x):
        return (self.dist.logp(self.transform_used.backward(x)) +
                self.transform_used.jacobian_det(x))

transform = Transform


class Log(ElemwiseTransform):
    name = "log"

    def backward(self, x):
        return T.exp(x)

    def forward(self, x):
        return T.log(x)

log = Log()

inverse_logit = T.nnet.sigmoid


def logit(x):
    return T.log(x/(1-x))


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
        self.b = b

    def backward(self, x):
        a, b = self.a, self.b
        r = (b - a) * T.exp(x) / (1 + T.exp(x)) + a
        return r

    def forward(self, x):
        a, b = self.a, self.b
        r = T.log((x - a) / (b - x))
        return r

interval = Interval


class SumTo1(Transform):
    """Transforms K dimensional simplex space (values in [0,1] and sum to 1) to K - 1 vector of values in [0,1]
    """
    name = "sumto1"

    def backward(self, y):
        return T.concatenate([y, 1 - T.sum(y, keepdims=True)])

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

    def forward(self, x):
        # reverse cumsum
        x0 = x[:-1]
        s = T.extra_ops.cumsum(x0[::-1], 0)[::-1] + x[-1]
        z = x0/s
        Km1 = x.shape[0] - 1
        k = T.arange(Km1)[(slice(None), ) + (None, ) * (x.ndim - 1)]
        eq_share = - T.log(Km1 - k)   # logit(1./(Km1 + 1 - k))
        y = logit(z) - eq_share
        return y

    def backward(self, y):
        Km1 = y.shape[0]
        k = T.arange(Km1)[(slice(None), ) + (None, ) * (y.ndim - 1)]
        eq_share = - T.log(Km1 - k)   # logit(1./(Km1 + 1 - k))
        z = inverse_logit(y + eq_share)
        yl = T.concatenate([z, T.ones(y[:1].shape)])
        yu = T.concatenate([T.ones(y[:1].shape), 1-z])
        S = T.extra_ops.cumprod(yu, 0)
        x = S * yl
        return x

    def jacobian_det(self, y):
        Km1 = y.shape[0]
        k = T.arange(Km1)[(slice(None), ) + (None, ) * (y.ndim - 1)]
        eq_share = -T.log(Km1 - k)  # logit(1./(Km1 + 1 - k))
        yl = y + eq_share
        yu = T.concatenate([T.ones(y[:1].shape), 1-inverse_logit(yl)])
        S = T.extra_ops.cumprod(yu, 0)
        return T.sum(T.log(S[:-1]) - T.log1p(T.exp(yl)) - T.log1p(T.exp(-yl)),
                     0)

stick_breaking = StickBreaking()
