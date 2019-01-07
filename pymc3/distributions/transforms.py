import theano
import theano.tensor as tt

from ..model import FreeRV
from ..theanof import gradient, floatX
from . import distribution
from ..math import logit, invlogit
from .distribution import draw_values
import numpy as np
from scipy.special import logit as nplogit

__all__ = ['transform', 'stick_breaking', 'logodds', 'interval', 'log_exp_m1',
           'lowerbound', 'upperbound', 'ordered', 'log', 'sum_to_1',
           't_stick_breaking']


class Transform:
    """A transformation of a random variable from one space into another.

    Attributes
    ----------
    name : str
    """
    name = ""

    def forward(self, x):
        """Applies transformation forward to input variable `x`.
        When transform is used on some distribution `p`, it will transform the random variable `x` after sampling
        from `p`.

        Parameters
        ----------
        x : tensor
            Input tensor to be transformed.

        Returns
        --------
        tensor
            Transformed tensor.
        """
        raise NotImplementedError

    def forward_val(self, x, point):
        """Applies transformation forward to input array `x`.
        Similar to `forward` but for constant data.

        Parameters
        ----------
        x : array_like
            Input array to be transformed.
        point : array_like, optional
            Test value used to draw (fix) bounds-like transformations

        Returns
        --------
        array_like
            Transformed array.
        """
        raise NotImplementedError

    def backward(self, z):
        """Applies inverse of transformation to input variable `z`.
        When transform is used on some distribution `p`, which has observed values `z`, it is used to
        transform the values of `z` correctly to the support of `p`.

        Parameters
        ----------
        z : tensor
            Input tensor to be inverse transformed.

        Returns
        -------
        tensor
            Inverse transformed tensor.
        """
        raise NotImplementedError

    def jacobian_det(self, x):
        """Calculates logarithm of the absolute value of the Jacobian determinant for input `x`.

        Parameters
        ----------
        x : tensor
            Input to calculate Jacobian determinant of.

        Returns
        -------
        tensor
            The log abs Jacobian determinant of `x` w.r.t. this transform.
        """
        raise NotImplementedError

    def apply(self, dist):
        # avoid circular import
        return TransformedDistribution.dist(dist, self)

    def __str__(self):
        return self.name + " transform"


class ElemwiseTransform(Transform):

    def jacobian_det(self, x):
        grad = tt.reshape(gradient(tt.sum(self.backward(x)), [x]), x.shape)
        return tt.log(tt.abs_(grad))


class TransformedDistribution(distribution.Distribution):
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
        forward_val = transform.forward_val

        self.dist = dist
        self.transform_used = transform
        v = forward(FreeRV(name='v', distribution=dist))
        self.type = v.type

        super().__init__(v.shape.tag.test_value, v.dtype, testval, dist.defaults, *args, **kwargs)

        if transform.name == 'stickbreaking':
            b = np.hstack(((np.atleast_1d(self.shape) == 1)[:-1], False))
            # force the last dim not broadcastable
            self.type = tt.TensorType(v.dtype, b)

    def logp(self, x):
        logp_nojac = self.logp_nojac(x)
        jacobian_det = self.transform_used.jacobian_det(x)
        if logp_nojac.ndim > jacobian_det.ndim:
            logp_nojac = logp_nojac.sum(axis=-1)
        return logp_nojac + jacobian_det

    def logp_nojac(self, x):
        return self.dist.logp(self.transform_used.backward(x))

transform = Transform


class Log(ElemwiseTransform):
    name = "log"

    def backward(self, x):
        return tt.exp(x)

    def forward(self, x):
        return tt.log(x)

    def forward_val(self, x, point=None):
        return np.log(x)

    def jacobian_det(self, x):
        return x

log = Log()


class LogExpM1(ElemwiseTransform):
    name = "log_exp_m1"

    def backward(self, x):
        return tt.nnet.softplus(x)

    def forward(self, x):
        """Inverse operation of softplus
        y = Log(Exp(x) - 1)
          = Log(1 - Exp(-x)) + x
        """
        return tt.log(1.-tt.exp(-x)) + x

    def forward_val(self, x, point=None):
        return np.log(1.-np.exp(-x)) + x

    def jacobian_det(self, x):
        return -tt.nnet.softplus(-x)

log_exp_m1 = LogExpM1()


class LogOdds(ElemwiseTransform):
    name = "logodds"

    def backward(self, x):
        return invlogit(x, 0.0)

    def forward(self, x):
        return logit(x)

    def forward_val(self, x, point=None):
        return nplogit(x)

logodds = LogOdds()


class Interval(ElemwiseTransform):
    """Transform from real line interval [a,b] to whole real line."""

    name = "interval"

    def __init__(self, a, b):
        self.a = tt.as_tensor_variable(a)
        self.b = tt.as_tensor_variable(b)

    def backward(self, x):
        a, b = self.a, self.b
        r = (b - a) * tt.nnet.sigmoid(x) + a
        return r

    def forward(self, x):
        a, b = self.a, self.b
        return tt.log(x - a) - tt.log(b - x)

    def forward_val(self, x, point=None):
        # 2017-06-19
        # the `self.a-0.` below is important for the testval to propagates
        # For an explanation see pull/2328#issuecomment-309303811
        a, b = draw_values([self.a-0., self.b-0.],
                            point=point)
        return floatX(np.log(x - a) - np.log(b - x))

    def jacobian_det(self, x):
        s = tt.nnet.softplus(-x)
        return tt.log(self.b - self.a) - 2 * s - x

interval = Interval


class LowerBound(ElemwiseTransform):
    """Transform from real line interval [a,inf] to whole real line."""

    name = "lowerbound"

    def __init__(self, a):
        self.a = tt.as_tensor_variable(a)

    def backward(self, x):
        a = self.a
        r = tt.exp(x) + a
        return r

    def forward(self, x):
        a = self.a
        return tt.log(x - a)

    def forward_val(self, x, point=None):
        # 2017-06-19
        # the `self.a-0.` below is important for the testval to propagates
        # For an explanation see pull/2328#issuecomment-309303811
        a = draw_values([self.a-0.],
                        point=point)[0]
        return floatX(np.log(x - a))

    def jacobian_det(self, x):
        return x

lowerbound = LowerBound


class UpperBound(ElemwiseTransform):
    """Transform from real line interval [-inf,b] to whole real line."""

    name = "upperbound"

    def __init__(self, b):
        self.b = tt.as_tensor_variable(b)

    def backward(self, x):
        b = self.b
        r = b - tt.exp(x)
        return r

    def forward(self, x):
        b = self.b
        return tt.log(b - x)

    def forward_val(self, x, point=None):
        # 2017-06-19
        # the `self.b-0.` below is important for the testval to propagates
        # For an explanation see pull/2328#issuecomment-309303811
        b = draw_values([self.b-0.],
                        point=point)[0]
        return floatX(np.log(b - x))

    def jacobian_det(self, x):
        return x

upperbound = UpperBound


class Ordered(Transform):
    name = "ordered"

    def backward(self, y):
        x = tt.zeros(y.shape)
        x = tt.inc_subtensor(x[..., 0], y[..., 0])
        x = tt.inc_subtensor(x[..., 1:], tt.exp(y[..., 1:]))
        return tt.cumsum(x, axis=-1)

    def forward(self, x):
        y = tt.zeros(x.shape)
        y = tt.inc_subtensor(y[..., 0], x[..., 0])
        y = tt.inc_subtensor(y[..., 1:], tt.log(x[..., 1:] - x[..., :-1]))
        return y

    def forward_val(self, x, point=None):
        y = np.zeros_like(x)
        y[..., 0] = x[..., 0]
        y[..., 1:] = np.log(x[..., 1:] - x[..., :-1])
        return y

    def jacobian_det(self, y):
        return tt.sum(y[..., 1:], axis=-1)

ordered = Ordered()


class SumTo1(Transform):
    """
    Transforms K dimensional simplex space (values in [0,1] and sum to 1) to K - 1 vector of values in [0,1]
    This Transformation operates on the last dimension of the input tensor.
    """
    name = "sumto1"

    def backward(self, y):
        remaining = 1 - tt.sum(y[..., :], axis=-1, keepdims=True)
        return tt.concatenate([y[..., :], remaining], axis=-1)

    def forward(self, x):
        return x[..., :-1]

    def forward_val(self, x, point=None):
        return x[..., :-1]

    def jacobian_det(self, x):
        y = tt.zeros(x.shape)
        return tt.sum(y, axis=-1)

sum_to_1 = SumTo1()


class StickBreaking(Transform):
    """
    Transforms K dimensional simplex space (values in [0,1] and sum to 1) to K - 1 vector of real values.
    Primarily borrowed from the STAN implementation.

    Parameters
    ----------
    eps : float, positive value
        A small value for numerical stability in invlogit.
    """

    name = "stickbreaking"

    def __init__(self, eps=floatX(np.finfo(theano.config.floatX).eps)):
        self.eps = eps

    def forward(self, x_):
        x = x_.T
        # reverse cumsum
        x0 = x[:-1]
        s = tt.extra_ops.cumsum(x0[::-1], 0)[::-1] + x[-1]
        z = x0 / s
        Km1 = x.shape[0] - 1
        k = tt.arange(Km1)[(slice(None), ) + (None, ) * (x.ndim - 1)]
        eq_share = logit(1. / (Km1 + 1 - k).astype(str(x_.dtype)))
        y = logit(z) - eq_share
        return floatX(y.T)

    def forward_val(self, x_, point=None):
        x = x_.T
        # reverse cumsum
        x0 = x[:-1]
        s = np.cumsum(x0[::-1], 0)[::-1] + x[-1]
        z = x0 / s
        Km1 = x.shape[0] - 1
        k = np.arange(Km1)[(slice(None),) + (None,) * (x.ndim - 1)]
        eq_share = nplogit(1. / (Km1 + 1 - k).astype(str(x_.dtype)))
        y = nplogit(z) - eq_share
        return floatX(y.T)

    def backward(self, y_):
        y = y_.T
        Km1 = y.shape[0]
        k = tt.arange(Km1)[(slice(None), ) + (None, ) * (y.ndim - 1)]
        eq_share = logit(1. / (Km1 + 1 - k).astype(str(y_.dtype)))
        z = invlogit(y + eq_share, self.eps)
        yl = tt.concatenate([z, tt.ones(y[:1].shape)])
        yu = tt.concatenate([tt.ones(y[:1].shape), 1 - z])
        S = tt.extra_ops.cumprod(yu, 0)
        x = S * yl
        return floatX(x.T)

    def jacobian_det(self, y_):
        y = y_.T
        Km1 = y.shape[0]
        k = tt.arange(Km1)[(slice(None), ) + (None, ) * (y.ndim - 1)]
        eq_share = logit(1. / (Km1 + 1 - k).astype(str(y_.dtype)))
        yl = y + eq_share
        yu = tt.concatenate([tt.ones(y[:1].shape), 1 - invlogit(yl, self.eps)])
        S = tt.extra_ops.cumprod(yu, 0)
        return tt.sum(tt.log(S[:-1]) - tt.log1p(tt.exp(yl)) - tt.log1p(tt.exp(-yl)), 0).T

stick_breaking = StickBreaking()

t_stick_breaking = lambda eps: StickBreaking(eps)


class Circular(ElemwiseTransform):
    """Transforms a linear space into a circular one.
    """
    name = "circular"

    def backward(self, y):
        return tt.arctan2(tt.sin(y), tt.cos(y))

    def forward(self, x):
        return tt.as_tensor_variable(x)

    def forward_val(self, x, point=None):
        return x

    def jacobian_det(self, x):
        return tt.zeros(x.shape)

circular = Circular()


class CholeskyCovPacked(Transform):
    name = "cholesky-cov-packed"

    def __init__(self, n):
        self.diag_idxs = np.arange(1, n + 1).cumsum() - 1

    def backward(self, x):
        return tt.advanced_set_subtensor1(x, tt.exp(x[self.diag_idxs]), self.diag_idxs)

    def forward(self, y):
        return tt.advanced_set_subtensor1(y, tt.log(y[self.diag_idxs]), self.diag_idxs)

    def forward_val(self, y, point=None):
        y[..., self.diag_idxs] = np.log(y[..., self.diag_idxs])
        return y

    def jacobian_det(self, y):
        return tt.sum(y[self.diag_idxs])


class Chain(Transform):
    def __init__(self, transform_list):
        self.transform_list = transform_list
        self.name = '+'.join([transf.name for transf in self.transform_list])

    def forward(self, x):
        y = x
        for transf in self.transform_list:
            y = transf.forward(y)
        return y

    def forward_val(self, x, point=None):
        y = x
        for transf in self.transform_list:
            y = transf.forward_val(y)
        return y

    def backward(self, y):
        x = y
        for transf in reversed(self.transform_list):
            x = transf.backward(x)
        return x

    def jacobian_det(self, y):
        y = tt.as_tensor_variable(y)
        det_list = []
        ndim0 = y.ndim
        for transf in reversed(self.transform_list):
            det_ = transf.jacobian_det(y)
            det_list.append(det_)
            y = transf.backward(y)
            ndim0 = min(ndim0, det_.ndim)
        # match the shape of the smallest jacobian_det
        det = 0.
        for det_ in det_list:
            if det_.ndim > ndim0:
                det += det_.sum(axis=-1)
            else:
                det += det_
        return det
