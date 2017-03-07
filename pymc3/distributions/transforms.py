import sys
from numbers import Number

import theano.tensor as tt

from ..model import FreeRV
from ..theanof import gradient
from . import distribution
from ..math import logit, invlogit
import numpy as np

__all__ = ['transform', 'stick_breaking', 'logodds',
           'log', 'sum_to_1', 't_stick_breaking']


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
        grad = tt.reshape(gradient(tt.sum(self.backward(x)), [x]), x.shape)
        return tt.log(tt.abs_(grad))


def get_shape_for_transformed_rv(tensor, free_rv, transform_obj):
    """
    Get the shape of the transformed random variable to be created.

    Parameters
    ----------
    tensor : Theano variable.
        This object is being treated as a PyMC3 random variable.
    free_rv : PyMC3 FreeRV distribution variable.
        Should correspond to the transformed variable.
    transform_obj : instance of pymc3/distributions/transforms.Transform
        Should have two methods: `forward()` and `backward()`. These
        methods should take a Theano variable as input and return
        a theano variable as output. The `forward()` method should
        apply the desired transformation, and the `backward()` method
        perform the inverse of the desired transformation.

    Returns
    -------
    tuple of ints.
        Denotes the shape of the random variable to be created.
    """
    # Make sure the inputs are valid
    if not hasattr(tensor, "shape"):
        raise AttributeError("tensor must have a shape attribute.")
    if not hasattr(tensor.shape, "tag"):
        raise AttributeError("tensor.shape must have a tag attribute.")
    if not hasattr(free_rv, "dshape"):
        raise AttributeError("free_rv must have a dshape attribute.")
    if not isinstance(transform_obj, Transform):
        raise TypeError("transform_obj must be an instance of Transform.")

    if hasattr(tensor.shape.tag, "test_value"):
        transform_shape = tensor.shape.tag.test_value
    elif isinstance(transform_obj, (StickBreaking, SumTo1)):
        # Perform a final check of the function's arguments
        if len(free_rv.dshape) <= 0:
            raise ValueError("len(free_rv.dshape) must be greater than zero.")
        if not isinstance(free_rv.dshape[0], int):
            raise ValueError("free_rv.dshape[0] should be an integer.")

        transform_shape = tuple(free_rv.dshape[0] - 1)
    else:
        transform_shape = free_rv.dshape

    return transform_shape


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
        assert hasattr(testval.tag, "test_value")

        self.dist = dist
        self.transform_used = transform
        # Create the free RV for the random variable that is to be transformed
        free_rv = FreeRV(name='v', distribution=dist)
        v = forward(free_rv)
        self.type = v.type

        # super(TransformedDistribution, self).__init__(
        #     v.shape.tag.test_value, v.dtype,
        #     testval, dist.defaults, *args, **kwargs)

        # Get shape of the transformed random variable's theano representation
        # This function is used instead of v.shape.tag.test_value to avoid
        # attribute errors when theano.config.compute_test_value = 'off'
        transformed_shape = get_shape_for_transformed_rv(v,
                                                         free_rv,
                                                         transform)
        super(TransformedDistribution, self).__init__(
            transformed_shape, v.dtype,
            testval, dist.defaults, *args, **kwargs)

        if transform.name == 'stickbreaking':
            b = np.hstack(((np.atleast_1d(self.shape) == 1)[:-1], False))
            # force the last dim not broadcastable
            self.type = tt.TensorType(v.dtype, b)

    def logp(self, x):
        return (self.dist.logp(self.transform_used.backward(x)) +
                self.transform_used.jacobian_det(x))

transform = Transform


class Log(ElemwiseTransform):
    name = "log"

    def backward(self, x):
        return_expr = tt.exp(x)
        
        # Add a test_value if there is not one already. Useful when operating
        # with theano.config.compute_test_value = 'off'
        if not hasattr(return_expr.tag, "test_value"):
            # Get the numeric object to be used as the test value for x
            numeric_x = get_single_numeric_obj(x, name='x')
            # Set the test value for the theano variable to be returned
            return_expr.tag.test_value = np.exp(numeric_x)

        assert hasattr(return_expr.tag, "test_value")
        return return_expr

    def forward(self, x):
        return_expr = tt.log(x)

        # Add a test_value if there is not one already. Useful when operating
        # with theano.config.compute_test_value = 'off'
        if not hasattr(return_expr.tag, "test_value"):
            # Get the numeric object to be used as the test value for x
            numeric_x = get_single_numeric_obj(x, name='x')
            # Set the test value for the theano variable to be returned
            return_expr.tag.test_value = np.log(numeric_x)

        assert hasattr(return_expr.tag, "test_value")
        return return_expr

log = Log()


class LogOdds(ElemwiseTransform):
    name = "logodds"

    def __init__(self):
        pass

    def backward(self, x):
        return_expr = invlogit(x, 0.0)

        # Add a test_value if there is not one already. Useful when operating
        # with theano.config.compute_test_value = 'off'
        if not hasattr(return_expr.tag, "test_value"):
            # Get the numeric object to be used as the test value for x
            numeric_x = get_single_numeric_obj(x, name='x')
            # Compute the test value. Note that this is accurate only because
            # we pass 0.0 as 'eps' for the invlogit function.
            return_expr.tag.test_value = 1 / (1 + np.exp(-1 * numeric_x))

        return return_expr

    def forward(self, x):
        return_expr = logit(x)

        # Add a test_value if there is not one already. Useful when operating
        # with theano.config.compute_test_value = 'off'
        if not hasattr(return_expr.tag, "test_value"):
            # Get the numeric object to be used as the test value for x
            numeric_x = get_single_numeric_obj(x, name='x')
            # Set the test value
            return_expr.tag.test_value = np.log(numeric_x / (1. - numeric_x))
        return return_expr

logodds = LogOdds()


class Interval(ElemwiseTransform):
    """Transform from real line interval [a,b] to whole real line."""

    name = "interval"

    def __init__(self, a, b, eps=1e-6):
        self.a = a
        self.b = b
        self.eps = eps

    def backward(self, x):
        a, b = self.a, self.b
        r = (b - a) / (1 + tt.exp(-x)) + a

        # Add a test_value if there is not one already. Useful when operating
        # with theano.config.compute_test_value = 'off'
        if not hasattr(r.tag, "test_value"):
            # Get the numeric values of a, b, and x
            numeric_a = get_single_numeric_obj(a, name='a')
            numeric_b = get_single_numeric_obj(b, name='b')
            numeric_x = get_single_numeric_obj(x, name='x')
            # Set the test value for the theano variable to be returned
            r_test = ((numeric_b - numeric_a) /
                      (1 + np.exp(-numeric_x)) +
                      numeric_a)
            r.tag.test_value = r_test

        return r

    def forward(self, x):
        a, b, e = self.a, self.b, self.eps
        r = tt.log(tt.maximum((x - a) / tt.maximum(b - x, e), e))

        # Add a test_value if there is not one already. Useful when operating
        # with theano.config.compute_test_value = 'off'
        if not hasattr(r.tag, "test_value"):
            # Get the numeric values of a, b, and x
            numeric_a = get_single_numeric_obj(a, name='a')
            numeric_b = get_single_numeric_obj(b, name='b')
            numeric_x = get_single_numeric_obj(x, name='x')
            # Set the test value for the theano variable to be returned
            forward_ratio = ((numeric_x - numeric_a) /
                             np.max([numeric_b - numeric_x, e]))
            r_test = np.log(np.max([forward_ratio, e]))
            r.tag.test_value = r_test
        return r

interval = Interval


class LowerBound(ElemwiseTransform):
    """Transform from real line interval [a,inf] to whole real line."""

    name = "lowerbound"

    def __init__(self, a):
        self.a = a

    def backward(self, x):
        a = self.a
        r = tt.exp(x) + a

        # Add a test_value if there is not one already. Useful when operating
        # with theano.config.compute_test_value = 'off'
        if not hasattr(r.tag, "test_value"):
            # Get the numeric values of a and x
            numeric_a = get_single_numeric_obj(a, name='a')
            numeric_x = get_single_numeric_obj(x, name='x')
            # Set the test value for the theano variable to be returned
            r.tag.test_value = np.exp(numeric_x) + numeric_a
        return r

    def forward(self, x):
        a = self.a
        r = tt.log(x - a)

        # Add a test_value if there is not one already. Useful when operating
        # with theano.config.compute_test_value = 'off'
        if not hasattr(r.tag, "test_value"):
            # Get the numeric values of a and x
            numeric_a = get_single_numeric_obj(a, name='a')
            numeric_x = get_single_numeric_obj(x, name='x')
            # Set the test value for the theano variable to be returned
            r.tag.test_value = np.log(numeric_x - numeric_a)
        return r

lowerbound = LowerBound


class UpperBound(ElemwiseTransform):
    """Transform from real line interval [-inf,b] to whole real line."""

    name = "upperbound"

    def __init__(self, b):
        self.b = b

    def backward(self, x):
        b = self.b
        r = b - tt.exp(x)

        # Add a test_value if there is not one already. Useful when operating
        # with theano.config.compute_test_value = 'off'
        if not hasattr(r.tag, "test_value"):
            # Get the numeric values of b and x
            numeric_b = get_single_numeric_obj(b, name='b')
            numeric_x = get_single_numeric_obj(x, name='x')
            # Set the test value for the theano variable to be returned
            r.tag.test_value = numeric_b - np.exp(numeric_x)
        return r

    def forward(self, x):
        b = self.b
        r = tt.log(b - x)

        # Add a test_value if there is not one already. Useful when operating
        # with theano.config.compute_test_value = 'off'
        if not hasattr(r.tag, "test_value"):
            # Get the numeric values of b and x
            numeric_b = get_single_numeric_obj(b, name='b')
            numeric_x = get_single_numeric_obj(x, name='x')
            # Set the test value for the theano variable to be returned
            r.tag.test_value = np.log(numeric_b - numeric_x)
        return r

upperbound = UpperBound


class SumTo1(Transform):
    """Transforms K dimensional simplex space (values in [0,1] and sum to 1) to K - 1 vector of values in [0,1]
    """
    name = "sumto1"

    def backward(self, y):
        return_expr = tt.concatenate([y, 1 - tt.sum(y, keepdims=True)])

        # Add a test_value if there is not one already. Useful when operating
        # with theano.config.compute_test_value = 'off'
        if not hasattr(return_expr.tag, "test_value"):
            # Get the numeric values of y
            numeric_y = get_single_numeric_obj(y, name='y')

            # Figure out the shape of y
            y_shape = numeric_y.shape

            # Take an initial guess about what axis we should concatenate the
            # implicit y-value on
            concat_axis = 0

            # Figure out the actual concatenation axis and the sum of the given
            # y-values
            if len(y_shape) == 2:
                if y_shape[1] == 1:
                    # y_sum will have shape = (1, 1)
                    y_sum = np.sum(numeric_y, axis=0, keepdims=True)
                elif y_shape[0] == 1:
                    # y_sum will have shape = (1, 1)
                    y_sum = np.sum(numeric_y, axis=1, keepdims=True)
                    # Note that we're dealing with a row-vector so we should
                    # concatenate along the second axis.
                    concat_axis = 1
                else:
                    raise ValueError("y is not a column or row vector. Both "
                                     "elements of y.shape differ from 1.")
            elif len(y_shape) == 1:
                # y_sum will have shape = (1,)
                y_sum = np.sum(numeric_y, keepdims=True)
            else:
                raise ValueError("y is neither a 1D nor 2D object.")

            # Figure out the implicit remaining value of y
            implicit_y = 1 - y_sum
            # Set the test value
            return_expr.tag.test_value = np.concatenate((y, implicit_y),
                                                        axis=concat_axis)
        return return_expr

    def forward(self, x):
        return_expr = x[:-1]

        # Add a test_value if there is not one already. Useful when operating
        # with theano.config.compute_test_value = 'off'
        if not hasattr(return_expr.tag, "test_value"):
            numeric_x = get_single_numeric_obj(x, name='x')
            return_expr.tag.test_value = numeric_x[:-1]
        return return_expr

    def jacobian_det(self, x):
        return 0

sum_to_1 = SumTo1()


class StickBreaking(Transform):
    """Transforms K dimensional simplex space (values in [0,1] and sum to 1) to K - 1 vector of real values.
    Primarily borrowed from the STAN implementation.

    Parameters
    ----------
    eps : float, positive value
        A small value for numerical stability in invlogit.
    """

    name = "stickbreaking"

    def __init__(self, eps=0.0):
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
        return_expr = y.T

        # Add a test_value if there is not one already. Useful when operating
        # with theano.config.compute_test_value = 'off'
        if not hasattr(return_expr.tag, "test_value"):
            return_expr.tag.test_value = stickbreaking_forward_testval(x_)
        return return_expr

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
        return_expr = x.T

        # Add a test_value if there is not one already. Useful when operating
        # with theano.config.compute_test_value = 'off'
        if not hasattr(return_expr.tag, "test_value"):
            return_expr.tag.test_value =\
                stickbreaking_backward_testval(y_, self.eps)
        return return_expr

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

def stickbreaking_forward_testval(x_):
    """
    Creates the test value for the forward stickbreaking transformation.

    Parameters
    ----------
    x_ : same input as StickBreaking.forward. Most likely a theano Tensor.

    Returns
    -------
    test_value : ndarray.
        The result of applying the forward stickbreaking transformation to the
        test value of `x_`.
    """
    # Get the numeric version of x and transpose it
    x = get_single_numeric_obj(x_, name='x_').T

    # Define a numeric version of the logit transformation, since `logit()`
    # returns a theano expression instead of a  number or ndarray.
    def numeric_logit(value):
        return np.log(value / (1 - value))

    # Apply the forward stickbreaking transformation. Next 8 lines copied from
    # the forward method of the StickBreaking class. Simply changed theano
    # functions to numpy functions
    x0 = x[:-1]
    s = np.cumsum(x0[::-1], 0)[::-1] + x[-1]
    z = x0 / s
    Km1 = x.shape[0] - 1
    k = np.arange(Km1)[(slice(None), ) + (None, ) * (x.ndim - 1)]
    eq_share = numeric_logit(1. / (Km1 + 1 - k).astype(str(x_.dtype)))
    y = numeric_logit(z) - eq_share
    test_value = y.T

    return test_value


def stickbreaking_backward_testval(y_, eps):
    """
    Creates the test value for the backward stickbreaking transformation.

    Parameters
    ----------
    y_ : same input as StickBreaking.backward. Most likely a theano Tensor.
    eps : float.
        Determines the value used to prevent some sort of under- or over-flow.

    Returns
    -------
    test_value : ndarray.
        The result of applying the backward stickbreaking transformation to the
        test value of `y_`.
    """
    # Get the numeric version of y and transpose it
    y = get_single_numeric_obj(x_, name='x_').T

    # Define a numeric version of the logit transformation, since `logit()`
    # returns a theano expression instead of a  number or ndarray.
    def numeric_logit(value):
        return np.log(value / (1 - value))

    # Define a numeric version of the logistic transformation, since
    # `invlogit()` returns a theano expression instead of a  number or ndarray.
    def numeric_invlogit(value, eps):
        return (1 - 2 * eps) / (1 + np.exp(-x)) + eps

    # Apply the backward stickbreaking transformation. Next 8 lines copied from
    # the backward method of the StickBreaking class. Simply changed theano
    # functions to numpy functions
    Km1 = y.shape[0]
    k = np.arange(Km1)[(slice(None), ) + (None, ) * (y.ndim - 1)]
    eq_share = numeric_logit(1. / (Km1 + 1 - k).astype(str(y_.dtype)))
    z = numeric_invlogit(y + eq_share, eps)
    yl = np.concatenate([z, np.ones(y[:1].shape)])
    yu = np.concatenate([np.ones(y[:1].shape), 1 - z])
    S = np.cumprod(yu, 0)
    x = S * yl
    test_value = x.T

    return test_value


class Circular(Transform):
    """Transforms a linear space into a circular one.
    """
    name = "circular"

    def backward(self, y):
        return_expr = tt.arctan2(tt.sin(y), tt.cos(y))

        # Add a test_value if there is not one already. Useful when operating
        # with theano.config.compute_test_value = 'off'
        if not hasattr(return_expr.tag, "test_value"):
            numeric_y = get_single_numeric_obj(y, name='y')
            test_value = np.arctan2(np.sin(numeric_y), np.cos(numeric_y))
            return_expr.tag.test_value = test_value

        return return_expr

    def forward(self, x):
        # Note that all inputs should already have a test_value, so there is no
        # need to explicitly add such a value.
        return x

    def jacobian_det(self, x):
        return 0

circular = Circular()


def get_single_numeric_obj(source_obj, name=''):
    """
    Extract a single numeric object from the source_obj: typically a theano
    Tensor, TensorConstant, or TensorVariable.

    Parameters
    ----------
    source_obj : number, ndarray, Tensor, TensorConstant, or TensorVariable
    name : str.
        The name of the given source_obj.

    Returns
    -------
    numeric: Number or np.ndarray
        The .tag.test_value or value for the theano Tensor or TensorConstant
        respectively. Alternatively, returns the source_obj if
        `isinstance(source_obj, (Number, np.ndarray))`
    """ 
    if name == '':
        name = "source_obj"

    # Get the numeric version of the parameters
    if isinstance(source_obj, (Number, np.ndarray)):
        numeric = source_obj
    elif isinstance(source_obj, tt.TensorConstant):
        numeric = source_obj.value
    elif isinstance(source_obj, (tt.Tensor, tt.TensorVariable)):
        if not hasattr(source_obj.tag, "test_value"):
            raise ValueError("{}.tag.test_value does not exist.".format(name))
        numeric = source_obj.tag.test_value
    else:
        msg = ("'{}' is not a Number, ndarray, theano Tensor, TensorConstant, "
               "or TensorVariable.")
        raise ValueError(msg.format(name))

    return numeric
