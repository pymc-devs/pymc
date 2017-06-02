import numpy as np
import theano.tensor as tt

__all__ = ['Zero', 'Constant', 'Linear']


class Mean(object):
    """
    Base class for mean functions
    """

    def __call__(self, X):
        R"""
        Evaluate the mean function.

        Parameters
        ----------
        X : The training inputs to the mean function.
        """
        raise NotImplementedError

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Prod(self, other)

class Zero(Mean):
    def __call__(self, X):
        return tt.zeros(tt.stack([X.shape[0], 1]), dtype="float32")

class Constant(Mean):
    """
    Constant mean function for Gaussian process.

    Parameters
    ----------
    c : variable, float, or integer
        Constant mean value
    """

    def __init__(self, c=0.0):
        super(Constant, self).__init__()
        self.c = c

    def __call__(self, X):
        return tt.ones(tt.stack([X.shape[0], 1])) * self.c


class Linear(Mean):
    def __init__(self, coeffs, intercept=0):
        """
        Linear mean function for Gaussian process.

        Parameters
        ----------
        coeffs : variables
            Linear coefficients
        intercept : variable, array or integer
            Intercept for linear function (Defaults to zero)
        """
        super(Linear, self).__init__()
        self.b = intercept
        self.A = coeffs

    def __call__(self, X):
        m = tt.dot(X, self.A) + self.b
        return tt.reshape(m, (X.shape[0], 1))


class Add(Mean):
    def __init__(self, first_mean, second_mean):
        super(Add, self).__init__()
        self.m1 = first_mean
        self.m2 = second_mean

    def __call__(self, X):
        return tt.add(self.m1(X), self.m2(X))


class Prod(Mean):
    def __init__(self, first_mean, second_mean):
        super(Prod, self).__init__()
        self.m1 = first_mean
        self.m2 = second_mean

    def __call__(self, X):
        return tt.mul(self.m1(X), self.m2(X))
