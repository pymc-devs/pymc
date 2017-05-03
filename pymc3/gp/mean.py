import numpy as np
import theano.tensor as tt

__all__ = ['Zero', 'Constant', 'Linear']


class ParameterizedFunction(object):
    R"""
    Base class for all functions.  The input_dim and active_dims parameters
    allow for combining functions which act on subspaces to be added and
    multiplied using Python built-in operators like "+" and "*".

    Parameters
    ----------
    input_dim : integer
        The number of input dimensions, or columns of X (or Z) the kernel will operate on.
    active_dims : A list of booleans whose length equals input_dim.
	The booleans indicate whether or not the covariance function operates
        over that dimension or column of X.
    """

    def __init__(self, input_dim, active_dims=None):
        self.input_dim = input_dim
        if active_dims is None:
            self.active_dims = np.arange(input_dim)
        else:
            self.active_dims = np.array(active_dims)
            if len(active_dims) != input_dim:
                raise ValueError("Length of active_dims must match input_dim")

    def __call__(self, X):
        R"""
        Evaluate the function.

        Parameters
        ----------
        X : The inputs to the function.
        """
        raise NotImplementedError

    def _slice(self, X):
        X = X[:, self.active_dims]
        return X

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)


class Mean(ParameterizedFunction):
    """
    Base class for mean functions
    """
    def __init__(self, input_dim, active_dims=None):
        super(Mean, self).__init__(input_dim, active_dims)

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
        X = self._slice(X)
        return tt.zeros(tt.stack([X.shape[0], 1]), dtype="float32")

class Constant(Mean):
    """
    Constant mean function for Gaussian process.

    Parameters
    ----------
    c : variable, array or integer
        Constant mean value
    """

    def __init__(self, input_dim, c=0, active_dims=None):
        super(Constant, self).__init__(input_dim, active_dims)
        self.c = c

    def __call__(self, X):
        X = self._slice(X)
        return tt.ones(tt.stack([X.shape[0], 1])) * self.c

class Linear(Mean):
    def __init__(self, input_dim, coeffs, intercept=0, active_dims=None):
        """
        Linear mean function for Gaussian process.

        Parameters
        ----------
        coeffs : variables
            Linear coefficients
        intercept : variable, array or integer
            Intercept for linear function (Defaults to zero)
        """
        super(Linear, self).__init__(input_dim, active_dims)
        self.b = intercept
        self.A = coeffs

    def __call__(self, X):
        X = self._slice(X)
        m = tt.dot(X, self.A) + self.b
        return tt.reshape(m, (X.shape[0], 1))

class Add(Mean):
    def __init__(self, first_mean, second_mean):
        Mean.__init__(self)
        self.m1 = first_mean
        self.m2 = second_mean

    def __call__(self, X):
        return tt.add(self.m1(X), self.m2(X))


class Prod(Mean):
    def __init__(self, first_mean, second_mean):
        Mean.__init__(self)
        self.m1 = first_mean
        self.m2 = second_mean

    def __call__(self, X):
        return tt.mul(self.m1(X), self.m2(X))

