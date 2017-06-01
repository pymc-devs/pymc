import numpy as np
import theano.tensor as tt

__all__ = ['Zero', 'Constant', 'Linear']

"""
how far to go with this?
  - should i just keep as is?
  - plan to include basis functions that follow the func(x, args=(,)) form or mean function form?
    - is this form better/required for this to work cleanly?
  - is there a way to factor Combination, Add, Prod, __array_wrap__ out of cov.py and into ParameterizedFunction?
"""


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

    def __add__(self, other):
        return Add([self, other])

    def __mul__(self, other):
        return Prod([self, other])

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __array_wrap__(self, result):
        # can this be moved to parameterizedfunction?
        """
        Required to allow radd/rmul by numpy arrays.
        """
        r,c = result.shape
        A = np.zeros((r,c))
        for i in range(r):
            for j in range(c):
                A[i,j] = result[i,j].factor_list[1]
        if isinstance(result[0][0], Add):
            return result[0][0].factor_list[0] + A
        elif isinstance(result[0][0], Prod):
            return result[0][0].factor_list[0] * A
        else:
            raise RuntimeError(result[0][0])


class Combination(ParameterizedFunction):
    def __init__(self, factor_list):
        input_dim = np.max([factor.input_dim for factor in
                            filter(lambda x: isinstance(x, ParameterizedFunction), factor_list)])
        ParameterizedFunction.__init__(self, input_dim=input_dim)
        self.factor_list = []
        for factor in factor_list:
            if isinstance(factor, self.__class__):
                self.factor_list.extend(factor.factor_list)
            else:
                self.factor_list.append(factor)


class Add(Combination):
    def __call__(self, X):
        return reduce((lambda x, y: x + y),
                      [f(X) if isinstance(f, ParameterizedFunction) else f for f in self.factor_list])


class Prod(Combination):
    def __call__(self, X):
        return reduce((lambda x, y: x * y),
                      [f(X) if isinstance(f, ParameterizedFunction) else f for f in self.factor_list])


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

class Zero(Mean):
    def __call__(self, X):
        X = self._slice(X)
        return tt.zeros(tt.stack([X.shape[0], 1]), dtype="float32")

class Constant(Mean):
    """
    Constant mean function for Gaussian process.

    Parameters
    ----------
    c : variable, float, or integer
        Constant mean value
    """

    def __init__(self, input_dim, c=0.0, active_dims=None):
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



