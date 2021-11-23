#   Copyright 2020 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import aesara.tensor as at

__all__ = ["Zero", "Constant", "Linear"]


class Mean:
    R"""
    Base class for mean functions
    """

    def __call__(self, X):
        R"""
        Evaluate the mean function.

        Parameters
        ----------
        X: The training inputs to the mean function.
        """
        raise NotImplementedError

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Prod(self, other)


class Zero(Mean):
    R"""
    Zero mean function for Gaussian process.

    """

    def __call__(self, X):
        return at.alloc(0.0, X.shape[0])


class Constant(Mean):
    R"""
    Constant mean function for Gaussian process.

    Parameters
    ----------
    c: variable, array or integer
        Constant mean value
    """

    def __init__(self, c=0):
        super().__init__()
        self.c = c

    def __call__(self, X):
        return at.alloc(1.0, X.shape[0]) * self.c


class Linear(Mean):
    R"""
    Linear mean function for Gaussian process.

    Parameters
    ----------
    coeffs: variables
        Linear coefficients
    intercept: variable, array or integer
        Intercept for linear function (Defaults to zero)
    """

    def __init__(self, coeffs, intercept=0):
        super().__init__()
        self.b = intercept
        self.A = coeffs

    def __call__(self, X):
        return at.squeeze(at.dot(X, self.A) + self.b)


class Add(Mean):
    def __init__(self, first_mean, second_mean):
        super().__init__()
        self.m1 = first_mean
        self.m2 = second_mean

    def __call__(self, X):
        return at.add(self.m1(X), self.m2(X))


class Prod(Mean):
    def __init__(self, first_mean, second_mean):
        super().__init__()
        self.m1 = first_mean
        self.m2 = second_mean

    def __call__(self, X):
        return at.mul(self.m1(X), self.m2(X))
