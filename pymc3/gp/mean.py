import theano.tensor as tt
import numpy as np

__all__ = []

class Mean(object):
    """
    Base class for mean functions
    
    To implement a mean function, write the __call__ method. This takes a
    tensor X and returns a tensor m(X). In accordance with the GPflow
    standard, each row of X represents one datum, and each row of Y is computed
    independently for each row of X.
    MeanFunction classes can have parameters, see the Linear class for an
    example.
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
        return tt.zeros(tt.stack([tt.shape(X)[0], 1]), dtype='float32')
        
class Constant(Mean):

    def __init__(self, c=0):
        MeanFunction.__init__(self)
        self.c = c

    def __call__(self, X):
        shape = tt.stack([tt.shape(X)[0], 1])
        return tt.tile(tt.reshape(self.c, (1, -1)), shape)
        
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
        