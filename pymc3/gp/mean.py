import theano.tensor as tt
# import numpy as np

__all__ = ['Zero', 'Constant']

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
        