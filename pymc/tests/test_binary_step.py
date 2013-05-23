"""Test the binary step method.
Assume we have a series of heads (0) or tail (1) experiments :
[0,0,1,1,0,0,0,1,1,0]
Assume that the coin is either fair (p=.5), or unfair (p!=.5).
We have two parameters:
 * fair : which can be True or False,
 * coin : the probability of tail on flipping the coin
"""

from numpy.testing import TestCase
import pymc
import numpy as np


class BinaryTestModel:
    series = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1])

    fair = pymc.Bernoulli('fair', p=.5, value=1)

    @pymc.deterministic
    def coin(fair=fair):
        if fair:
            return .5
        else:
            return .1
    # @stoch
    # def coin(value=.5, fair=fair):
    #     """Return the probability of a tail on flipping a coin."""
    #     if fair is True:
    #         return pymc.beta_like(value, 1e6, 1e6)
    #     else:
    #         return pymc.uniform_like(value, .3, .7)

    tail = pymc.Bernoulli('tail', p=coin, value=series, observed=True)


class TestBinary(TestCase):

    def test(self):
        S = pymc.MCMC(input=BinaryTestModel)
        S.sample(1000, 500, progress_bar=0)
        f = S.fair.trace()
        assert(1.0 * f.sum() / len(f) > .5)

if __name__ == '__main__':
    import unittest
    unittest.main()
