import numpy as np
import theano.tensor as tt
from scipy import special
from theano.scalar.basic_scipy import GammaLn, Psi
from theano import scalar

__all__ = ['gammaln', 'multigammaln', 'psi', 'trigamma']

scalar_gammaln = GammaLn(scalar.upgrade_to_float, name='scalar_gammaln')
gammaln = tt.Elemwise(scalar_gammaln, name='gammaln')


def multigammaln(a, p):
    """Multivariate Log Gamma

    :Parameters:
        a : tensor like
        p : int degrees of freedom
            p > 0
    """
    i = tt.arange(1, p + 1)
    return (p * (p - 1) * tt.log(np.pi) / 4.
            + tt.sum(gammaln(a + (1. - i) / 2.), axis=0))

scalar_psi = Psi(scalar.upgrade_to_float, name='scalar_psi')
psi = tt.Elemwise(scalar_psi, name='psi')


class Trigamma(scalar.UnaryScalarOp):
    """
    Compute 2nd derivative of gammaln(x)
    """
    @staticmethod
    def st_impl(x):
        return special.polygamma(1, x)

    def impl(self, x):
        return Psi.st_impl(x)

    # def grad()  no gradient now

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

scalar_trigamma = Trigamma(scalar.upgrade_to_float, name='scalar_trigamma')
trigamma = tt.Elemwise(scalar_trigamma, name='trigamma')
