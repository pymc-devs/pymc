import numpy as np
import theano.tensor as tt
from theano.scalar.basic_scipy import GammaLn, Psi, I0, I1
from theano import scalar

__all__ = ['gammaln', 'multigammaln', 'psi', 'i0', 'i1']

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

scalar_i0 = I0(scalar.upgrade_to_float, name='scalar_i0')
i0 = tt.Elemwise(scalar_i0, name='i0')

scalar_i1 = I1(scalar.upgrade_to_float, name='scalar_i1')
i1 = tt.Elemwise(scalar_i1, name='i1')
